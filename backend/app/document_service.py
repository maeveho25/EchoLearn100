# backend/app/document_service.py
import boto3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO

load_dotenv()

class DocumentService:
    """Service for managing documents stored in S3 and DynamoDB"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.bedrock_client = boto3.client('bedrock-runtime')
        
        # Configuration
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.documents_table_name = os.getenv('DYNAMODB_DOCUMENTS_TABLE', 'EchoLearn-Documents')
        self.sessions_table_name = os.getenv('DYNAMODB_SESSIONS_TABLE', 'EchoLearn-Sessions')
        
        # Tables
        self.documents_table = self.dynamodb.Table(self.documents_table_name)
        self.sessions_table = self.dynamodb.Table(self.sessions_table_name)
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict]:
        """Get document metadata from DynamoDB"""
        try:
            response = self.documents_table.get_item(
                Key={'DocumentId': document_id}
            )
            return response.get('Item')
        except ClientError as e:
            print(f"❌ Error getting document {document_id}: {e}")
            return None
    
    def get_document_content_from_s3(self, document_id: str) -> Optional[str]:
        """Get document text content from S3 or DynamoDB"""
        try:
            # First try to get from DynamoDB (faster)
            doc_metadata = self.get_document_by_id(document_id)
            if not doc_metadata:
                return None
            
            # If text is stored in DynamoDB, return it
            if 'Text' in doc_metadata and doc_metadata['Text']:
                print(f"✅ Retrieved text from DynamoDB for {document_id}")
                return doc_metadata['Text']
            
            # If not in DynamoDB, try to get from S3
            s3_key = doc_metadata.get('S3Key')
            if not s3_key:
                print(f"❌ No S3 key found for document {document_id}")
                return None
            
            return self.extract_text_from_s3(s3_key)
            
        except Exception as e:
            print(f"❌ Error getting document content for {document_id}: {e}")
            return None
    
    def extract_text_from_s3(self, s3_key: str) -> Optional[str]:
        """Extract text from PDF stored in S3"""
        try:
            # Download PDF from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            pdf_content = response['Body'].read()
            
            # Extract text using PyPDF2
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text_content = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text_content += page_text + "\n"
                print(f"📄 Extracted {len(page_text)} characters from page {page_num + 1}")
            
            if text_content.strip():
                print(f"✅ Successfully extracted text from S3: {len(text_content)} characters")
                return text_content
            else:
                print(f"❌ No text could be extracted from S3 file: {s3_key}")
                return None
                
        except Exception as e:
            print(f"❌ Error extracting text from S3 {s3_key}: {e}")
            return None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents with metadata"""
        try:
            documents = []
            
            # Scan with pagination
            response = self.documents_table.scan()
            documents.extend(response['Items'])
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.documents_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                documents.extend(response['Items'])
            
            # Sort by creation date (newest first)
            documents.sort(key=lambda x: x.get('ProcessedAt', ''), reverse=True)
            
            # Add summary info
            for doc in documents:
                doc['summary'] = self.generate_document_summary(doc)
            
            return documents
            
        except ClientError as e:
            print(f"❌ Error getting all documents: {e}")
            return []
    
    def generate_document_summary(self, document: Dict) -> Dict:
        """Generate a summary of document metrics"""
        try:
            return {
                'word_count': document.get('WordCount', 0),
                'page_count': document.get('PageCount', 0),
                'subject': document.get('subject', 'Unknown'),
                'difficulty': document.get('difficulty_level', 5),
                'file_size_mb': round(document.get('FileSize', 0) / (1024 * 1024), 2),
                'has_embeddings': bool(document.get('Embeddings')),
                'processing_status': document.get('Status', 'unknown'),
                'processed_at': document.get('ProcessedAt', ''),
                's3_uri': document.get('S3Uri', ''),
                'preview': document.get('Text', '')[:300] + '...' if document.get('Text') else 'No preview available'
            }
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return {
                'word_count': 0,
                'page_count': 0,
                'subject': 'Unknown',
                'difficulty': 5,
                'error': str(e)
            }
    
    def search_documents(self, query: str, subject: str = None, difficulty: int = None) -> List[Dict]:
        """Search documents by content, subject, or difficulty"""
        try:
            documents = self.get_all_documents()
            
            filtered_docs = []
            query_lower = query.lower() if query else ''
            
            for doc in documents:
                # Text search
                text_match = not query or query_lower in doc.get('Text', '').lower()
                
                # Filename search
                filename_match = not query or query_lower in doc.get('FileName', '').lower()
                
                # Subject filter
                subject_match = not subject or doc.get('subject', '').lower() == subject.lower()
                
                # Difficulty filter
                difficulty_match = not difficulty or doc.get('difficulty_level', 5) == difficulty
                
                if (text_match or filename_match) and subject_match and difficulty_match:
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def get_document_sessions(self, document_id: str) -> List[Dict]:
        """Get all learning sessions for a document"""
        try:
            # Use scan with filter (in production, use GSI)
            response = self.sessions_table.scan(
                FilterExpression='DocumentId = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id}
            )
            
            sessions = response['Items']
            sessions.sort(key=lambda x: x.get('CreatedAt', ''), reverse=True)
            
            return sessions
            
        except ClientError as e:
            print(f"❌ Error getting sessions for document {document_id}: {e}")
            return []
    
    def update_document_analysis(self, document_id: str, analysis_data: Dict) -> bool:
        """Update document with additional analysis data"""
        try:
            # Build update expression
            update_expression = "SET "
            expression_values = {}
            
            for key, value in analysis_data.items():
                update_expression += f"{key} = :{key}, "
                expression_values[f":{key}"] = value
            
            # Add updated timestamp
            update_expression += "UpdatedAt = :updated_at"
            expression_values[":updated_at"] = datetime.utcnow().isoformat()
            
            self.documents_table.update_item(
                Key={'DocumentId': document_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            
            print(f"✅ Updated document {document_id} with analysis data")
            return True
            
        except ClientError as e:
            print(f"❌ Error updating document {document_id}: {e}")
            return False
    
    def generate_document_embeddings(self, document_id: str) -> bool:
        """Generate embeddings for a document if not already present"""
        try:
            # Get document
            doc = self.get_document_by_id(document_id)
            if not doc:
                return False
            
            # Check if embeddings already exist
            if doc.get('Embeddings'):
                print(f"✅ Document {document_id} already has embeddings")
                return True
            
            # Get text content
            text_content = self.get_document_content_from_s3(document_id)
            if not text_content:
                print(f"❌ No text content found for document {document_id}")
                return False
            
            # Generate embeddings using Bedrock Titan
            text_for_embedding = text_content[:7000]  # Limit for token constraints
            
            response = self.bedrock_client.invoke_model(
                modelId='amazon.titan-embed-text-v1',
                body=json.dumps({
                    'inputText': text_for_embedding
                })
            )
            
            result = json.loads(response['body'].read())
            embeddings = result.get('embedding', [])
            
            if embeddings:
                # Update document with embeddings
                self.update_document_analysis(document_id, {'Embeddings': embeddings})
                print(f"✅ Generated and stored embeddings for document {document_id}")
                return True
            else:
                print(f"❌ Failed to generate embeddings for document {document_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating embeddings for document {document_id}: {e}")
            return False
    
    def get_document_analytics(self, document_id: str) -> Dict:
        """Get analytics for a document"""
        try:
            # Get document metadata
            doc = self.get_document_by_id(document_id)
            if not doc:
                return {}
            
            # Get sessions
            sessions = self.get_document_sessions(document_id)
            
            # Calculate analytics
            total_sessions = len(sessions)
            question_sessions = [s for s in sessions if s.get('Questions')]
            speech_sessions = [s for s in sessions if s.get('Type') == 'speech_analysis']
            
            avg_similarity = 0
            if speech_sessions:
                similarities = [s.get('SimilarityScore', 0) for s in speech_sessions]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            
            return {
                'document_id': document_id,
                'filename': doc.get('FileName', 'Unknown'),
                'total_sessions': total_sessions,
                'question_sessions': len(question_sessions),
                'speech_sessions': len(speech_sessions),
                'average_similarity_score': round(avg_similarity, 2),
                'last_accessed': sessions[0].get('CreatedAt') if sessions else None,
                'document_metrics': self.generate_document_summary(doc)
            }
            
        except Exception as e:
            print(f"❌ Error getting analytics for document {document_id}: {e}")
            return {}

# Singleton instance
document_service = DocumentService()