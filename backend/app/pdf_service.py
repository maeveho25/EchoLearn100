import os
import boto3
import uuid
import PyPDF2
from io import BytesIO
from datetime import datetime
from botocore.exceptions import ClientError
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class PDFService:
    """Service for handling PDF uploads and processing"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bedrock_client = boto3.client('bedrock-runtime')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        
        # Temporary: Don't fail if bucket not configured
        if not self.bucket_name:
            print("⚠️ S3_BUCKET_NAME not configured - S3 features disabled")
            self.s3_enabled = False
        else:
            print(f"✅ S3 bucket configured: {self.bucket_name}")
            self.s3_enabled = True
    
    def upload_pdf_to_s3(self, file_content: bytes, filename: str) -> Dict:
        """Upload PDF file to S3 with detailed debugging"""
        
        print(f"🔍 PDF Service upload_pdf_to_s3 called")
        print(f"📄 Filename: {filename}")
        print(f"📊 Content type: {type(file_content)}")
        print(f"📊 Content is None: {file_content is None}")
        print(f"📊 Content length: {len(file_content) if file_content else 'N/A'}")
        
        # Enhanced validation
        if file_content is None:
            raise Exception("File content is None - cannot upload to S3")
        
        if not isinstance(file_content, bytes):
            raise Exception(f"Expected bytes, got {type(file_content)}")
        
        if len(file_content) == 0:
            raise Exception("File content is empty")
        
        try:
            # Generate unique S3 key
            file_extension = filename.split('.')[-1].lower()
            if file_extension != 'pdf':
                raise ValueError("Only PDF files are allowed")
            
            s3_key = f"uploads/{uuid.uuid4()}-{filename}"
            
            print(f"📤 Uploading to S3: bucket={self.bucket_name}, key={s3_key}")
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_content,
                ContentType='application/pdf',
                Metadata={
                    'original_filename': filename,
                    'upload_timestamp': datetime.utcnow().isoformat()
                }
            )
            
            print(f"✅ S3 upload successful")
            
            return {
                'bucket': self.bucket_name,
                's3_key': s3_key,
                's3_uri': f"s3://{self.bucket_name}/{s3_key}",
                'file_size': len(file_content)
            }
            
        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
            raise Exception(f"Failed to upload PDF to S3: {str(e)}")
    
    def generate_embeddings(self, text: str) -> Optional[list]:
        """Generate embeddings using AWS Bedrock Titan"""
        try:
            # Limit text to avoid token limits
            text_for_embedding = text[:7000]
            
            response = self.bedrock_client.invoke_model(
                modelId='amazon.titan-embed-text-v1',
                body=json.dumps({
                    'inputText': text_for_embedding
                })
            )
            
            result = json.loads(response['body'].read())
            embeddings = result.get('embedding', [])
            
            if embeddings:
                print(f"✅ Generated embeddings: {len(embeddings)} dimensions")
                return embeddings
            else:
                print("⚠️ No embeddings returned from Titan")
                return None
                
        except Exception as e:
            print(f"⚠️ Failed to generate embeddings: {e}")
            return None
    
    def estimate_difficulty(self, text: str, word_count: int) -> int:
        """Estimate difficulty level (1-10) based on content analysis"""
        try:
            # Simple heuristic based on vocabulary complexity and length
            
            # Check for complex terms
            complex_indicators = [
                'algorithm', 'optimization', 'implementation', 'methodology',
                'theoretical', 'empirical', 'statistical', 'computational',
                'sophisticated', 'comprehensive', 'fundamental', 'advanced'
            ]
            
            beginner_indicators = [
                'introduction', 'basic', 'simple', 'easy', 'beginner',
                'overview', 'primer', 'fundamentals', 'getting started'
            ]
            
            text_lower = text.lower()
            
            complex_score = sum(1 for term in complex_indicators if term in text_lower)
            beginner_score = sum(1 for term in beginner_indicators if term in text_lower)
            
            # Base difficulty on word count
            if word_count < 500:
                base_difficulty = 3
            elif word_count < 1500:
                base_difficulty = 5
            elif word_count < 3000:
                base_difficulty = 7
            else:
                base_difficulty = 8
            
            # Adjust based on vocabulary
            if beginner_score > complex_score:
                base_difficulty = max(1, base_difficulty - 2)
            elif complex_score > beginner_score:
                base_difficulty = min(10, base_difficulty + 1)
            
            return base_difficulty
            
        except Exception:
            return 5  # Default medium difficulty

# Singleton instance
pdf_service = PDFService()