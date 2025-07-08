import boto3
import json
from datetime import datetime
from typing import List, Dict, Optional
from botocore.exceptions import ClientError

class DynamoDBService:
    """Service class for DynamoDB operations"""
    
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.documents_table = self.dynamodb.Table('EchoLearn-Documents')
        self.sessions_table = self.dynamodb.Table('EchoLearn-Sessions')
        self.analytics_table = self.dynamodb.Table('EchoLearn-Analytics')
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Retrieve a document by ID"""
        try:
            response = self.documents_table.get_item(
                Key={'DocumentId': document_id}
            )
            return response.get('Item')
        except ClientError as e:
            print(f"Error getting document {document_id}: {e}")
            return None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents with pagination"""
        try:
            documents = []
            
            # Scan with pagination
            response = self.documents_table.scan()
            documents.extend(response['Items'])
            
            # Handle pagination if needed
            while 'LastEvaluatedKey' in response:
                response = self.documents_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                documents.extend(response['Items'])
            
            # Sort by creation date (newest first)
            documents.sort(key=lambda x: x.get('CreatedAt', ''), reverse=True)
            return documents
            
        except ClientError as e:
            print(f"Error getting all documents: {e}")
            return []
    
    def insert_document(self, document_data: Dict) -> str:
        """Insert a new document"""
        try:
            document_data['CreatedAt'] = datetime.utcnow().isoformat()
            self.documents_table.put_item(Item=document_data)
            return document_data['DocumentId']
        except ClientError as e:
            print(f"Error inserting document: {e}")
            raise
    
    def update_document(self, document_id: str, updates: Dict) -> bool:
        """Update document fields"""
        try:
            # Build update expression
            update_expression = "SET "
            expression_values = {}
            
            for key, value in updates.items():
                update_expression += f"{key} = :{key}, "
                expression_values[f":{key}"] = value
            
            # Remove trailing comma
            update_expression = update_expression.rstrip(', ')
            
            self.documents_table.update_item(
                Key={'DocumentId': document_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            return True
            
        except ClientError as e:
            print(f"Error updating document {document_id}: {e}")
            return False
    
    def search_documents(self, query: str, subject: str = None) -> List[Dict]:
        """Search documents by text content or subject"""
        try:
            documents = self.get_all_documents()
            
            # Simple text search (in production, use ElasticSearch or similar)
            filtered_docs = []
            query_lower = query.lower()
            
            for doc in documents:
                # Search in text content
                text_match = query_lower in doc.get('Text', '').lower()
                
                # Search in filename
                filename_match = query_lower in doc.get('FileName', '').lower()
                
                # Subject filter
                subject_match = not subject or doc.get('Subject', '').lower() == subject.lower()
                
                if (text_match or filename_match) and subject_match:
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def save_session(self, session_data: Dict) -> str:
        """Save a learning session"""
        try:
            session_data['CreatedAt'] = datetime.utcnow().isoformat()
            self.sessions_table.put_item(Item=session_data)
            return session_data['SessionId']
        except ClientError as e:
            print(f"Error saving session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a learning session"""
        try:
            response = self.sessions_table.get_item(
                Key={'SessionId': session_id}
            )
            return response.get('Item')
        except ClientError as e:
            print(f"Error getting session {session_id}: {e}")
            return None
    
    def get_user_sessions(self, document_id: str) -> List[Dict]:
        """Get all sessions for a document"""
        try:
            # In production, you'd want a GSI on DocumentId
            response = self.sessions_table.scan(
                FilterExpression='DocumentId = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id}
            )
            
            sessions = response['Items']
            sessions.sort(key=lambda x: x.get('CreatedAt', ''), reverse=True)
            return sessions
            
        except ClientError as e:
            print(f"Error getting sessions for document {document_id}: {e}")
            return []
    
    def save_analytics(self, user_id: str, event_type: str, data: Dict) -> bool:
        """Save analytics event"""
        try:
            analytics_item = {
                'UserId': user_id,
                'Timestamp': datetime.utcnow().isoformat(),
                'EventType': event_type,
                'Data': data
            }
            
            self.analytics_table.put_item(Item=analytics_item)
            return True
            
        except ClientError as e:
            print(f"Error saving analytics: {e}")
            return False

db_service = DynamoDBService()