import os
import boto3
import uuid
import PyPDF2
import json  # Missing import!
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
        print(f"📊 Content length: {len(file_content) if file_content else 'N/A'}")
        
        # Enhanced validation
        if file_content is None:
            raise Exception("File content is None - cannot upload to S3")
        
        if not isinstance(file_content, bytes):
            raise Exception(f"Expected bytes, got {type(file_content)}")
        
        if len(file_content) == 0:
            raise Exception("File content is empty")
        
        if not self.s3_enabled:
            raise Exception("S3 not configured - check S3_BUCKET_NAME environment variable")
        
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
    
    def extract_text_from_pdf(self, file_content: bytes) -> Dict:
        """Extract text from PDF content"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_content = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text_content += page_text + "\n"
                print(f"Extracted {len(page_text)} characters from page {page_num + 1}")
            
            if not text_content.strip():
                raise Exception("No text could be extracted from the PDF")
            
            word_count = len(text_content.split())
            page_count = len(pdf_reader.pages)
            
            print(f"Total extracted: {len(text_content)} characters, {word_count} words, {page_count} pages")
            
            return {
                'text': text_content,
                'word_count': word_count,
                'page_count': page_count
            }
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

# Singleton instance
pdf_service = PDFService()