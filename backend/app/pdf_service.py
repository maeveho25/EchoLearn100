import os
import boto3
import uuid
import PyPDF2
from io import BytesIO
from datetime import datetime
from botocore.exceptions import ClientError
from typing import Dict, Optional

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
        """Upload PDF file to S3 and return metadata"""
        try:
            # Generate unique S3 key
            file_extension = filename.split('.')[-1].lower()
            if file_extension != 'pdf':
                raise ValueError("Only PDF files are allowed")
            
            s3_key = f"uploads/{uuid.uuid4()}-{filename}"
            
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
            
            return {
                'bucket': self.bucket_name,
                's3_key': s3_key,
                's3_uri': f"s3://{self.bucket_name}/{s3_key}",
                'file_size': len(file_content)
            }
            
        except Exception as e:
            raise Exception(f"Failed to upload PDF to S3: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> Dict:
        """Extract text and metadata from PDF content"""
        try:
            # Create PDF reader from bytes
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        full_text += page_text + "\n\n"
                        page_texts.append({
                            'page_number': page_num + 1,
                            'text': page_text.strip(),
                            'word_count': len(page_text.split())
                        })
                except Exception as page_error:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {page_error}")
                    continue
            
            if not full_text.strip():
                raise Exception("No text could be extracted from PDF. The PDF might be image-based or corrupted.")
            
            # Calculate statistics
            word_count = len(full_text.split())
            char_count = len(full_text)
            page_count = len(pdf_reader.pages)
            
            # Detect subject/topic (simple keyword-based detection)
            subject = self.detect_subject(full_text)
            
            return {
                'text': full_text.strip(),
                'page_count': page_count,
                'word_count': word_count,
                'char_count': char_count,
                'pages': page_texts,
                'subject': subject,
                'extraction_success': True
            }
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def detect_subject(self, text: str) -> str:
        """Simple subject detection based on keywords"""
        text_lower = text.lower()
        
        # Define keyword patterns for different subjects
        subjects = {
            'Machine Learning': [
                'machine learning', 'neural network', 'deep learning', 'algorithm', 
                'supervised learning', 'unsupervised learning', 'classification', 
                'regression', 'training data', 'model', 'prediction'
            ],
            'Data Science': [
                'data science', 'data analysis', 'statistics', 'pandas', 'numpy',
                'visualization', 'dataset', 'correlation', 'hypothesis'
            ],
            'Computer Science': [
                'computer science', 'programming', 'software', 'algorithm',
                'data structure', 'complexity', 'coding'
            ],
            'Mathematics': [
                'mathematics', 'calculus', 'algebra', 'geometry', 'theorem',
                'equation', 'formula', 'proof'
            ],
            'Physics': [
                'physics', 'quantum', 'mechanics', 'thermodynamics', 'relativity',
                'energy', 'force', 'motion'
            ],
            'Biology': [
                'biology', 'cell', 'dna', 'protein', 'organism', 'evolution',
                'genetics', 'ecosystem'
            ]
        }
        
        # Count keyword matches for each subject
        subject_scores = {}
        for subject, keywords in subjects.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                subject_scores[subject] = score
        
        # Return subject with highest score, or 'General' if no clear match
        if subject_scores:
            return max(subject_scores, key=subject_scores.get)
        else:
            return 'General'
    
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