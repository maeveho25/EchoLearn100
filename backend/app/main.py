# backend/app/main.py
# EchoLearn API - Complete Production Implementation

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
import uuid
import asyncio
import re
import numpy as np
import time
import urllib.request
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import PyPDF2
from io import BytesIO
from typing import Dict, List, Optional, Any

# Load environment variables
load_dotenv()

app = FastAPI(
    title="EchoLearn API - Production Ready",
    description="AI Learning Assistant with AWS Bedrock and real-time speech analysis",
    version="4.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
AWS_REGION = os.getenv('AWS_REGION', 'ap-southeast-2')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
TITAN_EMBED_MODEL_ID = os.getenv('TITAN_EMBED_MODEL', 'amazon.titan-embed-text-v1')
DOCUMENTS_TABLE = os.getenv('DYNAMODB_DOCUMENTS_TABLE', 'EchoLearn-Documents')
SESSIONS_TABLE = os.getenv('DYNAMODB_SESSIONS_TABLE', 'EchoLearn-Sessions')

# Settings
CLAUDE_TEMPERATURE = float(os.getenv('CLAUDE_TEMPERATURE', '0.7'))
CLAUDE_MAX_TOKENS = int(os.getenv('CLAUDE_MAX_TOKENS', '2000'))
MAX_PDF_SIZE_MB = int(os.getenv('MAX_PDF_SIZE_MB', '10'))
ENABLE_DEBUG_LOGGING = os.getenv('ENABLE_DEBUG_LOGGING', 'true').lower() == 'true'

# Validate required environment variables
required_env_vars = ['S3_BUCKET_NAME', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

print(f"🚀 EchoLearn API v4.0.0 - Production Mode")
print(f"   AWS Region: {AWS_REGION}")
print(f"   Claude Model: {BEDROCK_MODEL_ID}")
print(f"   S3 Bucket: {S3_BUCKET_NAME}")
print(f"   Documents Table: {DOCUMENTS_TABLE}")
print(f"   Sessions Table: {SESSIONS_TABLE}")

# Initialize AWS clients
try:
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
    transcribe_client = boto3.client('transcribe', region_name=AWS_REGION)
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    
    # Test connection
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    
    print(f"✅ AWS connected successfully")
    print(f"   Account: {identity.get('Account')}")
    
except Exception as e:
    print(f"❌ AWS connection failed: {e}")
    raise RuntimeError("AWS connection failed. Please check your credentials and configuration.")

# ================== UTILITY FUNCTIONS ==================

def debug_log(message: str):
    """Debug logging function"""
    if ENABLE_DEBUG_LOGGING:
        print(f"🔍 DEBUG: {message}")

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if isinstance(value, str):
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return default

async def get_document_content(document_id: str) -> Dict:
    """Get document content from DynamoDB"""
    try:
        def get_doc():
            documents_table = dynamodb.Table(DOCUMENTS_TABLE)
            response = documents_table.get_item(Key={'DocumentId': document_id})
            return response.get('Item')
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_doc)
        
    except Exception as e:
        debug_log(f"Error getting document {document_id}: {e}")
        return None

async def transcribe_audio_with_aws(audio_s3_uri: str) -> str:
    """Transcribe audio using AWS Transcribe"""
    try:
        # Create unique job name
        job_name = f"echolearn-{str(uuid.uuid4())[:8]}-{int(time.time())}"
        
        def start_transcription():
            return transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': audio_s3_uri},
                MediaFormat='wav',
                LanguageCode='en-US',
                Settings={
                    'ShowSpeakerLabels': False,
                    'MaxSpeakerLabels': 1,
                    'VocabularyFilterMethod': 'remove'
                }
            )
        
        loop = asyncio.get_event_loop()
        transcribe_response = await loop.run_in_executor(None, start_transcription)
        debug_log(f"Started transcription job: {job_name}")
        
        # Poll for completion
        max_wait_time = 180  # 3 minutes
        wait_interval = 3
        waited = 0
        
        while waited < max_wait_time:
            await asyncio.sleep(wait_interval)
            waited += wait_interval
            
            def check_status():
                return transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            
            status_response = await loop.run_in_executor(None, check_status)
            job_status = status_response['TranscriptionJob']['TranscriptionJobStatus']
            
            debug_log(f"Transcription status: {job_status} (waited {waited}s)")
            
            if job_status == 'COMPLETED':
                transcript_uri = status_response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                
                # Get transcript
                def get_transcript():
                    with urllib.request.urlopen(transcript_uri) as response:
                        return json.loads(response.read().decode())
                
                transcript_data = await loop.run_in_executor(None, get_transcript)
                
                if transcript_data.get('results', {}).get('transcripts'):
                    transcript_text = transcript_data['results']['transcripts'][0]['transcript']
                    debug_log(f"Transcription completed: '{transcript_text}'")
                    
                    # Clean up job
                    def cleanup():
                        try:
                            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
                        except:
                            pass
                    
                    await loop.run_in_executor(None, cleanup)
                    return transcript_text
                else:
                    raise Exception("No transcript found in results")
                    
            elif job_status == 'FAILED':
                failure_reason = status_response['TranscriptionJob'].get('FailureReason', 'Unknown')
                raise Exception(f"Transcription failed: {failure_reason}")
        
        raise Exception(f"Transcription timeout after {max_wait_time} seconds")
        
    except Exception as e:
        debug_log(f"AWS Transcribe error: {e}")
        raise Exception(f"Speech transcription failed: {str(e)}")

async def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity using embeddings"""
    try:
        # Generate embeddings for both texts
        embedding1 = await generate_embeddings(text1)
        embedding2 = await generate_embeddings(text2)
        
        if not embedding1 or not embedding2:
            return 0.0
        
        # Calculate cosine similarity
        def cosine_similarity(vec_a, vec_b):
            a_np = np.array(vec_a, dtype=np.float32)
            b_np = np.array(vec_b, dtype=np.float32)
            
            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        
        similarity = cosine_similarity(embedding1, embedding2)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        
    except Exception as e:
        debug_log(f"Similarity calculation error: {e}")
        return 0.0

async def generate_ai_feedback(student_answer: str, document_text: str, similarity_score: float) -> str:
    """Generate personalized feedback using Claude"""
    try:
        feedback_prompt = f"""You are an AI tutor providing constructive feedback to a student. Here's the context:

Study Material: {document_text[:1000]}...

Student's Answer: "{student_answer}"

Semantic Similarity Score: {similarity_score:.1f} (0.0 = no similarity, 1.0 = perfect match)

Please provide helpful feedback that includes:
1. What the student understood correctly
2. Areas that need improvement or clarification
3. Specific suggestions for better understanding
4. Encouragement and next steps

Keep your feedback:
- Constructive and supportive
- Specific and actionable
- Appropriate for the similarity score
- Encouraging regardless of performance
- Under 200 words

Format as plain text, no special formatting."""
        
        def call_claude():
            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 400,
                    'temperature': 0.7,
                    'messages': [{'role': 'user', 'content': feedback_prompt}]
                })
            )
            return json.loads(response['body'].read())
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, call_claude)
        
        return result['content'][0]['text'].strip()
        
    except Exception as e:
        debug_log(f"AI feedback generation error: {e}")
        return f"I can see you mentioned: '{student_answer}'. This shows engagement with the material. Keep studying and try to elaborate more on the key concepts!"

async def analyze_document_with_claude(text_content: str) -> Dict:
    """Analyze document using Claude Bedrock"""
    
    analysis_text = text_content[:3000]
    
    analysis_prompt = f"""Analyze this educational document and provide a comprehensive analysis in JSON format:

DOCUMENT TEXT:
{analysis_text}

Please analyze and return ONLY a JSON object with these exact fields:
{{
    "subject": "primary academic subject (e.g., computer_science, mathematics, physics, chemistry, biology, business, history, literature, psychology, engineering, medicine, law, general)",
    "subject_confidence": 0.95,
    "difficulty_level": 7,
    "key_topics": ["list", "of", "main", "topics", "covered"],
    "educational_level": "target level (elementary, middle_school, high_school, undergraduate, graduate, professional)",
    "content_type": "type (textbook, lecture_notes, research_paper, tutorial, reference, exam_prep)",
    "summary": "brief 2-sentence summary of content and purpose"
}}

Return only the JSON object, no other text."""
    
    try:
        def call_bedrock():
            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 1000,
                    'temperature': 0.3,
                    'messages': [{'role': 'user', 'content': analysis_prompt}]
                })
            )
            return json.loads(response['body'].read())
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, call_bedrock)
        
        ai_response = result['content'][0]['text']
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        
        if json_match:
            analysis_data = json.loads(json_match.group())
            analysis_data['analysis_method'] = 'claude_bedrock'
            analysis_data['analyzed_at'] = datetime.utcnow().isoformat()
            return analysis_data
        else:
            raise Exception("Could not parse JSON from Claude response")
            
    except Exception as e:
        debug_log(f"Claude analysis failed: {e}")
        return {
            "subject": "general",
            "subject_confidence": 0.5,
            "difficulty_level": 5,
            "key_topics": ["document analysis failed"],
            "educational_level": "unknown",
            "content_type": "unknown",
            "summary": "Analysis failed due to technical issues. Document processing completed successfully.",
            "analysis_method": "fallback",
            "error": str(e),
            "analyzed_at": datetime.utcnow().isoformat()
        }

async def generate_questions_with_claude(text_content: str, difficulty: int, num_questions: int, question_type: str) -> List[Dict]:
    """Generate questions using Claude Bedrock"""
    
    if question_type == "multiple_choice":
        prompt = f"""Based on the following educational content, generate {num_questions} multiple choice questions at difficulty level {difficulty}/10.

Content:
{text_content[:3000]}

Please format your response as a valid JSON array with this exact structure:
[
    {{
        "question": "What is the main concept discussed in the content?",
        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
        "correct_answer": "A",
        "explanation": "Brief explanation of why this is correct",
        "difficulty": {difficulty},
        "type": "multiple_choice"
    }}
]

Requirements:
- Questions should test understanding, not just memorization
- Options should be plausible but clearly distinguishable
- Include brief explanations for correct answers
- Adjust complexity based on difficulty level (1=basic, 10=advanced)
- Return ONLY the JSON array, no additional text"""

    elif question_type == "short_answer":
        prompt = f"""Based on the following educational content, generate {num_questions} short answer questions at difficulty level {difficulty}/10.

Content:
{text_content[:3000]}

Please format your response as a valid JSON array with this exact structure:
[
    {{
        "question": "Explain the main concept discussed in the content.",
        "expected_answer": "Key points that should be covered in a good answer",
        "keywords": ["keyword1", "keyword2", "keyword3"],
        "difficulty": {difficulty},
        "type": "short_answer",
        "max_length": 200
    }}
]

Requirements:
- Questions should encourage critical thinking and application
- Expected answers should be comprehensive but concise
- Include 3-5 key keywords that indicate understanding
- Adjust complexity based on difficulty level
- Return ONLY the JSON array, no additional text"""

    else:  # mixed
        prompt = f"""Based on the following educational content, generate {num_questions} mixed questions (both multiple choice and short answer) at difficulty level {difficulty}/10.

Content:
{text_content[:3000]}

Please format your response as a valid JSON array mixing both question types:
[
    {{
        "question": "Multiple choice question text?",
        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
        "correct_answer": "A",
        "explanation": "Brief explanation",
        "difficulty": {difficulty},
        "type": "multiple_choice"
    }},
    {{
        "question": "Short answer question text?",
        "expected_answer": "Expected answer content",
        "keywords": ["keyword1", "keyword2", "keyword3"],
        "difficulty": {difficulty},
        "type": "short_answer",
        "max_length": 150
    }}
]

Return ONLY the JSON array, no additional text."""

    try:
        def call_claude():
            response = bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 2000,
                    'temperature': CLAUDE_TEMPERATURE,
                    'messages': [{'role': 'user', 'content': prompt}]
                })
            )
            return json.loads(response['body'].read())
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, call_claude)
        
        ai_response = result['content'][0]['text']
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
        
        if json_match:
            questions = json.loads(json_match.group())
            
            # Add IDs and metadata
            for i, q in enumerate(questions):
                q['id'] = str(uuid.uuid4())
                q['generated_at'] = datetime.utcnow().isoformat()
                
                # Validate question structure
                if question_type == "multiple_choice":
                    required_fields = ['question', 'options', 'correct_answer']
                elif question_type == "short_answer":
                    required_fields = ['question', 'expected_answer', 'keywords']
                else:
                    required_fields = ['question', 'type']
                
                missing_fields = [field for field in required_fields if field not in q]
                if missing_fields:
                    raise Exception(f"Question {i+1} missing fields: {missing_fields}")
            
            debug_log(f"Generated {len(questions)} questions successfully")
            return questions
        else:
            raise Exception("Could not parse JSON from Claude response")
            
    except Exception as e:
        debug_log(f"Claude question generation failed: {e}")
        raise Exception(f"AI question generation failed: {str(e)}")

async def generate_embeddings(text_content: str) -> List[float]:
    """Generate embeddings using Bedrock Titan"""
    try:
        def call_titan():
            response = bedrock_client.invoke_model(
                modelId=TITAN_EMBED_MODEL_ID,
                body=json.dumps({
                    'inputText': text_content[:7000]  # Limit for token constraints
                })
            )
            result = json.loads(response['body'].read())
            return result.get('embedding', [])
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, call_titan)
        
        if embeddings:
            debug_log(f"Generated embeddings: {len(embeddings)} dimensions")
            return embeddings
        else:
            debug_log("No embeddings returned from Titan")
            return []
            
    except Exception as e:
        debug_log(f"Embedding generation failed: {e}")
        return []

# ================== API ENDPOINTS ==================

@app.get("/")
def root():
    """API root endpoint with configuration info"""
    return {
        "message": "🚀 EchoLearn API v4.0.0 - Production Ready",
        "status": "active",
        "version": "4.0.0",
        "mode": "production",
        "configuration": {
            "aws_region": AWS_REGION,
            "claude_model": BEDROCK_MODEL_ID,
            "s3_bucket": S3_BUCKET_NAME,
            "documents_table": DOCUMENTS_TABLE,
            "sessions_table": SESSIONS_TABLE
        },
        "features": {
            "pdf_processing": True,
            "ai_question_generation": True,
            "speech_analysis": True,
            "real_time_feedback": True,
            "semantic_similarity": True,
            "progress_tracking": True,
            "aws_transcribe": True,
            "aws_bedrock": True
        }
    }

@app.get("/health")
def health_check():
    """Comprehensive health check"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "4.0.0",
        "region": AWS_REGION,
        "services": {}
    }
    
    # Test S3
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        status["services"]["s3"] = f"✅ Connected to {S3_BUCKET_NAME}"
    except Exception as e:
        status["services"]["s3"] = f"❌ S3 Error: {str(e)}"
        status["status"] = "unhealthy"
    
    # Test DynamoDB
    try:
        list(dynamodb.tables.all())
        status["services"]["dynamodb"] = "✅ Connected"
    except Exception as e:
        status["services"]["dynamodb"] = f"❌ DynamoDB Error: {str(e)}"
        status["status"] = "unhealthy"
    
    # Test Bedrock
    try:
        bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 10,
                'messages': [{'role': 'user', 'content': 'Hi'}]
            })
        )
        status["services"]["bedrock"] = "✅ Connected and working"
    except Exception as e:
        status["services"]["bedrock"] = f"❌ Bedrock Error: {str(e)}"
        status["status"] = "unhealthy"
    
    # Test Transcribe
    try:
        transcribe_client.list_transcription_jobs(MaxResults=1)
        status["services"]["transcribe"] = "✅ Connected"
    except Exception as e:
        status["services"]["transcribe"] = f"❌ Transcribe Error: {str(e)}"
        status["status"] = "unhealthy"
    
    return status

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF with comprehensive document analysis"""
    
    debug_log(f"PDF Upload started: {file.filename}")
    
    # Validate file
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    
    try:
        # Read file content
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > MAX_PDF_SIZE_MB:
            raise HTTPException(400, f"File too large. Maximum size: {MAX_PDF_SIZE_MB}MB")
        
        if len(file_content) < 100:
            raise HTTPException(400, "File too small or empty")
        
        if not file_content.startswith(b'%PDF'):
            raise HTTPException(400, "Invalid PDF file format")
        
        debug_log(f"File validation passed: {file_size_mb:.2f}MB")
        
        # Extract text from PDF
        def extract_pdf_text():
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            return text_content, len(pdf_reader.pages)
        
        loop = asyncio.get_event_loop()
        text_content, page_count = await loop.run_in_executor(None, extract_pdf_text)
        
        if not text_content.strip():
            raise HTTPException(400, "Could not extract text from PDF")
        
        word_count = len(text_content.split())
        debug_log(f"Text extraction complete: {word_count} words, {page_count} pages")
        
        # Analyze document with Claude
        document_analysis = await analyze_document_with_claude(text_content)
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Upload to S3
        s3_key = f"uploads/{document_id}-{file.filename}"
        
        def upload_to_s3():
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=file_content,
                ContentType='application/pdf',
                Metadata={
                    'original_filename': file.filename,
                    'document_id': document_id,
                    'upload_timestamp': datetime.utcnow().isoformat()
                }
            )
            return f"s3://{S3_BUCKET_NAME}/{s3_key}"
        
        s3_uri = await loop.run_in_executor(None, upload_to_s3)
        debug_log(f"S3 upload successful: {s3_uri}")
        
        # Generate embeddings
        embeddings = await generate_embeddings(text_content)
        
        # Store document metadata
        document_data = {
            'DocumentId': document_id,
            'FileName': file.filename,
            'Text': text_content,
            'Embeddings': embeddings,
            'ProcessedAt': datetime.utcnow().isoformat(),
            'WordCount': word_count,
            'PageCount': page_count,
            'FileSizeMB': str(round(file_size_mb, 2)),  # Convert to string
            'FileSize': len(file_content),
            'S3Uri': s3_uri,
            'S3Key': s3_key,
            'S3Bucket': S3_BUCKET_NAME,
            'Status': 'processed',
            'Region': AWS_REGION,
            'ProcessedBy': 'EchoLearn v4.0.0',
            # Convert all float values to strings for DynamoDB
            'subject': document_analysis.get('subject', 'general'),
            'subject_confidence': str(document_analysis.get('subject_confidence', 0.5)),
            'difficulty_level': document_analysis.get('difficulty_level', 5),
            'key_topics': document_analysis.get('key_topics', []),
            'educational_level': document_analysis.get('educational_level', 'unknown'),
            'content_type': document_analysis.get('content_type', 'unknown'),
            'summary': document_analysis.get('summary', 'No summary available'),
            'analysis_method': document_analysis.get('analysis_method', 'claude_bedrock'),
            'analyzed_at': document_analysis.get('analyzed_at', datetime.utcnow().isoformat())
        }
        
        def store_in_dynamodb():
            documents_table = dynamodb.Table(DOCUMENTS_TABLE)
            documents_table.put_item(Item=document_data)
        
        await loop.run_in_executor(None, store_in_dynamodb)
        debug_log(f"Stored in DynamoDB: {DOCUMENTS_TABLE}")
        
        # Create response
        text_preview = text_content[:500] + "..." if len(text_content) > 500 else text_content
        
        return {
            "success": True,
            "message": "PDF processed successfully with AI analysis!",
            "filename": file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "document_id": document_id,
            "s3_uri": s3_uri,
            "processing_results": {
                "word_count": word_count,
                "page_count": page_count,
                "text_preview": text_preview,
                "analysis": {
                    'subject': document_analysis.get('subject', 'general'),
                    'subject_confidence': float(document_analysis.get('subject_confidence', 0.5)),
                    'difficulty_level': document_analysis.get('difficulty_level', 5),
                    'key_topics': document_analysis.get('key_topics', []),
                    'educational_level': document_analysis.get('educational_level', 'unknown'),
                    'content_type': document_analysis.get('content_type', 'unknown'),
                    'summary': document_analysis.get('summary', 'No summary available')
                },
                "region": AWS_REGION,
                "has_embeddings": len(embeddings) > 0,
                "embedding_dimensions": len(embeddings)
            },
            "next_steps": {
                "generate_questions": f"/generate-questions (POST with document_id: {document_id})",
                "start_learning": "Use document_id to start learning session"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"PDF processing error: {e}")
        raise HTTPException(500, f"PDF processing failed: {str(e)}")

@app.post("/generate-questions")
async def generate_questions(
    document_id: str = Form(...),
    difficulty: int = Form(5),
    num_questions: int = Form(3),
    question_type: str = Form("multiple_choice")
):
    """Generate questions using Claude with real document content"""
    
    debug_log(f"Generating {num_questions} {question_type} questions (difficulty: {difficulty}) for document: {document_id}")
    
    try:
        # Get document content
        document_data = await get_document_content(document_id)
        
        if not document_data:
            raise HTTPException(404, f"Document {document_id} not found")
        
        text_content = document_data.get('Text', '')
        if not text_content:
            raise HTTPException(400, "Document has no text content")
        
        # Generate questions using Claude
        questions = await generate_questions_with_claude(
            text_content, 
            difficulty, 
            num_questions, 
            question_type
        )
        
        # Store session in DynamoDB
        session_id = str(uuid.uuid4())
        session_data = {
            'SessionId': session_id,
            'DocumentId': document_id,
            'Questions': questions,
            'CreatedAt': datetime.utcnow().isoformat(),
            'QuestionType': question_type,
            'Difficulty': difficulty,
            'NumQuestions': len(questions),
            'Status': 'active'
        }
        
        def store_session():
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            sessions_table.put_item(Item=session_data)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, store_session)
        debug_log(f"Stored session in DynamoDB: {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "document_id": document_id,
            "questions": questions,
            "difficulty": difficulty,
            "question_type": question_type,
            "total_questions": len(questions),
            "ai_generated": True,
            "model_used": "Claude 3.5 Sonnet via AWS Bedrock",
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"Question generation error: {e}")
        raise HTTPException(500, f"Question generation failed: {str(e)}")

@app.post("/analyze-speech")
async def analyze_speech(
    audio: UploadFile = File(...),
    document_id: str = Form(...),
    question_id: str = Form(None)
):
    """Analyze student's speech answer using AWS Transcribe and semantic analysis"""
    
    debug_log(f"Speech analysis started for document: {document_id}")
    
    # Validate audio file
    if not audio.content_type or not audio.content_type.startswith('audio/'):
        raise HTTPException(400, "Only audio files are allowed")
    
    try:
        # Read audio content
        audio_content = await audio.read()
        
        if len(audio_content) < 1000:
            raise HTTPException(400, "Audio file too small")
        
        # Get document content first
        document_data = await get_document_content(document_id)
        if not document_data:
            raise HTTPException(404, f"Document {document_id} not found")
        
        document_text = document_data.get('Text', '')
        if not document_text:
            raise HTTPException(400, "Document has no text content")
        
        # Upload audio to S3 for transcription
        audio_key = f"audio/{uuid.uuid4()}-{audio.filename}"
        
        def upload_audio_to_s3():
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=audio_key,
                Body=audio_content,
                ContentType=audio.content_type,
                Metadata={
                    'document_id': document_id,
                    'question_id': question_id or '',
                    'upload_timestamp': datetime.utcnow().isoformat()
                }
            )
            return f"s3://{S3_BUCKET_NAME}/{audio_key}"
        
        loop = asyncio.get_event_loop()
        audio_s3_uri = await loop.run_in_executor(None, upload_audio_to_s3)
        debug_log(f"Audio uploaded to S3: {audio_s3_uri}")
        
        # Transcribe audio using AWS Transcribe
        transcript_text = await transcribe_audio_with_aws(audio_s3_uri)
        
        if not transcript_text:
            raise HTTPException(400, "Could not transcribe audio - no speech detected")
        
        debug_log(f"Transcription completed: '{transcript_text}'")
        
        # Calculate semantic similarity
        similarity_score = await calculate_semantic_similarity(transcript_text, document_text)
        similarity_percentage = similarity_score * 100
        
        debug_log(f"Semantic similarity: {similarity_percentage:.2f}%")
        
        # Generate AI feedback
        feedback = await generate_ai_feedback(transcript_text, document_text, similarity_score)
        
        # Store analysis results
        analysis_id = str(uuid.uuid4())
        analysis_data = {
            'AnalysisId': analysis_id,
            'DocumentId': document_id,
            'QuestionId': question_id or 'general',
            'Transcript': transcript_text,
            'SimilarityScore': str(similarity_score),  # Convert to string
            'SimilarityPercentage': str(similarity_percentage),  # Convert to string
            'Feedback': feedback,
            'AudioS3Uri': audio_s3_uri,
            'AudioKey': audio_key,
            'CreatedAt': datetime.utcnow().isoformat(),
            'Status': 'completed',
            'ProcessedBy': 'EchoLearn v4.0.0'
        }
        
        def store_analysis():
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            sessions_table.put_item(Item=analysis_data)
        
        await loop.run_in_executor(None, store_analysis)
        debug_log(f"Analysis stored with ID: {analysis_id}")
        
        # Determine performance level
        if similarity_percentage >= 80:
            performance_level = "Excellent"
        elif similarity_percentage >= 60:
            performance_level = "Good"
        elif similarity_percentage >= 40:
            performance_level = "Fair"
        else:
            performance_level = "Needs Improvement"
        
        return {
            "success": True,
            "analysis_id": analysis_id,
            "document_id": document_id,
            "question_id": question_id,
            "transcript": transcript_text,
            "similarity_score": float(similarity_score),
            "similarity_percentage": float(similarity_percentage),
            "performance_level": performance_level,
            "feedback": feedback,
            "audio_s3_uri": audio_s3_uri,
            "processing_details": {
                "transcription_service": "AWS Transcribe",
                "similarity_method": "Semantic Embedding Cosine Similarity",
                "feedback_generator": "Claude 3.5 Sonnet",
                "processed_at": datetime.utcnow().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"Speech analysis error: {e}")
        raise HTTPException(500, f"Speech analysis failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        def get_documents():
            documents_table = dynamodb.Table(DOCUMENTS_TABLE)
            response = documents_table.scan()
            documents = response['Items']
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = documents_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                documents.extend(response['Items'])
            
            # Sort by processing date (newest first)
            documents.sort(key=lambda x: x.get('ProcessedAt', ''), reverse=True)
            return documents
        
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, get_documents)
        
        return {
            "success": True,
            "documents": documents,
            "total_count": len(documents)
        }
        
    except Exception as e:
        debug_log(f"Error listing documents: {e}")
        raise HTTPException(500, f"Failed to list documents: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document(document_id: str):
    """Get specific document details"""
    try:
        document = await get_document_content(document_id)
        
        if not document:
            raise HTTPException(404, f"Document {document_id} not found")
        
        return {
            "success": True,
            "document": document
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"Error getting document {document_id}: {e}")
        raise HTTPException(500, f"Failed to get document: {str(e)}")

@app.get("/documents/{document_id}/sessions")
async def get_document_sessions(document_id: str):
    """Get all learning sessions for a document"""
    try:
        def get_sessions():
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            response = sessions_table.scan(
                FilterExpression='DocumentId = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id}
            )
            sessions = response['Items']
            sessions.sort(key=lambda x: x.get('CreatedAt', ''), reverse=True)
            return sessions
        
        loop = asyncio.get_event_loop()
        sessions = await loop.run_in_executor(None, get_sessions)
        
        return {
            "success": True,
            "document_id": document_id,
            "sessions": sessions,
            "total_count": len(sessions)
        }
        
    except Exception as e:
        debug_log(f"Error getting sessions for {document_id}: {e}")
        raise HTTPException(500, f"Failed to get sessions: {str(e)}")

@app.post("/documents/search")
async def search_documents(
    query: str = Form(""),
    subject: str = Form(None),
    difficulty: int = Form(None)
):
    """Search documents by content, subject, or difficulty"""
    try:
        def search_docs():
            documents_table = dynamodb.Table(DOCUMENTS_TABLE)
            response = documents_table.scan()
            documents = response['Items']
            
            # Filter by query
            if query:
                query_lower = query.lower()
                documents = [doc for doc in documents if 
                           query_lower in doc.get('Text', '').lower() or 
                           query_lower in doc.get('FileName', '').lower()]
            
            # Filter by subject
            if subject:
                documents = [doc for doc in documents if 
                           doc.get('subject', '').lower() == subject.lower()]
            
            # Filter by difficulty
            if difficulty:
                documents = [doc for doc in documents if 
                           doc.get('difficulty_level', 5) == difficulty]
            
            return documents
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, search_docs)
        
        return {
            "success": True,
            "query": query,
            "filters": {
                "subject": subject,
                "difficulty": difficulty
            },
            "results": results,
            "total_count": len(results)
        }
        
    except Exception as e:
        debug_log(f"Error searching documents: {e}")
        raise HTTPException(500, f"Search failed: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated data"""
    try:
        # Get document to find S3 key
        document = await get_document_content(document_id)
        if not document:
            raise HTTPException(404, f"Document {document_id} not found")
        
        def delete_operations():
            # Delete from S3
            s3_key = document.get('S3Key')
            if s3_key:
                try:
                    s3_client.delete_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    debug_log(f"Deleted S3 object: {s3_key}")
                except Exception as e:
                    debug_log(f"Failed to delete S3 object: {e}")
            
            # Delete from DynamoDB
            documents_table = dynamodb.Table(DOCUMENTS_TABLE)
            documents_table.delete_item(Key={'DocumentId': document_id})
            
            # Delete associated sessions
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            sessions_response = sessions_table.scan(
                FilterExpression='DocumentId = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id}
            )
            
            deleted_sessions = 0
            for session in sessions_response['Items']:
                session_key = {'SessionId': session['SessionId']} if 'SessionId' in session else {'AnalysisId': session['AnalysisId']}
                sessions_table.delete_item(Key=session_key)
                deleted_sessions += 1
            
            debug_log(f"Deleted document {document_id} and {deleted_sessions} sessions")
            return deleted_sessions
        
        loop = asyncio.get_event_loop()
        deleted_sessions = await loop.run_in_executor(None, delete_operations)
        
        return {
            "success": True,
            "message": f"Document {document_id} deleted successfully",
            "deleted_sessions": deleted_sessions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"Error deleting document {document_id}: {e}")
        raise HTTPException(500, f"Failed to delete document: {str(e)}")

@app.get("/stats")
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        def get_stats():
            # Get document stats
            documents_table = dynamodb.Table(DOCUMENTS_TABLE)
            doc_response = documents_table.scan()
            documents = doc_response['Items']
            
            # Get session stats
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            session_response = sessions_table.scan()
            sessions = session_response['Items']
            
            # Calculate statistics
            total_documents = len(documents)
            total_sessions = len(sessions)
            
            # Separate question sessions from analysis sessions
            question_sessions = [s for s in sessions if s.get('Questions')]
            analysis_sessions = [s for s in sessions if s.get('Transcript')]
            
            total_questions = sum(session.get('NumQuestions', 0) for session in question_sessions)
            
            # Calculate average similarity for analysis sessions
            similarity_scores = []
            for s in analysis_sessions:
                score = safe_float(s.get('SimilarityScore', 0))
                similarity_scores.append(score)
            
            avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
            
            # Document analysis
            difficulties = [doc.get('difficulty_level', 5) for doc in documents if doc.get('difficulty_level')]
            avg_difficulty = sum(difficulties) / len(difficulties) if difficulties else 5
            
            subjects = [doc.get('subject', 'general') for doc in documents if doc.get('subject')]
            most_common_subject = max(set(subjects), key=subjects.count) if subjects else 'general'
            
            # Get last activity
            all_timestamps = []
            all_timestamps.extend([doc.get('ProcessedAt', '') for doc in documents])
            all_timestamps.extend([session.get('CreatedAt', '') for session in sessions])
            last_activity = max(all_timestamps) if all_timestamps else ''
            
            # Calculate storage usage
            total_file_size = sum(doc.get('FileSize', 0) for doc in documents)
            total_words = sum(doc.get('WordCount', 0) for doc in documents)
            
            return {
                "total_documents": total_documents,
                "total_sessions": total_sessions,
                "question_sessions": len(question_sessions),
                "analysis_sessions": len(analysis_sessions),
                "total_questions_generated": total_questions,
                "average_difficulty": round(avg_difficulty, 1),
                "average_similarity_score": round(avg_similarity * 100, 1),
                "most_common_subject": most_common_subject,
                "last_activity": last_activity,
                "total_file_size_mb": round(total_file_size / (1024 * 1024), 2),
                "total_words_processed": total_words,
                "subjects_distribution": {subject: subjects.count(subject) for subject in set(subjects)},
                "question_types": {
                    qt: len([s for s in question_sessions if s.get('QuestionType') == qt])
                    for qt in ['multiple_choice', 'short_answer', 'mixed']
                },
                "performance_levels": {
                    "excellent": len([s for s in analysis_sessions if safe_float(s.get('SimilarityScore', 0)) >= 0.8]),
                    "good": len([s for s in analysis_sessions if 0.6 <= safe_float(s.get('SimilarityScore', 0)) < 0.8]),
                    "fair": len([s for s in analysis_sessions if 0.4 <= safe_float(s.get('SimilarityScore', 0)) < 0.6]),
                    "needs_improvement": len([s for s in analysis_sessions if safe_float(s.get('SimilarityScore', 0)) < 0.4])
                }
            }
        
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, get_stats)
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        debug_log(f"Error getting stats: {e}")
        raise HTTPException(500, f"Failed to get statistics: {str(e)}")

@app.get("/documents/{document_id}/analytics")
async def get_document_analytics(document_id: str):
    """Get detailed analytics for a specific document"""
    try:
        # Get document
        document = await get_document_content(document_id)
        if not document:
            raise HTTPException(404, f"Document {document_id} not found")
        
        def get_analytics():
            # Get sessions for this document
            sessions_table = dynamodb.Table(SESSIONS_TABLE)
            sessions_response = sessions_table.scan(
                FilterExpression='DocumentId = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id}
            )
            sessions = sessions_response['Items']
            
            # Separate session types
            question_sessions = [s for s in sessions if s.get('Questions')]
            analysis_sessions = [s for s in sessions if s.get('Transcript')]
            
            # Calculate analytics
            question_types = {}
            difficulty_distribution = {}
            
            for session in question_sessions:
                qt = session.get('QuestionType', 'unknown')
                question_types[qt] = question_types.get(qt, 0) + 1
                
                diff = session.get('Difficulty', 5)
                difficulty_distribution[diff] = difficulty_distribution.get(diff, 0) + 1
            
            # Performance analytics
            similarity_scores = [s.get('SimilarityScore', 0) for s in analysis_sessions]
            avg_performance = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
            
            # Recent activity
            recent_sessions = sorted(sessions, key=lambda x: x.get('CreatedAt', ''), reverse=True)[:10]
            
            return {
                "document_id": document_id,
                "document_info": {
                    "filename": document.get('FileName', 'Unknown'),
                    "subject": document.get('subject', 'Unknown'),
                    "difficulty_level": document.get('difficulty_level', 5),
                    "word_count": document.get('WordCount', 0),
                    "page_count": document.get('PageCount', 0),
                    "processed_at": document.get('ProcessedAt', ''),
                    "file_size_mb": document.get('FileSizeMB', 0)
                },
                "usage_stats": {
                    "total_sessions": len(sessions),
                    "question_sessions": len(question_sessions),
                    "analysis_sessions": len(analysis_sessions),
                    "question_types": question_types,
                    "difficulty_distribution": difficulty_distribution,
                    "average_performance": round(avg_performance * 100, 1),
                    "last_accessed": sessions[0].get('CreatedAt') if sessions else None
                },
                "recent_activity": recent_sessions
            }
        
        loop = asyncio.get_event_loop()
        analytics = await loop.run_in_executor(None, get_analytics)
        
        return {
            "success": True,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"Error getting analytics for {document_id}: {e}")
        raise HTTPException(500, f"Failed to get analytics: {str(e)}")

@app.post("/documents/{document_id}/regenerate-embeddings")
async def regenerate_embeddings(document_id: str):
    """Regenerate embeddings for a document"""
    try:
        # Get document
        document = await get_document_content(document_id)
        if not document:
            raise HTTPException(404, f"Document {document_id} not found")
        
        text_content = document.get('Text', '')
        if not text_content:
            raise HTTPException(400, "Document has no text content")
        
        # Generate new embeddings
        embeddings = await generate_embeddings(text_content)
        
        if not embeddings:
            raise HTTPException(500, "Failed to generate embeddings")
        
        # Update document with new embeddings
        def update_embeddings():
            documents_table = dynamodb.Table(DOCUMENTS_TABLE)
            documents_table.update_item(
                Key={'DocumentId': document_id},
                UpdateExpression='SET Embeddings = :emb, UpdatedAt = :updated',
                ExpressionAttributeValues={
                    ':emb': embeddings,
                    ':updated': datetime.utcnow().isoformat()
                }
            )
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, update_embeddings)
        
        return {
            "success": True,
            "message": f"Embeddings regenerated for document {document_id}",
            "embedding_dimensions": len(embeddings)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"Error regenerating embeddings for {document_id}: {e}")
        raise HTTPException(500, f"Failed to regenerate embeddings: {str(e)}")

@app.post("/batch-process")
async def batch_process_documents(files: List[UploadFile] = File(...)):
    """Process multiple PDF files in batch"""
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Process each file individually
            result = await upload_pdf(file)
            results.append({
                "filename": file.filename,
                "status": "success",
                "document_id": result["document_id"],
                "details": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    successful = len([r for r in results if r["status"] == "success"])
    failed = len([r for r in results if r["status"] == "error"])
    
    return {
        "success": True,
        "total_files": len(files),
        "successful": successful,
        "failed": failed,
        "results": results
    }

# ================== STARTUP AND SHUTDOWN ==================

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    debug_log("EchoLearn API starting up...")
    
    # Test AWS connectivity
    try:
        # Test DynamoDB tables exist
        existing_tables = list(dynamodb.tables.all())
        table_names = [table.name for table in existing_tables]
        
        if DOCUMENTS_TABLE not in table_names:
            debug_log(f"Warning: DynamoDB table {DOCUMENTS_TABLE} not found")
        
        if SESSIONS_TABLE not in table_names:
            debug_log(f"Warning: DynamoDB table {SESSIONS_TABLE} not found")
            
        # Test S3 bucket access
        s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
        debug_log(f"S3 bucket {S3_BUCKET_NAME} accessible")
        
        # Test Bedrock access
        bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 10,
                'messages': [{'role': 'user', 'content': 'test'}]
            })
        )
        debug_log("Bedrock Claude access verified")
        
        # Test Transcribe access
        transcribe_client.list_transcription_jobs(MaxResults=1)
        debug_log("AWS Transcribe access verified")
        
        debug_log("All AWS services connectivity verified")
        
    except Exception as e:
        debug_log(f"AWS connectivity check failed: {e}")
        raise RuntimeError(f"AWS services not accessible: {e}")
    
    debug_log("EchoLearn API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    debug_log("EchoLearn API shutting down...")
    debug_log("Shutdown complete")

# ================== MAIN ENTRY POINT ==================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    
    print(f"\n🚀 Starting EchoLearn API v4.0.0 - Production Mode")
    print(f"🌏 Region: {AWS_REGION}")
    print(f"🤖 Claude Model: {BEDROCK_MODEL_ID}")
    print(f"📊 S3 Bucket: {S3_BUCKET_NAME}")
    print(f"🗄️  DynamoDB Tables: {DOCUMENTS_TABLE}, {SESSIONS_TABLE}")
    print(f"📡 Port: {port}")
    print(f"🌐 API Docs: http://localhost:{port}/docs")
    print(f"📊 Health Check: http://localhost:{port}/health")
    print(f"📈 System Stats: http://localhost:{port}/stats")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )