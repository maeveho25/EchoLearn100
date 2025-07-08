from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import os
import tempfile
from pdf_service import pdf_service
from dynamodb_service import db_service
import json

load_dotenv()

app = FastAPI(
    title="EchoLearn API",
    description="AI Learning Assistant API for Hackathon",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS clients
try:
    s3_client = boto3.client('s3')
    bedrock_client = boto3.client('bedrock-runtime')
    dynamodb = boto3.resource('dynamodb')
    AWS_AVAILABLE = True
except Exception as e:
    print(f"AWS not configured: {e}")
    AWS_AVAILABLE = False

@app.get("/")
def root():
    return {
        "message": "EchoLearn API is running!",
        "status": "active",
        "aws_available": AWS_AVAILABLE,
        "endpoints": {
            "docs": "/docs",
            "health": "/health", 
            "upload": "/upload-pdf",
            "questions": "/generate-questions",
            "speech": "/analyze-speech",
            "test_ai": "/test-ai"
        }
    }

@app.get("/test")
def test():
    return {"test": "API working!"}

@app.get("/health")
def health_check():
    """Health check - test AWS connectivity"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    if AWS_AVAILABLE:
        # Test S3
        try:
            print("TEST S3:")
            s3_client.list_buckets()
            status["services"]["s3"] = "Connected"
        except Exception as e:
            status["services"]["s3"] = f"Error: {str(e)}"
        
        # Test DynamoDB
        try:
            print("TEST DynamoDB:")
            list(dynamodb.tables.all())
            status["services"]["dynamodb"] = "Connected"
        except Exception as e:
            status["services"]["dynamodb"] = f"Error: {str(e)}"
        
        # Test Bedrock
        try:
            print("TEST Bedrock:")
            status["services"]["bedrock"] = "Client ready"
        except Exception as e:
            status["services"]["bedrock"] = f"Error: {str(e)}"
    else:
        status["services"]["aws"] = "Not configured"
    
    return status

@app.post("/test-ai")
def test_ai(prompt: str = Form("What is machine learning?")):
    if not AWS_AVAILABLE:
        return {
            "success": False,
            "error": "AWS not configured",
            "mock_response": "This would be an AI-generated response about: " + prompt
        }
    
    try:
        response = bedrock_client.invoke_model(
            modelId=os.getenv("BEDROCK_MODEL_ID"),
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 200,
                'messages': [{'role': 'user', 'content': prompt}]
            })
        )
        
        result = json.loads(response['body'].read())
        ai_response = result['content'][0]['text']
        
        return {
            "success": True,
            "prompt": prompt,
            "ai_response": ai_response,
            "service": "AWS Bedrock Claude 3.5"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "mock_response": f"Mock AI response for: {prompt}"
        }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process REAL PDF file - No samples"""
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are allowed")
    
    if not AWS_AVAILABLE:
        raise HTTPException(503, "AWS services not available")
    
    try:
        print(f"📄 Processing uploaded file: {file.filename}")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        print(f"📊 File size: {file_size} bytes")
        
        # Validate file size (max 10MB for demo)
        if file_size > 10 * 1024 * 1024:
            raise HTTPException(400, "File too large. Maximum size is 10MB")
        
        if file_size < 1000:
            raise HTTPException(400, "File too small. Please upload a valid PDF")
        
        print("📤 Uploading to S3...")
        s3_info = pdf_service.upload_pdf_to_s3(file_content, file.filename)
        
        print("📝 Extracting text from PDF...")
        text_info = pdf_service.extract_text_from_pdf(file_content)
        
        print("🧠 Generating embeddings...")
        embeddings = pdf_service.generate_embeddings(text_info['text'])

        difficulty = pdf_service.estimate_difficulty(text_info['text'], text_info['word_count'])
        
        document_id = str(uuid.uuid4())
        
        document_data = {
            'DocumentId': document_id,
            'FileName': file.filename,
            'Text': text_info['text'],
            'Embeddings': embeddings or [],
            'ProcessedAt': datetime.utcnow().isoformat(),
            'WordCount': text_info['word_count'],
            'PageCount': text_info['page_count'],
            'CharCount': text_info['char_count'],
            'Subject': text_info['subject'],
            'Difficulty': difficulty,
            'S3Bucket': s3_info['bucket'],
            'S3Key': s3_info['s3_key'],
            'FileSize': file_size,
            'Status': 'processed',
            'CreatedAt': datetime.utcnow().isoformat(),
            'OriginalFilename': file.filename
        }
        
        print("💾 Saving to DynamoDB...")
        db_service.insert_document(document_data)
    
        db_service.save_analytics(
            user_id='demo-user',
            event_type='pdf_uploaded',
            data={
                'document_id': document_id,
                'filename': file.filename,
                'file_size': file_size,
                'word_count': text_info['word_count'],
                'subject': text_info['subject'],
                'difficulty': difficulty
            }
        )
        
        print(f"✅ PDF processing completed: {document_id}")
        
        return {
            "success": True,
            "message": "PDF uploaded and processed successfully!",
            "document_id": document_id,
            "filename": file.filename,
            "file_size": file_size,
            "processing_results": {
                "word_count": text_info['word_count'],
                "page_count": text_info['page_count'],
                "subject": text_info['subject'],
                "difficulty": difficulty,
                "has_embeddings": len(embeddings) > 0 if embeddings else False,
                "text_preview": text_info['text'][:300] + "..." if len(text_info['text']) > 300 else text_info['text']
            },
            "s3_location": s3_info['s3_uri'],
            "next_steps": "Use this document_id to generate questions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"❌ PDF upload failed: {error_msg}")
        
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {error_msg}"
        )

# Add endpoint to test with existing documents
@app.get("/documents/sample")
def get_sample_documents():
    """Get documents for testing - shows real uploaded documents only"""
    try:
        documents = db_service.get_all_documents()
        
        if not documents:
            return {
                "success": True,
                "message": "No documents found. Please upload a PDF first.",
                "documents": [],
                "upload_endpoint": "/upload-pdf"
            }
        
        return {
            "success": True,
            "documents": documents,
            "total_count": len(documents),
            "message": "These are real uploaded documents"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to get documents: {str(e)}")
    
@app.post("/generate-questions")
def generate_questions(
    document_id: str = Form(...),
    difficulty: int = Form(5),
    num_questions: int = Form(3),
    question_type: str = Form("multiple_choice")
):
    
    # Check AWS availability first
    if not AWS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="AWS Bedrock service unavailable. Cannot generate AI questions."
        )
    
    try:
        # TODO: Get REAL document content from DynamoDB
        # For demo, use sample content to show AI generation works
        
        sample_content = f"""
        Educational content about machine learning and artificial intelligence.
        Machine learning enables computers to learn patterns from data without explicit programming.
        Key concepts include supervised learning (classification, regression), 
        unsupervised learning (clustering, dimensionality reduction), 
        and reinforcement learning (reward-based learning).
        Applications include computer vision, natural language processing, 
        recommendation systems, and autonomous vehicles.
        """
        
        # Generate REAL questions using Bedrock Claude
        prompt = f"""
Based on this educational content, create {num_questions} {question_type} questions at difficulty level {difficulty}/10:

Content: {sample_content}

Requirements:
- Questions should test understanding, not memorization
- Include plausible but incorrect options
- Appropriate for difficulty level {difficulty}/10

Format as JSON array:
[
    {{
        "question": "Question text here?",
        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
        "correct_answer": "A",
        "explanation": "Brief explanation why this is correct"
    }}
]

Return ONLY the JSON array, no other text.
"""
        
        # Call Claude to generate questions
        response = bedrock_client.invoke_model(
            modelId='apac.anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1500,
                'messages': [{'role': 'user', 'content': prompt}]
            })
        )
        
        result = json.loads(response['body'].read())
        ai_response = result['content'][0]['text']
        
        # Parse AI response to extract questions
        import re
        json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
        
        if not json_match:
            raise HTTPException(
                status_code=500,
                detail="AI failed to generate properly formatted questions"
            )
        
        try:
            questions = json.loads(json_match.group())
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, 
                detail="AI generated invalid JSON format"
            )
        
        if not questions or len(questions) == 0:
            raise HTTPException(
                status_code=500,
                detail="AI generated empty question list"
            )
        
        # Add metadata to questions
        for i, q in enumerate(questions):
            q['id'] = str(uuid.uuid4())
            q['document_id'] = document_id
            q['generated_at'] = datetime.utcnow().isoformat()
            
            # Validate question structure
            required_fields = ['question', 'options', 'correct_answer']
            missing_fields = [field for field in required_fields if field not in q]
            if missing_fields:
                raise HTTPException(
                    status_code=500,
                    detail=f"AI generated invalid question #{i+1}: missing {missing_fields}"
                )
        
        return {
            "success": True,
            "document_id": document_id,
            "questions": questions,
            "difficulty": difficulty,
            "question_type": question_type,
            "total_questions": len(questions),
            "ai_generated": True,
            "model_used": "Claude 3.5 Sonnet",
            "generated_at": datetime.utcnow().isoformat()
        }
            
    except HTTPException:
        raise
        
    except Exception as e:
        error_message = str(e)
        print(f"Question generation error: {error_message}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {error_message}"
        )

@app.post("/analyze-speech")
async def analyze_speech(
    audio: UploadFile = File(...),
    document_id: str = Form(...),
    question_id: str = Form(None)
):
    
    # Validate audio file
    if not audio.content_type.startswith('audio/'):
        raise HTTPException(400, "Only audio files allowed")
    
    if not AWS_AVAILABLE:
        return {
            "success": False,
            "error": "AWS not configured",
            "mock": {
                "transcript": "This would be the transcribed speech",
                "similarity_score": 85.5,
                "feedback": "Mock feedback about the answer"
            }
        }
    
    try:
        content = await audio.read()

        audio_key = f"audio/{uuid.uuid4()}-{audio.filename}"
        bucket = os.getenv('S3_BUCKET_NAME')
        
        s3_client.put_object(
            Bucket=bucket,
            Key=audio_key,
            Body=content,
            ContentType=audio.content_type
        )
        
        # TODO: 
        # 1. Use AWS Transcribe to convert speech to text
        # 2. Use AWS Bedrock to analyze similarity with reference answer
        # 3. Generate feedback

        mock_transcript = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
        mock_similarity = 87.5
        mock_feedback = "Great answer! You correctly identified machine learning as a subset of AI and mentioned the key concept of learning from data. You could improve by discussing specific applications or algorithms."
        
        return {
            "success": True,
            "document_id": document_id,
            "question_id": question_id,
            "transcript": mock_transcript,
            "similarity_score": mock_similarity,
            "feedback": mock_feedback,
            "audio_s3_uri": f"s3://{bucket}/{audio_key}",
            "note": "This is a mock analysis. Real implementation will use AWS Transcribe + Bedrock.",
            "performance_level": "Good" if mock_similarity > 80 else "Needs Improvement"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Speech analysis failed: {str(e)}")

@app.get("/documents")
def list_documents():
    mock_documents = [
        {
            "document_id": "doc-123",
            "filename": "machine_learning_basics.pdf",
            "uploaded_at": "2024-01-15T10:30:00Z",
            "status": "processed",
            "page_count": 15,
            "word_count": 5420
        },
        {
            "document_id": "doc-456", 
            "filename": "data_science_intro.pdf",
            "uploaded_at": "2024-01-14T14:20:00Z",
            "status": "processed",
            "page_count": 8,
            "word_count": 3200
        }
    ]
    
    return {
        "success": True,
        "documents": mock_documents,
        "total_count": len(mock_documents),
        "note": "Mock data. Real implementation will fetch from DynamoDB."
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting EchoLearn API on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)