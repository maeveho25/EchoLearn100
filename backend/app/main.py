from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import boto3
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import os
import tempfile

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
        "message": "🚀 EchoLearn API is running!",
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
    """Test AI generation với AWS Bedrock"""
    if not AWS_AVAILABLE:
        return {
            "success": False,
            "error": "AWS not configured",
            "mock_response": "This would be an AI-generated response about: " + prompt
        }
    
    try:
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
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
    """Upload PDF to S3"""
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    if not AWS_AVAILABLE:
        return {
            "success": False,
            "error": "AWS not configured", 
            "mock": {
                "filename": file.filename,
                "size": len(await file.read()),
                "message": "Would upload to S3 if AWS was configured"
            }
        }
    
    try:
        # Read file content
        content = await file.read()
        
        # Generate S3 key
        s3_key = f"uploads/{uuid.uuid4()}-{file.filename}"
        bucket = os.getenv('S3_BUCKET_NAME')
        
        if not bucket:
            raise HTTPException(500, "S3_BUCKET_NAME not configured in .env")
        
        # Upload to S3
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=content,
            ContentType='application/pdf'
        )
        
        # TODO: Call Lambda function to process PDF
        # For now, return success with mock processing
        
        return {
            "success": True,
            "message": "PDF uploaded successfully!",
            "filename": file.filename,
            "size": len(content),
            "s3_uri": f"s3://{bucket}/{s3_key}",
            "document_id": str(uuid.uuid4()),  # Mock document ID
            "next_step": "Call /generate-questions with this document_id"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.post("/generate-questions")
def generate_questions(
    document_id: str = Form(...),
    difficulty: int = Form(5),
    num_questions: int = Form(3),
    question_type: str = Form("multiple_choice")
):
    """Generate questions for a document"""
    
    # Mock questions for testing
    mock_questions = [
        {
            "id": str(uuid.uuid4()),
            "question": "What is the main topic discussed in this document?",
            "type": "multiple_choice",
            "options": ["A) Machine Learning", "B) Data Science", "C) Programming", "D) Mathematics"],
            "correct_answer": "A",
            "difficulty": difficulty
        },
        {
            "id": str(uuid.uuid4()),
            "question": "Which of the following algorithms is commonly used in supervised learning?",
            "type": "multiple_choice", 
            "options": ["A) K-means", "B) Linear Regression", "C) PCA", "D) DBSCAN"],
            "correct_answer": "B",
            "difficulty": difficulty
        },
        {
            "id": str(uuid.uuid4()),
            "question": "Explain the difference between supervised and unsupervised learning.",
            "type": "short_answer",
            "expected_keywords": ["labeled data", "training", "prediction", "patterns"],
            "difficulty": difficulty
        }
    ]
    
    # Filter by question type
    if question_type == "multiple_choice":
        questions = [q for q in mock_questions if q["type"] == "multiple_choice"]
    elif question_type == "short_answer":
        questions = [q for q in mock_questions if q["type"] == "short_answer"] 
    else:
        questions = mock_questions
    
    # Limit number of questions
    questions = questions[:num_questions]
    
    # TODO: Use AWS Bedrock to generate real questions based on document content
    
    return {
        "success": True,
        "document_id": document_id,
        "questions": questions,
        "difficulty": difficulty,
        "question_type": question_type,
        "total_questions": len(questions),
        "note": "These are mock questions. Real implementation will use AWS Bedrock.",
        "next_step": "Call /analyze-speech with audio file and question_id"
    }

@app.post("/analyze-speech")
async def analyze_speech(
    audio: UploadFile = File(...),
    document_id: str = Form(...),
    question_id: str = Form(None)
):
    """Analyze student's speech answer"""
    
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
        # Read audio content
        content = await audio.read()
        
        # Upload audio to S3
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
        
        # Mock response for testing
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
    """List uploaded documents (mock data for testing)"""
    
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