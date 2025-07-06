# question_generator.py
import json
import boto3
from datetime import datetime
import uuid
import re

def lambda_handler(event, context):
    """
    Generate questions based on document content using Claude:
    1. Get document from DynamoDB
    2. Call Bedrock Claude to generate questions
    3. Parse and format questions
    4. Store in DynamoDB and return
    """
    try:
        print(f"Received event: {json.dumps(event, default=str)}")
        
        # Initialize AWS clients
        bedrock_client = boto3.client('bedrock-runtime')
        dynamodb = boto3.resource('dynamodb')
        
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        document_id = body.get('documentId')
        difficulty = body.get('difficulty', 5)
        question_type = body.get('type', 'multiple_choice')
        num_questions = body.get('numQuestions', 3)
        
        if not document_id:
            raise ValueError("documentId is required")
        
        print(f"Generating {num_questions} {question_type} questions for document {document_id}, difficulty {difficulty}")
        
        # Get document from DynamoDB
        documents_table = dynamodb.Table('EchoLearn-Documents')
        doc_response = documents_table.get_item(Key={'DocumentId': document_id})
        
        if 'Item' not in doc_response:
            raise ValueError(f"Document not found: {document_id}")
        
        document = doc_response['Item']
        text_content = document['Text']
        file_name = document.get('FileName', 'Unknown')
        
        print(f"Found document: {file_name}, {len(text_content)} characters")
        
        # Prepare prompt based on question type
        if question_type == 'multiple_choice':
            prompt = f"""
Based on the following educational content, generate {num_questions} multiple choice questions at difficulty level {difficulty}/10.

Content:
{text_content[:3000]}

Please format your response as a valid JSON array with this exact structure:
[
    {{
        "question": "What is the main concept discussed in the content?",
        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
        "correct_answer": "A) Option 1",
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
- Return ONLY the JSON array, no additional text
"""
        
        elif question_type == 'short_answer':
            prompt = f"""
Based on the following educational content, generate {num_questions} short answer questions at difficulty level {difficulty}/10.

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
- Return ONLY the JSON array, no additional text
"""
        
        else:  # mixed
            prompt = f"""
Based on the following educational content, generate {num_questions} mixed questions (both multiple choice and short answer) at difficulty level {difficulty}/10.

Content:
{text_content[:3000]}

Please format your response as a valid JSON array mixing both question types:
[
    {{
        "question": "Multiple choice question text?",
        "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
        "correct_answer": "A) Option 1",
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

Return ONLY the JSON array, no additional text.
"""
        
        # Call Claude via Bedrock
        try:
            response = bedrock_client.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 2000,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                })
            )
            
            result = json.loads(response['body'].read())
            claude_response = result['content'][0]['text']
            print(f"Claude response received: {len(claude_response)} characters")
            
        except Exception as e:
            raise Exception(f"Failed to call Bedrock Claude: {str(e)}")
        
        # Parse JSON from Claude response
        try:
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', claude_response, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            else:
                # Fallback: try to find JSON-like content
                start_idx = claude_response.find('[')
                end_idx = claude_response.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_text = claude_response[start_idx:end_idx]
                else:
                    raise ValueError("No JSON array found in Claude response")
            
            questions = json.loads(json_text)
            print(f"Parsed {len(questions)} questions successfully")
            
            # Validate questions structure
            for i, q in enumerate(questions):
                if 'question' not in q or 'type' not in q:
                    raise ValueError(f"Invalid question structure at index {i}")
                
                # Add question ID
                q['questionId'] = str(uuid.uuid4())
                q['documentId'] = document_id
            
        except Exception as e:
            print(f"Error parsing Claude response: {e}")
            print(f"Claude response was: {claude_response}")
            raise Exception(f"Failed to parse questions from Claude response: {str(e)}")
        
        # Store questions session in DynamoDB
        session_id = str(uuid.uuid4())
        sessions_table = dynamodb.Table('EchoLearn-Sessions')
        
        session_item = {
            'SessionId': session_id,
            'DocumentId': document_id,
            'Questions': questions,
            'CreatedAt': datetime.utcnow().isoformat(),
            'QuestionType': question_type,
            'Difficulty': difficulty,
            'NumQuestions': len(questions),
            'Status': 'active'
        }
        
        sessions_table.put_item(Item=session_item)
        print(f"Stored questions session with ID: {session_id}")
        
        # Prepare response
        response_data = {
            'sessionId': session_id,
            'documentId': document_id,
            'questions': questions,
            'questionType': question_type,
            'difficulty': difficulty,
            'numQuestions': len(questions),
            'fileName': file_name
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(response_data, default=str)
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"Error generating questions: {error_message}")
        
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': error_message,
                'details': 'Question generation failed'
            })
        }