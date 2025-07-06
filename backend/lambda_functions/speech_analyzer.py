# speech_analyzer.py
import json
import boto3
import numpy as np
from datetime import datetime
import uuid
import time
import urllib.request

def lambda_handler(event, context):
    """
    Analyze student's spoken answer:
    1. Get audio file from S3
    2. Transcribe using AWS Transcribe
    3. Generate embeddings and calculate similarity
    4. Generate feedback using Claude
    5. Store results and return analysis
    """
    try:
        print(f"Received event: {json.dumps(event, default=str)}")
        
        # Initialize AWS clients
        transcribe_client = boto3.client('transcribe')
        bedrock_client = boto3.client('bedrock-runtime')
        dynamodb = boto3.resource('dynamodb')
        s3_client = boto3.client('s3')
        
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        audio_s3_key = body.get('audioKey')
        document_id = body.get('documentId')
        question_id = body.get('questionId')
        bucket_name = body.get('bucket')
        
        if not all([audio_s3_key, document_id, bucket_name]):
            raise ValueError("Missing required parameters: audioKey, documentId, bucket")
        
        print(f"Analyzing audio: {audio_s3_key} for document: {document_id}")
        
        # Create unique transcription job name
        job_name = f"echolearn-{str(uuid.uuid4())[:8]}-{int(time.time())}"
        audio_uri = f's3://{bucket_name}/{audio_s3_key}'
        
        # Start transcription job
        try:
            transcribe_response = transcribe_client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': audio_uri},
                MediaFormat='wav',  # Assuming WAV format
                LanguageCode='en-US',
                Settings={
                    'ShowSpeakerLabels': False,
                    'MaxSpeakerLabels': 1,
                    'VocabularyFilterMethod': 'remove'
                }
            )
            print(f"Started transcription job: {job_name}")
            
        except Exception as e:
            raise Exception(f"Failed to start transcription job: {str(e)}")
        
        # Poll for transcription completion
        max_wait_time = 180  # 3 minutes timeout
        wait_interval = 3    # Check every 3 seconds
        waited = 0
        transcript_text = ""
        
        print("Waiting for transcription to complete...")
        
        while waited < max_wait_time:
            time.sleep(wait_interval)
            waited += wait_interval
            
            try:
                status_response = transcribe_client.get_transcription_job(
                    TranscriptionJobName=job_name
                )
                
                job_status = status_response['TranscriptionJob']['TranscriptionJobStatus']
                print(f"Transcription status: {job_status} (waited {waited}s)")
                
                if job_status == 'COMPLETED':
                    # Get transcript from S3 URI
                    transcript_uri = status_response['TranscriptionJob']['Transcript']['TranscriptFileUri']
                    
                    with urllib.request.urlopen(transcript_uri) as response:
                        transcript_data = json.loads(response.read().decode())
                    
                    # Extract transcript text
                    if transcript_data.get('results', {}).get('transcripts'):
                        transcript_text = transcript_data['results']['transcripts'][0]['transcript']
                        print(f"Transcription completed: '{transcript_text}'")
                    else:
                        raise Exception("No transcript found in transcription results")
                    
                    break
                    
                elif job_status == 'FAILED':
                    failure_reason = status_response['TranscriptionJob'].get('FailureReason', 'Unknown error')
                    raise Exception(f"Transcription failed: {failure_reason}")
                    
            except Exception as e:
                print(f"Error checking transcription status: {e}")
                continue
        
        if not transcript_text:
            raise Exception(f"Transcription timeout after {max_wait_time} seconds")
        
        # Clean up transcription job
        try:
            transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            print("Cleaned up transcription job")
        except Exception as e:
            print(f"Warning: Could not delete transcription job: {e}")
        
        # Get document content for comparison
        documents_table = dynamodb.Table('EchoLearn-Documents')
        doc_response = documents_table.get_item(Key={'DocumentId': document_id})
        
        if 'Item' not in doc_response:
            raise ValueError(f"Document not found: {document_id}")
        
        document = doc_response['Item']
        document_text = document['Text']
        document_title = document.get('FileName', 'Unknown Document')
        
        print(f"Retrieved document: {document_title}")
        
        # Generate embeddings for semantic similarity
        def get_embedding(text_input):
            """Generate embedding using Bedrock Titan"""
            try:
                response = bedrock_client.invoke_model(
                    modelId='amazon.titan-embed-text-v1',
                    body=json.dumps({
                        'inputText': text_input[:7000]  # Limit to avoid token limits
                    })
                )
                
                result = json.loads(response['body'].read())
                return result.get('embedding', [])
                
            except Exception as e:
                print(f"Error generating embedding: {e}")
                return []
        
        # Get embeddings for spoken answer
        spoken_embedding = get_embedding(transcript_text)
        
        # Use stored document embeddings or generate new ones
        if 'Embeddings' in document and document['Embeddings'] and len(document['Embeddings']) > 0:
            document_embedding = document['Embeddings']
            print("Using stored document embeddings")
        else:
            print("Generating new document embeddings")
            # Process document in chunks if too large
            document_text_chunk = document_text[:4000]  # Smaller chunk for reliability
            document_embedding = get_embedding(document_text_chunk)
            
            # Update document with new embeddings
            if document_embedding:
                try:
                    documents_table.update_item(
                        Key={'DocumentId': document_id},
                        UpdateExpression='SET Embeddings = :emb',
                        ExpressionAttributeValues={':emb': document_embedding}
                    )
                    print("Stored new document embeddings")
                except Exception as e:
                    print(f"Warning: Could not store embeddings: {e}")
        
        # Calculate cosine similarity
        similarity_score = 0.0
        if spoken_embedding and document_embedding:
            try:
                def cosine_similarity(vec_a, vec_b):
                    """Calculate cosine similarity between two vectors"""
                    a_np = np.array(vec_a, dtype=np.float32)
                    b_np = np.array(vec_b, dtype=np.float32)
                    
                    # Calculate dot product and norms
                    dot_product = np.dot(a_np, b_np)
                    norm_a = np.linalg.norm(a_np)
                    norm_b = np.linalg.norm(b_np)
                    
                    # Avoid division by zero
                    if norm_a == 0 or norm_b == 0:
                        return 0.0
                    
                    return dot_product / (norm_a * norm_b)
                
                similarity_score = cosine_similarity(spoken_embedding, document_embedding)
                similarity_percentage = max(0, min(100, similarity_score * 100))
                
                print(f"Calculated similarity: {similarity_percentage:.2f}%")
                
            except Exception as e:
                print(f"Error calculating similarity: {e}")
                similarity_percentage = 0.0
        else:
            print("Could not calculate similarity - missing embeddings")
            similarity_percentage = 0.0
        
        # Generate feedback using Claude
        feedback_text = ""
        try:
            feedback_prompt = f"""
You are an AI tutor providing constructive feedback to a student. Here's the context:

Study Material: {document_text[:1000]}...

Student's Spoken Answer: "{transcript_text}"

Semantic Similarity Score: {similarity_percentage:.1f}%

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

Format as plain text, no special formatting.
"""
            
            feedback_response = bedrock_client.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps({
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 400,
                    'messages': [
                        {
                            'role': 'user',
                            'content': feedback_prompt
                        }
                    ]
                })
            )
            
            feedback_result = json.loads(feedback_response['body'].read())
            feedback_text = feedback_result['content'][0]['text'].strip()
            print("Generated AI feedback")
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            feedback_text = f"I can see you mentioned: '{transcript_text}'. This shows engagement with the material. Keep studying and try to elaborate more on the key concepts next time!"
        
        # Store analysis results
        analysis_id = str(uuid.uuid4())
        sessions_table = dynamodb.Table('EchoLearn-Sessions')
        
        analysis_item = {
            'SessionId': analysis_id,
            'DocumentId': document_id,
            'QuestionId': question_id or 'general',
            'Type': 'speech_analysis',
            'Transcript': transcript_text,
            'SimilarityScore': float(similarity_percentage),
            'Feedback': feedback_text,
            'AudioKey': audio_s3_key,
            'CreatedAt': datetime.utcnow().isoformat(),
            'ProcessingTime': waited,
            'Status': 'completed'
        }
        
        sessions_table.put_item(Item=analysis_item)
        print(f"Stored analysis results with ID: {analysis_id}")
        
        # Prepare response
        response_data = {
            'analysisId': analysis_id,
            'transcript': transcript_text,
            'similarityScore': float(similarity_percentage),
            'feedback': feedback_text,
            'documentId': document_id,
            'questionId': question_id,
            'audioKey': audio_s3_key,
            'processingTime': waited,
            'documentTitle': document_title
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
        print(f"Error in speech analysis: {error_message}")
        
        # Clean up transcription job on error
        try:
            if 'job_name' in locals():
                transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
        except:
            pass
        
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
                'details': 'Speech analysis failed'
            })
        }