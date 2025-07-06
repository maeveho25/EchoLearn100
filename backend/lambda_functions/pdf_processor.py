import json
import boto3
import PyPDF2
from io import BytesIO
import uuid
from datetime import datetime

def lambda_handler(event, context):
    """
    Process uploaded PDF file:
    1. Download from S3
    2. Extract text using PyPDF2
    3. Generate embeddings using Bedrock
    4. Store metadata in DynamoDB
    """
    try:
        print(f"Received event: {json.dumps(event, default=str)}")
        
        # Initialize AWS clients
        s3_client = boto3.client('s3')
        bedrock_client = boto3.client('bedrock-runtime')
        dynamodb = boto3.resource('dynamodb')
        
        # Parse event (API Gateway or S3 trigger)
        if 'Records' in event:
            # S3 trigger event
            bucket = event['Records'][0]['s3']['bucket']['name']
            key = event['Records'][0]['s3']['object']['key']
        else:
            # API Gateway event
            if isinstance(event.get('body'), str):
                body = json.loads(event['body'])
            else:
                body = event.get('body', {})
            
            bucket = body.get('bucket')
            key = body.get('key')
            
            if not bucket or not key:
                raise ValueError("Missing bucket or key in request")
        
        print(f"Processing PDF: {key} from bucket: {bucket}")
        
        # Download PDF from S3
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            pdf_content = response['Body'].read()
            print(f"Downloaded PDF, size: {len(pdf_content)} bytes")
        except Exception as e:
            raise Exception(f"Failed to download PDF from S3: {str(e)}")
        
        # Extract text from PDF
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
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
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
        
        # Generate embeddings using Bedrock Titan
        embeddings = []
        try:
            # Limit text to avoid token limits (Titan has ~8000 token limit)
            text_for_embedding = text_content[:7000]
            
            embedding_response = bedrock_client.invoke_model(
                modelId='amazon.titan-embed-text-v1',
                body=json.dumps({
                    'inputText': text_for_embedding
                })
            )
            
            embedding_data = json.loads(embedding_response['body'].read())
            embeddings = embedding_data.get('embedding', [])
            print(f"Generated embeddings with {len(embeddings)} dimensions")
            
        except Exception as e:
            print(f"Warning: Could not generate embeddings: {e}")
            embeddings = []  # Continue without embeddings
        
        # Store document in DynamoDB
        document_id = str(uuid.uuid4())
        documents_table = dynamodb.Table('EchoLearn-Documents')
        
        document_item = {
            'DocumentId': document_id,
            'FileName': key,
            'Text': text_content,
            'Embeddings': embeddings,
            'ProcessedAt': datetime.utcnow().isoformat(),
            'WordCount': word_count,
            'PageCount': page_count,
            'S3Bucket': bucket,
            'S3Key': key,
            'FileSize': len(pdf_content),
            'Status': 'processed'
        }
        
        documents_table.put_item(Item=document_item)
        print(f"Stored document in DynamoDB with ID: {document_id}")
        
        # Prepare response
        response_data = {
            'message': 'PDF processed successfully',
            'documentId': document_id,
            'fileName': key,
            'wordCount': word_count,
            'pageCount': page_count,
            'fileSize': len(pdf_content),
            'preview': text_content[:300] + "..." if len(text_content) > 300 else text_content,
            'hasEmbeddings': len(embeddings) > 0
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        error_message = str(e)
        print(f"Error processing PDF: {error_message}")
        
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
                'details': 'PDF processing failed'
            })
        }