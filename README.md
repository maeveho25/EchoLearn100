source venv/bin/activate && pip install fastapi python-multipart uvicorn boto3 python-dotenv

cd backend/app
uvicorn main:app --reload --host 0.0.0.0 --port 8000

streamlit run streamlit_app.py