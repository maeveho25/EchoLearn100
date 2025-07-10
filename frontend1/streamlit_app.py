# streamlit_app.py
# EchoLearn Streamlit Frontend - Production Ready

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
from typing import Dict, List, Optional
import numpy as np
from audio_recorder_streamlit import audio_recorder
import wave

# Page configuration
st.set_page_config(
    page_title="EchoLearn - AI Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
MAX_FILE_SIZE_MB = 10

# Initialize session state
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'current_questions' not in st.session_state:
    st.session_state.current_questions = []
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .success-message {
        padding: 1rem;
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        color: #155724;
        margin: 1rem 0;
    }
    
    .error-message {
        padding: 1rem;
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        color: #721c24;
        margin: 1rem 0;
    }
    
    .warning-message {
        padding: 1rem;
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        color: #856404;
        margin: 1rem 0;
    }
    
    .question-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .performance-excellent {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .performance-good {
        background: #cce7ff;
        color: #004085;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .performance-fair {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .performance-needs-improvement {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    
    .sidebar-logo {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# API Helper Functions
def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_pdf(file_content, filename):
    """Upload PDF to API"""
    try:
        files = {"file": (filename, file_content, "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload-pdf", files=files, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def generate_questions(document_id, difficulty=5, num_questions=3, question_type="multiple_choice"):
    """Generate questions for document"""
    try:
        data = {
            "document_id": document_id,
            "difficulty": difficulty,
            "num_questions": num_questions,
            "question_type": question_type
        }
        response = requests.post(f"{API_BASE_URL}/generate-questions", data=data, timeout=30)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def analyze_speech(audio_data, document_id, question_id=None):
    """Analyze speech using API"""
    try:
        files = {"audio": ("recording.wav", audio_data, "audio/wav")}
        data = {
            "document_id": document_id,
            "question_id": question_id or ""
        }
        response = requests.post(f"{API_BASE_URL}/analyze-speech", files=files, data=data, timeout=120)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_documents():
    """Get all documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_system_stats():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_document_analytics(document_id):
    """Get analytics for specific document"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{document_id}/analytics", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

# Helper Functions
def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        if isinstance(value, str):
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        if isinstance(value, str):
            return int(float(value))
        return int(value)
    except (ValueError, TypeError):
        return default

def show_message(message, message_type="info"):
    """Show styled message"""
    if message_type == "success":
        st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
    elif message_type == "error":
        st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)
    elif message_type == "warning":
        st.markdown(f'<div class="warning-message">{message}</div>', unsafe_allow_html=True)
    else:
        st.info(message)

def format_performance_level(level):
    """Format performance level with styling"""
    level_lower = level.lower()
    if "excellent" in level_lower:
        return f'<span class="performance-excellent">{level}</span>'
    elif "good" in level_lower:
        return f'<span class="performance-good">{level}</span>'
    elif "fair" in level_lower:
        return f'<span class="performance-fair">{level}</span>'
    else:
        return f'<span class="performance-needs-improvement">{level}</span>'

def create_progress_chart(scores):
    """Create progress chart"""
    if not scores:
        return None
    
    df = pd.DataFrame(scores)
    fig = px.line(df, x='session', y='score', 
                  title='Learning Progress Over Time',
                  labels={'score': 'Similarity Score (%)', 'session': 'Session Number'})
    fig.update_layout(showlegend=False)
    return fig

def create_performance_distribution(stats):
    """Create performance distribution chart"""
    if not stats or 'performance_levels' not in stats:
        return None
    
    levels = stats['performance_levels']
    df = pd.DataFrame(list(levels.items()), columns=['Level', 'Count'])
    
    colors = {
        'excellent': '#28a745',
        'good': '#007bff',
        'fair': '#ffc107',
        'needs_improvement': '#dc3545'
    }
    
    fig = px.bar(df, x='Level', y='Count', 
                 title='Performance Distribution',
                 color='Level',
                 color_discrete_map=colors)
    return fig

# Main App Header
st.markdown("""
<div class="main-header">
    <h1>🧠 EchoLearn</h1>
    <p>AI-Powered Learning Assistant with AWS Integration</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🧠 EchoLearn</h2>
        <p>Navigate your learning journey</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.json(health_data)
    
    # Navigation
    page = st.selectbox(
        "Choose a page:",
        ["📚 Upload Documents", "🎯 Learning Session", "📊 Analytics Dashboard", "⚙️ System Status"]
    )

# Page: Upload Documents
if page == "📚 Upload Documents":
    st.header("📚 Upload Learning Materials")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF files for analysis",
        type=['pdf'],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file is not None:
        # File validation
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            show_message(f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB", "error")
        else:
            # Show file info
            st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Process Document", use_container_width=True):
                with st.spinner("Processing document with AI..."):
                    # Upload to API
                    success, result = upload_pdf(uploaded_file.getvalue(), uploaded_file.name)
                    
                    if success:
                        st.session_state.current_document = result
                        show_message("✅ Document processed successfully!", "success")
                        
                        # Show processing results
                        st.subheader("📊 Processing Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📄 Pages", result["processing_results"]["page_count"])
                        
                        with col2:
                            st.metric("📝 Words", result["processing_results"]["word_count"])
                        
                        with col3:
                            st.metric("📏 File Size", f"{result['file_size_mb']:.1f} MB")
                        
                        with col4:
                            analysis = result["processing_results"]["analysis"]
                            difficulty = analysis.get('difficulty_level', 'N/A')
                            st.metric("🎯 Difficulty", f"{difficulty}/10" if difficulty != 'N/A' else 'N/A')
                        
                        # Show analysis
                        st.subheader("🔍 AI Analysis")
                        analysis = result["processing_results"]["analysis"]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Subject:**", analysis.get("subject", "Unknown").title())
                            st.write("**Educational Level:**", analysis.get("educational_level", "Unknown").title())
                            st.write("**Content Type:**", analysis.get("content_type", "Unknown").title())
                        
                        with col2:
                            confidence = safe_float(analysis.get('subject_confidence', 0))
                            st.write("**Confidence:**", f"{confidence:.1%}")
                            st.write("**Has Embeddings:**", "✅" if result["processing_results"]["has_embeddings"] else "❌")
                            st.write("**Embedding Dims:**", result["processing_results"]["embedding_dimensions"])
                        
                        # Show key topics
                        if analysis.get("key_topics"):
                            st.write("**Key Topics:**")
                            for topic in analysis["key_topics"]:
                                st.write(f"• {topic}")
                        
                        # Show summary
                        st.write("**Summary:**")
                        st.write(analysis.get("summary", "No summary available"))
                        
                        # Show text preview
                        with st.expander("📖 Text Preview"):
                            st.text(result["processing_results"]["text_preview"])
                        
                        # Next steps
                        st.subheader("🎯 Next Steps")
                        st.info("📝 Go to **Learning Session** to generate questions and start practicing!")
                        
                    else:
                        show_message(f"❌ Error: {result.get('error', 'Unknown error')}", "error")
    
    # Show existing documents
    st.subheader("📚 Your Documents")
    
    if st.button("🔄 Refresh Documents"):
        success, data = get_documents()
        if success:
            st.session_state.documents = data.get("documents", [])
    
    # Get and display documents
    success, data = get_documents()
    if success:
        documents = data.get("documents", [])
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc.get('FileName', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Subject:**", doc.get('subject', 'Unknown').title())
                        st.write("**Pages:**", doc.get('PageCount', 0))
                        st.write("**Words:**", doc.get('WordCount', 0))
                    
                    with col2:
                        difficulty_level = doc.get('difficulty_level', 'N/A')
                        st.write("**Difficulty:**", f"{difficulty_level}/10" if difficulty_level != 'N/A' else 'N/A')
                        st.write("**Processed:**", doc.get('ProcessedAt', 'Unknown')[:16])
                        
                        # Safe conversion for FileSizeMB
                        file_size = doc.get('FileSizeMB', 0)
                        try:
                            file_size_float = float(file_size)
                            st.write("**Size:**", f"{file_size_float:.1f} MB")
                        except (ValueError, TypeError):
                            st.write("**Size:**", f"{file_size} MB")
                    
                    if st.button(f"📖 Use This Document", key=f"use_{doc['DocumentId']}"):
                        st.session_state.current_document = {
                            "document_id": doc['DocumentId'],
                            "filename": doc.get('FileName', 'Unknown'),
                            "processing_results": {
                                "analysis": doc
                            }
                        }
                        st.success(f"✅ Selected: {doc.get('FileName')}")
                        st.rerun()
        else:
            st.info("📭 No documents uploaded yet. Upload your first PDF above!")
    else:
        show_message("❌ Could not load documents", "error")

# Page: Learning Session
elif page == "🎯 Learning Session":
    st.header("🎯 Interactive Learning Session")
    
    # Check if document is selected
    if not st.session_state.current_document:
        show_message("📚 Please upload and select a document first!", "warning")
        st.stop()
    
    # Document info
    current_doc = st.session_state.current_document
    st.success(f"📄 **Current Document:** {current_doc.get('filename', 'Unknown')}")
    
    # Question generation settings
    st.subheader("⚙️ Question Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        difficulty = st.slider("🎯 Difficulty Level", 1, 10, 5)
    
    with col2:
        num_questions = st.slider("📝 Number of Questions", 1, 10, 3)
    
    with col3:
        question_type = st.selectbox(
            "❓ Question Type",
            ["multiple_choice", "short_answer", "mixed"]
        )
    
    # Generate questions
    if st.button("🎲 Generate Questions", use_container_width=True):
        with st.spinner("Generating questions with AI..."):
            success, result = generate_questions(
                current_doc["document_id"],
                difficulty=difficulty,
                num_questions=num_questions,
                question_type=question_type
            )
            
            if success:
                st.session_state.current_questions = result["questions"]
                st.session_state.current_session = result["session_id"]
                show_message("✅ Questions generated successfully!", "success")
            else:
                show_message(f"❌ Error: {result.get('error', 'Unknown error')}", "error")
    
    # Display questions
    if st.session_state.current_questions:
        st.subheader("📝 Generated Questions")
        
        for i, question in enumerate(st.session_state.current_questions, 1):
            st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
            st.write(f"**Question {i}:**")
            st.write(question["question"])
            
            # Multiple choice options
            if question.get("options"):
                st.write("**Options:**")
                for option in question["options"]:
                    st.write(f"  {option}")
                
                if question.get("correct_answer"):
                    with st.expander("Show Answer"):
                        st.write(f"**Correct:** {question['correct_answer']}")
                        if question.get("explanation"):
                            st.write(f"**Explanation:** {question['explanation']}")
            
            # Short answer
            elif question.get("expected_answer"):
                with st.expander("Show Expected Answer"):
                    st.write(f"**Expected:** {question['expected_answer']}")
                    if question.get("keywords"):
                        st.write(f"**Keywords:** {', '.join(question['keywords'])}")
            
            # Audio recording for this question
            st.write("🎤 **Record your answer:**")
            audio_data = audio_recorder(
                text="Click to record",
                recording_color="#e8b62c",
                neutral_color="#6aa36f",
                icon_name="microphone",
                icon_size="2x",
                key=f"recorder_{i}"
            )
            
            if audio_data:
                st.audio(audio_data, format="audio/wav")
                
                if st.button(f"🔍 Analyze Answer", key=f"analyze_{i}"):
                    with st.spinner("Analyzing your answer..."):
                        success, result = analyze_speech(
                            audio_data,
                            current_doc["document_id"],
                            question["id"]
                        )
                        
                        if success:
                            # Store result
                            st.session_state.analysis_results.append(result)
                            
                            # Display analysis
                            st.subheader("📊 Analysis Results")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                similarity_score = safe_float(result.get('similarity_percentage', 0))
                                st.metric("🎯 Similarity Score", f"{similarity_score:.1f}%")
                            
                            with col2:
                                performance_html = format_performance_level(result['performance_level'])
                                st.markdown(f"**Performance:** {performance_html}", unsafe_allow_html=True)
                            
                            with col3:
                                st.metric("📝 Words Spoken", len(result['transcript'].split()))
                            
                            # Show transcript
                            st.write("**🗣️ Your Answer:**")
                            st.info(result['transcript'])
                            
                            # Show AI feedback
                            st.write("**🤖 AI Feedback:**")
                            st.success(result['feedback'])
                            
                        else:
                            show_message(f"❌ Analysis failed: {result.get('error', 'Unknown error')}", "error")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

# Page: Analytics Dashboard
elif page == "📊 Analytics Dashboard":
    st.header("📊 Analytics Dashboard")
    
    # Get system stats
    success, stats_data = get_system_stats()
    
    if success:
        stats = stats_data.get("stats", {})
        
        # Overview metrics
        st.subheader("📈 Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Documents", stats.get("total_documents", 0))
        
        with col2:
            st.metric("🎯 Total Sessions", stats.get("total_sessions", 0))
        
        with col3:
            st.metric("❓ Questions Generated", stats.get("total_questions_generated", 0))
        
        with col4:
            avg_performance = safe_float(stats.get('average_similarity_score', 0))
            st.metric("🏆 Avg Performance", f"{avg_performance:.1f}%")
        
        # Performance distribution
        if stats.get("performance_levels"):
            st.subheader("🎯 Performance Distribution")
            perf_chart = create_performance_distribution(stats)
            if perf_chart:
                st.plotly_chart(perf_chart, use_container_width=True)
        
        # Subject distribution
        if stats.get("subjects_distribution"):
            st.subheader("📚 Subject Distribution")
            subjects = stats["subjects_distribution"]
            
            df = pd.DataFrame(list(subjects.items()), columns=['Subject', 'Count'])
            fig = px.pie(df, values='Count', names='Subject', title='Documents by Subject')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        if stats.get("last_activity"):
            st.write(f"**Last Activity:** {stats['last_activity'][:16]}")
        
        # Storage usage
        st.subheader("💾 Storage Usage")
        col1, col2 = st.columns(2)
        
        with col1:
            storage_size = safe_float(stats.get('total_file_size_mb', 0))
            st.metric("📏 Total Storage", f"{storage_size:.1f} MB")
        
        with col2:
            total_words = safe_int(stats.get('total_words_processed', 0))
            st.metric("📝 Words Processed", f"{total_words:,}")
        
        # Progress over time (if we have analysis results)
        if st.session_state.analysis_results:
            st.subheader("📈 Your Progress")
            
            scores = []
            for i, result in enumerate(st.session_state.analysis_results, 1):
                similarity_score = safe_float(result.get('similarity_percentage', 0))
                scores.append({
                    'session': i,
                    'score': similarity_score
                })
            
            progress_chart = create_progress_chart(scores)
            if progress_chart:
                st.plotly_chart(progress_chart, use_container_width=True)
    
    else:
        show_message("❌ Could not load analytics data", "error")

# Page: System Status
elif page == "⚙️ System Status":
    st.header("⚙️ System Status")
    
    # API health check
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("✅ System is healthy!")
        
        # Show detailed status
        st.subheader("🔧 Service Status")
        
        services = health_data.get("services", {})
        for service, status in services.items():
            if "✅" in status:
                st.success(f"**{service.upper()}**: {status}")
            else:
                st.error(f"**{service.upper()}**: {status}")
        
        # Show configuration
        st.subheader("⚙️ Configuration")
        st.json({
            "API URL": API_BASE_URL,
            "Max File Size": f"{MAX_FILE_SIZE_MB} MB",
            "Version": health_data.get("version", "Unknown"),
            "Region": health_data.get("region", "Unknown")
        })
        
    else:
        st.error("❌ System is not healthy!")
        st.json(health_data)
    
    # Test API endpoints
    st.subheader("🧪 Test API Endpoints")
    
    if st.button("🔄 Test All Endpoints"):
        with st.spinner("Testing endpoints..."):
            # Test health
            health_ok, _ = check_api_health()
            st.write(f"**/health**: {'✅' if health_ok else '❌'}")
            
            # Test documents
            docs_ok, _ = get_documents()
            st.write(f"**/documents**: {'✅' if docs_ok else '❌'}")
            
            # Test stats
            stats_ok, _ = get_system_stats()
            st.write(f"**/stats**: {'✅' if stats_ok else '❌'}")
    
    # Show session state (for debugging)
    with st.expander("🔍 Debug Info"):
        st.write("**Session State:**")
        st.json({
            "current_document": bool(st.session_state.current_document),
            "current_questions": len(st.session_state.current_questions),
            "analysis_results": len(st.session_state.analysis_results)
        })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧠 EchoLearn - AI-Powered Learning Assistant</p>
    <p>Built with AWS Bedrock, Transcribe, and Streamlit</p>
    <p>© 2024 EchoLearn Team</p>
</div>
""", unsafe_allow_html=True)