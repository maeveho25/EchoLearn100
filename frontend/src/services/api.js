const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = {
  async uploadPDF(file) {
    console.log('📤 Starting PDF upload:', {
      name: file?.name,
      type: file?.type,
      size: file?.size
    });

    // Comprehensive file validation
    if (!file) {
      throw new Error('No file provided');
    }

    if (!(file instanceof File)) {
      throw new Error('Invalid file object');
    }

    if (file.type !== 'application/pdf') {
      throw new Error('Please select a PDF file only');
    }

    if (file.size === 0) {
      throw new Error('File is empty');
    }

    if (file.size > 10 * 1024 * 1024) {
      throw new Error('File too large. Maximum size is 10MB');
    }

    if (file.size < 1000) {
      throw new Error('File too small. Please upload a valid PDF');
    }

    try {
      // Create FormData with proper validation
      const formData = new FormData();
      formData.append('file', file, file.name);
      
      console.log('📦 FormData created for:', file.name);

      const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type - let browser handle multipart boundary
      });

      console.log('📡 Upload response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Upload failed:', errorText);
        
        let errorMessage = `Upload failed (${response.status})`;
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.error || errorMessage;
        } catch {
          // If not JSON, use raw text
          errorMessage = errorText || errorMessage;
        }
        
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log('✅ Upload successful:', result);
      
      if (!result.success) {
        throw new Error(result.error || 'Upload failed');
      }

      return result;

    } catch (error) {
      console.error('❌ Upload error:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server. Please check if backend is running.');
      }
      
      throw error;
    }
  },

  async generateQuestions(documentId, options = {}) {
    console.log('🤖 Generating questions for document:', documentId);

    if (!documentId) {
      throw new Error('Document ID is required');
    }

    try {
      const formData = new FormData();
      formData.append('document_id', documentId);
      formData.append('difficulty', options.difficulty || 5);
      formData.append('num_questions', options.numQuestions || 3);
      formData.append('question_type', options.questionType || 'multiple_choice');

      console.log('🎯 Question parameters:', {
        document_id: documentId,
        difficulty: options.difficulty || 5,
        num_questions: options.numQuestions || 3,
        question_type: options.questionType || 'multiple_choice'
      });

      const response = await fetch(`${API_BASE_URL}/generate-questions`, {
        method: 'POST',
        body: formData,
      });

      console.log('📡 Question generation response:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Question generation failed:', errorText);
        
        let errorMessage = `Question generation failed (${response.status})`;
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.error || errorMessage;
        } catch {
          errorMessage = errorText || errorMessage;
        }
        
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log('✅ Questions generated:', result);
      
      if (!result.success) {
        throw new Error(result.error || 'Question generation failed');
      }

      return result;

    } catch (error) {
      console.error('❌ Question generation error:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server for question generation');
      }
      
      throw error;
    }
  },

  async analyzeSpeech(audioFile, documentId, questionId = null) {
    console.log('🎙️ Analyzing speech for document:', documentId);

    if (!audioFile) {
      throw new Error('Audio file is required');
    }

    if (!documentId) {
      throw new Error('Document ID is required');
    }

    try {
      const formData = new FormData();
      formData.append('audio', audioFile);
      formData.append('document_id', documentId);
      if (questionId) {
        formData.append('question_id', questionId);
      }

      const response = await fetch(`${API_BASE_URL}/analyze-speech`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Speech analysis failed:', errorText);
        
        let errorMessage = `Speech analysis failed (${response.status})`;
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.error || errorMessage;
        } catch {
          errorMessage = errorText || errorMessage;
        }
        
        throw new Error(errorMessage);
      }

      const result = await response.json();
      console.log('✅ Speech analyzed:', result);
      
      if (!result.success) {
        throw new Error(result.error || 'Speech analysis failed');
      }

      return result;

    } catch (error) {
      console.error('❌ Speech analysis error:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server for speech analysis');
      }
      
      throw error;
    }
  },

  async listDocuments(filters = {}) {
    console.log('📚 Fetching documents with filters:', filters);

    try {
      const params = new URLSearchParams();
      if (filters.subject) params.append('subject', filters.subject);
      if (filters.search) params.append('search', filters.search);
      
      const url = `${API_BASE_URL}/documents${params.toString() ? '?' + params.toString() : ''}`;
      
      const response = await fetch(url);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Document fetch failed:', errorText);
        throw new Error(`Failed to fetch documents (${response.status})`);
      }

      const result = await response.json();
      console.log('✅ Documents fetched:', result);
      
      return result;

    } catch (error) {
      console.error('❌ Document fetch error:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server to fetch documents');
      }
      
      throw error;
    }
  },

  async getDocumentDetails(documentId) {
    console.log('📖 Fetching document details:', documentId);

    if (!documentId) {
      throw new Error('Document ID is required');
    }

    try {
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ Document details fetch failed:', errorText);
        
        if (response.status === 404) {
          throw new Error('Document not found');
        }
        
        throw new Error(`Failed to fetch document details (${response.status})`);
      }

      const result = await response.json();
      console.log('✅ Document details fetched:', result);
      
      return result;

    } catch (error) {
      console.error('❌ Document details error:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server to fetch document details');
      }
      
      throw error;
    }
  },

  async checkHealth() {
    console.log('🏥 Checking API health...');

    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Health check failed (${response.status})`);
      }

      const result = await response.json();
      console.log('✅ Health check successful:', result);
      
      return result;

    } catch (error) {
      console.error('❌ Health check failed:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return { 
          status: 'unhealthy', 
          error: 'Cannot connect to server',
          timestamp: new Date().toISOString()
        };
      }
      
      throw error;
    }
  },

  async testAI(prompt = "What is machine learning?") {
    console.log('🧠 Testing AI with prompt:', prompt);

    try {
      const formData = new FormData();
      formData.append('prompt', prompt);

      const response = await fetch(`${API_BASE_URL}/test-ai`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('❌ AI test failed:', errorText);
        throw new Error(`AI test failed (${response.status})`);
      }

      const result = await response.json();
      console.log('✅ AI test successful:', result);
      
      return result;

    } catch (error) {
      console.error('❌ AI test error:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server for AI test');
      }
      
      throw error;
    }
  }
};