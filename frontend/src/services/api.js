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
      console.log('📡 Upload response headers:', response.headers);

      // Check if response is ok
      if (!response.ok) {
        let errorMessage = `Upload failed (${response.status})`;
        
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errorJson = await response.json();
            errorMessage = errorJson.detail || errorJson.error || errorMessage;
          } else {
            const errorText = await response.text();
            errorMessage = errorText || errorMessage;
          }
        } catch (parseError) {
          console.error('❌ Could not parse error response:', parseError);
          errorMessage = `Upload failed (${response.status}) - ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      // Check if response has content
      const contentLength = response.headers.get('content-length');
      if (contentLength === '0' || contentLength === null) {
        throw new Error('Server returned empty response');
      }

      // Parse JSON response
      let result;
      try {
        const responseText = await response.text();
        console.log('📡 Raw response:', responseText);
        
        if (!responseText) {
          throw new Error('Empty response from server');
        }
        
        result = JSON.parse(responseText);
      } catch (jsonError) {
        console.error('❌ JSON parse error:', jsonError);
        throw new Error('Server returned invalid JSON response');
      }

      console.log('✅ Upload successful:', result);
      
      // Validate response structure
      if (typeof result !== 'object' || result === null) {
        throw new Error('Invalid response format from server');
      }

      // Check for success field
      if (result.success === undefined) {
        // If no success field, check for other indicators
        if (result.document_id && result.filename) {
          result.success = true;
        } else if (result.error || result.detail) {
          result.success = false;
        } else {
          throw new Error('Unexpected response format from server');
        }
      }

      if (!result.success) {
        throw new Error(result.error || result.detail || 'Upload failed');
      }

      return result;

    } catch (error) {
      console.error('❌ Upload error:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to server. Please check if backend is running at ' + API_BASE_URL);
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
        let errorMessage = `Question generation failed (${response.status})`;
        
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errorJson = await response.json();
            errorMessage = errorJson.detail || errorJson.error || errorMessage;
          } else {
            const errorText = await response.text();
            errorMessage = errorText || errorMessage;
          }
        } catch (parseError) {
          errorMessage = `Question generation failed (${response.status}) - ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      // Parse response safely
      let result;
      try {
        const responseText = await response.text();
        if (!responseText) {
          throw new Error('Empty response from server');
        }
        result = JSON.parse(responseText);
      } catch (jsonError) {
        throw new Error('Server returned invalid JSON response');
      }

      console.log('✅ Questions generated:', result);
      
      // Validate response
      if (typeof result !== 'object' || result === null) {
        throw new Error('Invalid response format');
      }

      if (result.success === undefined) {
        if (result.questions && Array.isArray(result.questions)) {
          result.success = true;
        } else {
          result.success = false;
        }
      }

      if (!result.success) {
        throw new Error(result.error || result.detail || 'Question generation failed');
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
        let errorMessage = `Speech analysis failed (${response.status})`;
        
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errorJson = await response.json();
            errorMessage = errorJson.detail || errorJson.error || errorMessage;
          } else {
            const errorText = await response.text();
            errorMessage = errorText || errorMessage;
          }
        } catch (parseError) {
          errorMessage = `Speech analysis failed (${response.status}) - ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      // Parse response safely
      let result;
      try {
        const responseText = await response.text();
        if (!responseText) {
          throw new Error('Empty response from server');
        }
        result = JSON.parse(responseText);
      } catch (jsonError) {
        throw new Error('Server returned invalid JSON response');
      }

      console.log('✅ Speech analyzed:', result);
      
      // Validate response
      if (typeof result !== 'object' || result === null) {
        throw new Error('Invalid response format');
      }

      if (result.success === undefined) {
        if (result.transcript && result.similarity_score !== undefined) {
          result.success = true;
        } else {
          result.success = false;
        }
      }

      if (!result.success) {
        throw new Error(result.error || result.detail || 'Speech analysis failed');
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

  async checkHealth() {
    console.log('🏥 Checking API health...');

    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 10000, // 10 second timeout
      });

      if (!response.ok) {
        throw new Error(`Health check failed (${response.status})`);
      }

      // Parse response safely
      let result;
      try {
        const responseText = await response.text();
        if (!responseText) {
          throw new Error('Empty health check response');
        }
        result = JSON.parse(responseText);
      } catch (jsonError) {
        throw new Error('Invalid health check response');
      }

      console.log('✅ Health check successful:', result);
      
      return result;

    } catch (error) {
      console.error('❌ Health check failed:', error);
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return { 
          status: 'unhealthy', 
          error: 'Cannot connect to server at ' + API_BASE_URL,
          timestamp: new Date().toISOString(),
          suggestion: 'Please check if backend is running'
        };
      }
      
      return {
        status: 'unhealthy',
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  },

  async debugEnvironment() {
    console.log('🔍 Checking environment debug info...');

    try {
      const response = await fetch(`${API_BASE_URL}/debug-env`);

      if (!response.ok) {
        throw new Error(`Debug request failed (${response.status})`);
      }

      const result = await response.json();
      console.log('✅ Environment debug successful:', result);
      
      return result;

    } catch (error) {
      console.error('❌ Environment debug failed:', error);
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
        let errorMessage = `AI test failed (${response.status})`;
        
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errorJson = await response.json();
            errorMessage = errorJson.detail || errorJson.error || errorMessage;
          } else {
            const errorText = await response.text();
            errorMessage = errorText || errorMessage;
          }
        } catch (parseError) {
          errorMessage = `AI test failed (${response.status}) - ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      // Parse response safely
      let result;
      try {
        const responseText = await response.text();
        if (!responseText) {
          throw new Error('Empty AI test response');
        }
        result = JSON.parse(responseText);
      } catch (jsonError) {
        throw new Error('Invalid AI test response');
      }

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