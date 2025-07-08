const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8004';

export const api = {
  async uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    return response.json();
  },

  async generateQuestions(documentId, options = {}) {
    const formData = new FormData();
    formData.append('document_id', documentId);
    formData.append('difficulty', options.difficulty || 5);
    formData.append('num_questions', options.numQuestions || 3);
    formData.append('question_type', options.questionType || 'multiple_choice');

    const response = await fetch(`${API_BASE_URL}/generate-questions`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Failed to generate questions');
    }

    return response.json();
  },

  async analyzeSpeech(audioFile, documentId, questionId = null) {
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
      throw new Error('Speech analysis failed');
    }

    return response.json();
  },

  async listDocuments() {
    const response = await fetch(`${API_BASE_URL}/documents`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch documents');
    }

    return response.json();
  },

  async checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    
    if (!response.ok) {
      throw new Error('Health check failed');
    }

    return response.json();
  }
};
