// src/App.js
import React, { useState, useEffect } from 'react';
import { api } from './services/api';

const FileUpload = ({ onUpload, isLoading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const validateFile = (file) => {
    console.log('🔍 Validating file:', file);

    if (!file) {
      alert('No file selected');
      return false;
    }

    if (file.type !== 'application/pdf') {
      alert('Please select a PDF file only');
      return false;
    }

    if (file.size === 0) {
      alert('File is empty');
      return false;
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('File too large. Maximum size is 10MB');
      return false;
    }

    if (file.size < 1000) {
      alert('File too small. Please upload a valid PDF');
      return false;
    }

    console.log('✅ File validation passed:', {
      name: file.name,
      type: file.type,
      size: `${Math.round(file.size / 1024)}KB`
    });

    return true;
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    console.log('📁 Files dropped:', files);
    
    if (files.length > 0) {
      const file = files[0];
      if (validateFile(file)) {
        onUpload(file);
      }
    } else {
      alert('No files dropped');
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    console.log('📂 File selected:', file);
    
    if (file && validateFile(file)) {
      onUpload(file);
    }
    
    // Reset input to allow same file upload again
    e.target.value = '';
  };

  return (
    <div
      className={`relative overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-500 transform hover:scale-105 ${
        isDragOver 
          ? 'border-cyan-400 bg-cyan-900/30 shadow-[0_0_30px_rgba(0,229,255,0.4)]' 
          : isHovered
          ? 'border-cyan-500 bg-cyan-900/20 shadow-[0_0_20px_rgba(0,229,255,0.3)]'
          : 'border-cyan-700 bg-cyan-900/10'
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Animated Background Grid */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 animate-pulse"></div>
        <div className="grid grid-cols-8 grid-rows-8 h-full w-full">
          {Array.from({ length: 64 }).map((_, i) => (
            <div
              key={i}
              className="border border-cyan-700/30 animate-pulse"
              style={{
                animationDelay: `${i * 0.1}s`,
                animationDuration: '3s'
              }}
            />
          ))}
        </div>
      </div>

      {/* Scanning Lines */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-scan-vertical"></div>
        <div className="absolute w-1 h-full bg-gradient-to-b from-transparent via-cyan-400 to-transparent animate-scan-horizontal"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 p-12 text-center">
        <div className="mb-6">
          <div className="inline-block p-4 rounded-full bg-cyan-500/20 animate-pulse">
            <svg className="w-16 h-16 text-cyan-400 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
        </div>

        <h3 className="text-2xl font-bold text-cyan-300 mb-4 animate-glow">
          Real PDF Document Processor
        </h3>
        <p className="text-cyan-200 mb-4 text-lg">
          Drag & drop your PDF files or click to select
        </p>
        <p className="text-cyan-300 mb-6 text-sm">
          📄 PDF only • 🎯 Max 10MB • 🚀 AI-powered analysis
        </p>

        <input
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileSelect}
          className="hidden"
          id="file-upload"
          disabled={isLoading}
        />
        <label
          htmlFor="file-upload"
          className={`inline-block px-8 py-4 bg-gradient-to-r from-cyan-600 to-purple-600 text-white font-bold rounded-xl cursor-pointer transform transition-all duration-300 hover:scale-110 hover:shadow-[0_0_30px_rgba(0,229,255,0.6)] ${
            isLoading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Processing PDF...</span>
            </div>
          ) : (
            '📄 Select PDF File'
          )}
        </label>
      </div>
    </div>
  );
};

const TextOutput = ({ text, error }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (text) {
      setDisplayText('');
      setCurrentIndex(0);
    }
  }, [text]);

  useEffect(() => {
    if (text && currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, 20); // Faster typing effect
      return () => clearTimeout(timer);
    }
  }, [text, currentIndex]);

  return (
    <div className="relative">
      {/* Holographic border effect */}
      <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-cyan-500/20 to-purple-500/20 blur-sm animate-pulse"></div>
      
      <div className="relative p-6 rounded-xl bg-black/40 backdrop-blur-lg border border-cyan-500/30">
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
          <div className="w-3 h-3 bg-yellow-500 rounded-full animate-pulse" style={{ animationDelay: '0.3s' }}></div>
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" style={{ animationDelay: '0.6s' }}></div>
          <span className="text-cyan-400 font-mono text-sm">
            {error ? 'error_log.txt' : 'analysis_output.log'}
          </span>
        </div>

        <div className="font-mono text-sm leading-relaxed whitespace-pre-wrap">
          <span className={error ? 'text-red-400' : 'text-cyan-300'}>
            {displayText}
          </span>
          {currentIndex < text.length && (
            <span className="inline-block w-2 h-5 bg-cyan-400 animate-pulse ml-1"></span>
          )}
        </div>
      </div>
    </div>
  );
};

const StatusIndicator = ({ apiHealth }) => {
  if (!apiHealth) return null;

  const isHealthy = apiHealth.status === 'healthy';
  
  return (
    <div className={`mb-8 p-4 rounded-lg backdrop-blur-sm border ${
      isHealthy 
        ? 'bg-green-900/30 border-green-500/50' 
        : 'bg-yellow-900/30 border-yellow-500/50'
    }`}>
      <div className="flex items-center space-x-2">
        <div className={`w-4 h-4 rounded-full ${
          isHealthy ? 'bg-green-500 animate-pulse' : 'bg-yellow-500 animate-ping'
        }`}></div>
        <span className={isHealthy ? 'text-green-300' : 'text-yellow-300'}>
          {isHealthy 
            ? '✅ System operational - All services running' 
            : '⚠️ System recalibrating - Some features may be temporarily unavailable'
          }
        </span>
      </div>
      
      {apiHealth.services && (
        <div className="mt-2 text-xs text-gray-400">
          Services: {Object.entries(apiHealth.services).map(([service, status]) => (
            <span key={service} className="mr-2">
              {service}: {typeof status === 'string' ? status : 'ready'}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

function App() {
  const [generatedText, setGeneratedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [particles, setParticles] = useState([]);
  const [uploadedDocument, setUploadedDocument] = useState(null);

  useEffect(() => {
    checkApiHealth();
    generateParticles();
  }, []);

  const generateParticles = () => {
    const newParticles = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 5,
      duration: 3 + Math.random() * 4
    }));
    setParticles(newParticles);
  };

  const checkApiHealth = async () => {
    try {
      console.log('🏥 Checking API health...');
      const health = await api.checkHealth();
      setApiHealth(health);
      console.log('✅ API health check completed:', health);
    } catch (error) {
      console.error('❌ API health check failed:', error);
      setApiHealth({ 
        status: 'unhealthy', 
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  };

  const handleFileUpload = async (file) => {
    console.log('🎯 Starting file upload process...', file);
    
    setIsLoading(true);
    setError(null);
    setGeneratedText('');
    setUploadedDocument(null);
    
    try {
      // Step 1: Upload PDF
      console.log('📤 Step 1: Uploading PDF...');
      const uploadResult = await api.uploadPDF(file);
      console.log('✅ Upload completed:', uploadResult);
      
      if (!uploadResult.success) {
        throw new Error(uploadResult.error || 'Upload failed');
      }

      // Store uploaded document info
      setUploadedDocument(uploadResult);

      // Show upload success immediately
      const uploadSuccessText = `✅ PDF Upload Successful!

📄 File: ${uploadResult.filename}
📊 Size: ${Math.round(uploadResult.file_size / 1024)} KB
🆔 Document ID: ${uploadResult.document_id}

📝 Processing Results:
   • Word Count: ${uploadResult.processing_results?.word_count || 'N/A'}
   • Pages: ${uploadResult.processing_results?.page_count || 'N/A'}
   • Subject: ${uploadResult.processing_results?.subject || 'General'}
   • Difficulty: ${uploadResult.processing_results?.difficulty || 5}/10

💡 Content Preview:
${uploadResult.processing_results?.text_preview || 'No preview available'}

🤖 Generating AI questions...`;

      setGeneratedText(uploadSuccessText);

      // Step 2: Auto-generate questions
      console.log('🤖 Step 2: Generating questions...');
      
      setTimeout(async () => {
        try {
          const questionsResult = await api.generateQuestions(uploadResult.document_id, {
            difficulty: uploadResult.processing_results?.difficulty || 5,
            numQuestions: 5,
            questionType: 'multiple_choice'
          });
          
          console.log('✅ Questions generated:', questionsResult);

          if (!questionsResult.success) {
            throw new Error(questionsResult.error || 'Question generation failed');
          }

          // Add questions to display
          const questionsText = `

🤖 AI Generated Questions (${questionsResult.total_questions} questions):

${questionsResult.questions
  .map((q, i) => {
    if (q.type === 'multiple_choice') {
      return `${i + 1}. ${q.question}
   ${q.options ? q.options.join('\n   ') : ''}
   
   ✅ Correct: ${q.correct_answer}
   💡 ${q.explanation || 'No explanation provided'}`;
    } else {
      return `${i + 1}. ${q.question}
   
   💡 Expected keywords: ${q.keywords ? q.keywords.join(', ') : 'N/A'}`;
    }
  })
  .join('\n\n')}

📊 Generation Details:
   • Difficulty: ${questionsResult.difficulty}/10
   • Question Type: ${questionsResult.question_type}
   • AI Model: ${questionsResult.model_used || 'Claude 3.5 Sonnet'}
   • Subject: ${questionsResult.document_info?.subject || 'General'}

🎉 Ready for learning assessment!`;

          setGeneratedText(prevText => prevText + questionsText);
          
        } catch (questionError) {
          console.error('❌ Question generation failed:', questionError);
          const errorText = `

❌ Question Generation Failed:
${questionError.message}

The PDF was uploaded successfully, but AI question generation encountered an error.
Please try again or check the system status.`;
          
          setGeneratedText(prevText => prevText + errorText);
        }
      }, 2000); // 2 second delay for better UX

    } catch (error) {
      console.error('❌ Upload process failed:', error);
      
      let errorMessage = error.message;
      
      if (error.message.includes('Cannot connect to server')) {
        errorMessage += '\n\n🔧 Troubleshooting:\n• Check if backend is running (python main.py)\n• Verify API URL in .env file\n• Check network connection';
      } else if (error.message.includes('AWS')) {
        errorMessage += '\n\n🔧 AWS Configuration Issue:\n• Check AWS credentials\n• Verify S3 bucket exists\n• Check AWS region settings';
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    setError(null);
    setGeneratedText('');
    checkApiHealth();
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-black">
      {/* Animated Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-black via-purple-900/20 to-cyan-900/20"></div>
      
      {/* Floating Particles */}
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="absolute w-1 h-1 bg-cyan-400 rounded-full animate-float"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            animationDelay: `${particle.delay}s`,
            animationDuration: `${particle.duration}s`,
            boxShadow: '0 0 6px currentColor'
          }}
        />
      ))}

      {/* Matrix Rain Effect */}
      <div className="absolute inset-0 opacity-10">
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={i}
            className="absolute top-0 w-px bg-gradient-to-b from-cyan-400 to-transparent animate-matrix-rain"
            style={{
              left: `${i * 5}%`,
              animationDelay: `${i * 0.5}s`,
              animationDuration: `${3 + i * 0.2}s`
            }}
          />
        ))}
      </div>

      <div className="relative z-10 min-h-screen py-12 px-4">
        <div className="max-w-4xl mx-auto">
          {/* API Health Status */}
          <StatusIndicator apiHealth={apiHealth} />

          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent animate-glow">
              EchoLearn
            </h1>
            <div className="text-2xl text-cyan-300 mb-2 animate-pulse">
              Neural Intelligence • Real PDF Processing • AI-Powered Learning
            </div>
            <div className="w-64 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent mx-auto animate-pulse"></div>
          </div>

          {/* Main Upload Interface */}
          <div className="mb-8">
            <FileUpload onUpload={handleFileUpload} isLoading={isLoading} />
          </div>

          {/* Loading Animation */}
          {isLoading && (
            <div className="mb-8 p-8 text-center">
              <div className="inline-block relative">
                <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin"></div>
                <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-r-purple-400 rounded-full animate-spin-reverse"></div>
              </div>
              <p className="text-cyan-300 mt-4 animate-pulse">
                {isLoading && uploadedDocument 
                  ? 'AI neural networks are generating questions...' 
                  : 'Processing PDF with advanced AI analysis...'
                }
              </p>
              <div className="flex justify-center space-x-1 mt-4">
                {Array.from({ length: 3 }).map((_, i) => (
                  <div
                    key={i}
                    className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"
                    style={{ animationDelay: `${i * 0.2}s` }}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Error Display with Retry Option */}
          {error && (
            <div className="mb-8">
              <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-6 backdrop-blur-sm">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-red-500 rounded-full flex-shrink-0 flex items-center justify-center">
                    <span className="text-white text-sm font-bold">!</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-red-300 font-semibold mb-2">Upload Error</h3>
                    <pre className="text-red-200 text-sm whitespace-pre-wrap">{error}</pre>
                    <button 
                      onClick={handleRetry}
                      className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    >
                      🔄 Retry
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Success Output Display */}
          {generatedText && !error && (
            <div className="animate-slide-up">
              <TextOutput text={generatedText} error={false} />
            </div>
          )}

          {/* Action Buttons */}
          {uploadedDocument && !isLoading && (
            <div className="mt-8 flex flex-wrap gap-4 justify-center">
              <button
                onClick={() => handleFileUpload(null)}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl hover:scale-105 transition-all"
              >
                📄 Upload Another PDF
              </button>
              
              <button
                onClick={async () => {
                  try {
                    setIsLoading(true);
                    const questionsResult = await api.generateQuestions(uploadedDocument.document_id, {
                      difficulty: 8,
                      numQuestions: 3,
                      questionType: 'short_answer'
                    });
                    
                    const newText = `\n\n🧠 Advanced Short Answer Questions:\n\n${questionsResult.questions
                      .map((q, i) => `${i + 1}. ${q.question}\n   💡 Expected keywords: ${q.keywords?.join(', ') || 'N/A'}`)
                      .join('\n\n')}`;
                    
                    setGeneratedText(prev => prev + newText);
                  } catch (error) {
                    setError(error.message);
                  } finally {
                    setIsLoading(false);
                  }
                }}
                className="px-6 py-3 bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-xl hover:scale-105 transition-all"
                disabled={isLoading}
              >
                🧠 Generate Advanced Questions
              </button>
              
              <button
                onClick={() => checkApiHealth()}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-cyan-600 text-white rounded-xl hover:scale-105 transition-all"
              >
                🏥 Check System Status
              </button>
            </div>
          )}

          {/* Footer Info */}
          <div className="mt-16 text-center text-gray-400 text-sm">
            <p>🚀 Powered by AWS Bedrock • Claude 3.5 Sonnet • Real PDF Processing</p>
            <p>💡 Upload any PDF document to generate personalized AI questions</p>
            {uploadedDocument && (
              <p className="mt-2 text-cyan-300">
                📊 Last processed: {uploadedDocument.filename} • 
                Subject: {uploadedDocument.processing_results?.subject} • 
                Difficulty: {uploadedDocument.processing_results?.difficulty}/10
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Custom Styles */}
      <style jsx>{`
        @keyframes glow {
          0%, 100% { filter: drop-shadow(0 0 20px currentColor); }
          50% { filter: drop-shadow(0 0 30px currentColor) drop-shadow(0 0 40px currentColor); }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        @keyframes matrix-rain {
          0% { transform: translateY(-100vh); opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { transform: translateY(100vh); opacity: 0; }
        }
        
        @keyframes scan-vertical {
          0% { top: 0; }
          100% { top: 100%; }
        }
        
        @keyframes scan-horizontal {
          0% { left: 0; }
          100% { left: 100%; }
        }
        
        @keyframes slide-up {
          from { transform: translateY(50px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
        
        @keyframes spin-reverse {
          from { transform: rotate(360deg); }
          to { transform: rotate(0deg); }
        }
        
        .animate-glow { animation: glow 2s ease-in-out infinite; }
        .animate-float { animation: float 3s ease-in-out infinite; }
        .animate-matrix-rain { animation: matrix-rain 3s linear infinite; }
        .animate-scan-vertical { animation: scan-vertical 2s linear infinite; }
        .animate-scan-horizontal { animation: scan-horizontal 3s linear infinite; }
        .animate-slide-up { animation: slide-up 0.8s ease-out; }
        .animate-spin-reverse { animation: spin-reverse 1s linear infinite; }
      `}</style>
    </div>
  );
}

export default App;