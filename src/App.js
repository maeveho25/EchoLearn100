import React, { useState, useEffect } from 'react';
import { api } from './services/api';

// Mock API service for demonstration
// const api = {
//     checkHealth: () => Promise.resolve({ status: 'healthy' }),
//     uploadPDF: (file) => Promise.resolve({ success: true, document_id: 'doc123' }),
//     generateQuestions: (docId) => Promise.resolve({
//       questions: [
//         { question: "What is the main concept discussed in the document?", options: ["A) Machine Learning", "B) Data Science", "C) AI Ethics", "D) Neural Networks"] },
//         { question: "Which methodology is primarily used?", options: ["A) Supervised Learning", "B) Unsupervised Learning", "C) Reinforcement Learning", "D) Deep Learning"] },
//         { question: "What are the key benefits mentioned?", options: ["A) Efficiency", "B) Accuracy", "C) Scalability", "D) All of the above"] }
//       ],
//       difficulty: "Intermediate",
//       question_type: "Multiple Choice"
//     })
//   };

const FileUpload = ({ onUpload, isLoading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

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
    if (files.length > 0) {
      onUpload(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      onUpload(file);
    }
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
          Neural Document Processor
        </h3>
        <p className="text-cyan-200 mb-6 text-lg">
          Drag & drop your PDF files into the Analysis processing chamber
        </p>

        <input
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
          id="file-upload"
          disabled={isLoading}
        />
        <label
          htmlFor="file-upload"
          className="inline-block px-8 py-4 bg-gradient-to-r from-cyan-600 to-purple-600 text-white font-bold rounded-xl cursor-pointer transform transition-all duration-300 hover:scale-110 hover:shadow-[0_0_30px_rgba(0,229,255,0.6)] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <div className="flex items-center space-x-2">
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Processing...</span>
            </div>
          ) : (
            'Select File'
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
      }, 30);
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
          <span className="text-cyan-400 font-mono text-sm">Analysis_output.log</span>
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

function App() {
  const [generatedText, setGeneratedText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);
  const [particles, setParticles] = useState([]);

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
      const health = await api.checkHealth();
      setApiHealth(health);
    } catch (error) {
      console.error('API health check failed:', error);
      setApiHealth({ status: 'unhealthy' });
    }
  };

  const handleFileUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    setGeneratedText('');
    try {
      const uploadResult = await api.uploadPDF(file);
      if (uploadResult.success) {
        const questionsResult = await api.generateQuestions(uploadResult.document_id);
        const formattedText = `Analysis Complete!\n\n Your Generated Questions:\n${questionsResult.questions
          .map((q, i) => `\n${i + 1}. ${q.question}\n   ${q.options ? q.options.join('\n   ') : ''}`)
          .join('\n')}\n\n Difficulty Level: ${questionsResult.difficulty}\n🎯 Classification: ${questionsResult.question_type}`;
        setGeneratedText(formattedText);
      } else {
        throw new Error(uploadResult.error || 'Neural processing failed');
      }
    } catch (error) {
      console.error('Error processing file:', error);
      setError(error.message || 'Analysis processing error. Reinitializing neural pathways...');
    }
    setIsLoading(false);
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
          {/* API Health Warning */}
          {apiHealth && apiHealth.status !== 'healthy' && (
            <div className="mb-8 p-4 bg-yellow-900/30 border border-yellow-500/50 rounded-lg backdrop-blur-sm animate-pulse">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-yellow-500 rounded-full animate-ping"></div>
                <span className="text-yellow-300">⚠️ Analysis servers are recalibrating. Some features may be temporarily unavailable.</span>
              </div>
            </div>
          )}

          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent animate-glow">
              EchoLearn
            </h1>
            <div className="text-2xl text-cyan-300 mb-2 animate-pulse">
              Neural Intelligence • Infinite Learning • Empowering Thought
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
                Analysis neural networks are analyzing your document...
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

          {/* Output Display */}
          {(generatedText || error) && (
            <div className="animate-slide-up">
              <TextOutput text={error || generatedText} error={!!error} />
            </div>
          )}
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