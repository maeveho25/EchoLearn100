// src/components/LearningPage.js
import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const LearningPage = ({ documentId, questions, onBack }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const currentQuestion = questions[currentQuestionIndex];
  const totalQuestions = questions.length;

  // Audio recording setup
  useEffect(() => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          const recorder = new MediaRecorder(stream);
          setMediaRecorder(recorder);

          recorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
              setAudioBlob(event.data);
            }
          };
        })
        .catch(err => console.error('Error accessing microphone:', err));
    }
  }, []);

  const handleMultipleChoiceAnswer = (option) => {
    setAnswers(prev => ({
      ...prev,
      [currentQuestion.id]: {
        type: 'multiple_choice',
        answer: option,
        isCorrect: option === currentQuestion.correct_answer
      }
    }));
  };

  const handleShortAnswer = (answer) => {
    setAnswers(prev => ({
      ...prev,
      [currentQuestion.id]: {
        type: 'short_answer',
        answer: answer,
        keywords: currentQuestion.keywords || []
      }
    }));
  };

  const startRecording = () => {
    if (mediaRecorder && mediaRecorder.state === 'inactive') {
      setIsRecording(true);
      setAudioBlob(null);
      mediaRecorder.start();
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      setIsRecording(false);
      mediaRecorder.stop();
    }
  };

  const handleAudioSubmit = async () => {
    if (!audioBlob) return;

    setIsAnalyzing(true);
    try {
      const audioFile = new File([audioBlob], 'answer.wav', { type: 'audio/wav' });
      const result = await api.analyzeSpeech(audioFile, documentId, currentQuestion.id);
      
      setAnalysisResult(result);
      
      // Store answer with analysis
      setAnswers(prev => ({
        ...prev,
        [currentQuestion.id]: {
          type: 'audio_answer',
          transcript: result.transcript,
          similarityScore: result.similarity_score,
          feedback: result.feedback,
          audioBlob: audioBlob
        }
      }));
      
    } catch (error) {
      console.error('Audio analysis failed:', error);
      alert('Audio analysis failed: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const nextQuestion = () => {
    if (currentQuestionIndex < totalQuestions - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
      setAnalysisResult(null);
      setAudioBlob(null);
    } else {
      setShowResults(true);
    }
  };

  const previousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
      setAnalysisResult(null);
      setAudioBlob(null);
    }
  };

  const calculateResults = () => {
    const totalAnswered = Object.keys(answers).length;
    const correctAnswers = Object.values(answers).filter(answer => 
      answer.type === 'multiple_choice' ? answer.isCorrect : answer.similarityScore > 70
    ).length;

    return {
      totalAnswered,
      correctAnswers,
      percentage: totalAnswered > 0 ? Math.round((correctAnswers / totalAnswered) * 100) : 0
    };
  };

  if (showResults) {
    const results = calculateResults();
    
    return (
      <div className="min-h-screen bg-gradient-to-br from-black via-purple-900/20 to-cyan-900/20 p-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-black/40 backdrop-blur-lg border border-cyan-500/30 rounded-2xl p-8">
            <h2 className="text-3xl font-bold text-cyan-300 mb-6 text-center">
              🎉 Learning Session Complete!
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-gradient-to-r from-cyan-600/20 to-cyan-800/20 p-6 rounded-xl border border-cyan-500/30">
                <h3 className="text-xl font-semibold text-cyan-300 mb-2">Questions Answered</h3>
                <p className="text-3xl font-bold text-white">{results.totalAnswered}/{totalQuestions}</p>
              </div>
              
              <div className="bg-gradient-to-r from-green-600/20 to-green-800/20 p-6 rounded-xl border border-green-500/30">
                <h3 className="text-xl font-semibold text-green-300 mb-2">Correct Answers</h3>
                <p className="text-3xl font-bold text-white">{results.correctAnswers}</p>
              </div>
              
              <div className="bg-gradient-to-r from-purple-600/20 to-purple-800/20 p-6 rounded-xl border border-purple-500/30">
                <h3 className="text-xl font-semibold text-purple-300 mb-2">Success Rate</h3>
                <p className="text-3xl font-bold text-white">{results.percentage}%</p>
              </div>
            </div>

            <div className="space-y-4 mb-8">
              <h3 className="text-xl font-semibold text-cyan-300">Answer Review:</h3>
              {questions.map((question, index) => {
                const answer = answers[question.id];
                if (!answer) return null;

                return (
                  <div key={question.id} className="bg-gray-900/50 p-4 rounded-lg border border-gray-600">
                    <p className="text-cyan-200 font-medium mb-2">
                      {index + 1}. {question.question}
                    </p>
                    
                    {answer.type === 'multiple_choice' && (
                      <div>
                        <p className="text-white">Your answer: {answer.answer}</p>
                        <p className="text-white">Correct answer: {question.correct_answer}</p>
                        <p className={`font-semibold ${answer.isCorrect ? 'text-green-400' : 'text-red-400'}`}>
                          {answer.isCorrect ? '✅ Correct' : '❌ Incorrect'}
                        </p>
                      </div>
                    )}
                    
                    {answer.type === 'audio_answer' && (
                      <div>
                        <p className="text-white">Your answer: "{answer.transcript}"</p>
                        <p className="text-white">Similarity Score: {answer.similarityScore}%</p>
                        <p className="text-gray-300 text-sm">{answer.feedback}</p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="flex justify-center space-x-4">
              <button
                onClick={onBack}
                className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-xl hover:scale-105 transition-all"
              >
                🏠 Back to Upload
              </button>
              
              <button
                onClick={() => {
                  setShowResults(false);
                  setCurrentQuestionIndex(0);
                  setAnswers({});
                  setAnalysisResult(null);
                }}
                className="px-6 py-3 bg-gradient-to-r from-green-600 to-teal-600 text-white rounded-xl hover:scale-105 transition-all"
              >
                🔄 Retry Quiz
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-purple-900/20 to-cyan-900/20 p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <button
            onClick={onBack}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            ← Back
          </button>
          
          <div className="text-center">
            <h1 className="text-3xl font-bold text-cyan-300">EchoLearn Quiz</h1>
            <p className="text-cyan-200">Question {currentQuestionIndex + 1} of {totalQuestions}</p>
          </div>
          
          <div className="text-right">
            <p className="text-cyan-200">Progress: {Math.round(((currentQuestionIndex + 1) / totalQuestions) * 100)}%</p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-700 rounded-full h-2 mb-8">
          <div 
            className="bg-gradient-to-r from-cyan-500 to-purple-500 h-2 rounded-full transition-all duration-500"
            style={{ width: `${((currentQuestionIndex + 1) / totalQuestions) * 100}%` }}
          />
        </div>

        {/* Question Card */}
        <div className="bg-black/40 backdrop-blur-lg border border-cyan-500/30 rounded-2xl p-8 mb-8">
          <h2 className="text-2xl font-semibold text-cyan-300 mb-6">
            {currentQuestion.question}
          </h2>

          {/* Multiple Choice Questions */}
          {currentQuestion.type === 'multiple_choice' && (
            <div className="space-y-4">
              {currentQuestion.options.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleMultipleChoiceAnswer(option)}
                  className={`w-full p-4 text-left rounded-xl border-2 transition-all ${
                    answers[currentQuestion.id]?.answer === option
                      ? 'border-cyan-500 bg-cyan-900/30 text-cyan-200'
                      : 'border-gray-600 bg-gray-900/30 text-gray-200 hover:border-gray-500'
                  }`}
                >
                  {option}
                </button>
              ))}
              
              {answers[currentQuestion.id] && (
                <div className="mt-6 p-4 bg-gray-900/50 rounded-lg">
                  <p className="text-cyan-300 font-medium">Explanation:</p>
                  <p className="text-gray-200">{currentQuestion.explanation}</p>
                </div>
              )}
            </div>
          )}

          {/* Short Answer Questions */}
          {currentQuestion.type === 'short_answer' && (
            <div className="space-y-6">
              {/* Audio Recording */}
              <div className="bg-gray-900/50 p-6 rounded-xl border border-gray-600">
                <h3 className="text-xl font-semibold text-cyan-300 mb-4">
                  🎙️ Record Your Answer
                </h3>
                
                <div className="flex items-center space-x-4 mb-4">
                  <button
                    onClick={startRecording}
                    disabled={isRecording}
                    className={`px-6 py-3 rounded-xl font-medium transition-all ${
                      isRecording 
                        ? 'bg-red-600 text-white cursor-not-allowed' 
                        : 'bg-gradient-to-r from-green-600 to-teal-600 text-white hover:scale-105'
                    }`}
                  >
                    {isRecording ? '🔴 Recording...' : '🎤 Start Recording'}
                  </button>
                  
                  <button
                    onClick={stopRecording}
                    disabled={!isRecording}
                    className="px-6 py-3 bg-gradient-to-r from-red-600 to-pink-600 text-white rounded-xl hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    ⏹️ Stop Recording
                  </button>
                </div>

                {audioBlob && (
                  <div className="space-y-4">
                    <audio controls className="w-full">
                      <source src={URL.createObjectURL(audioBlob)} type="audio/wav" />
                    </audio>
                    
                    <button
                      onClick={handleAudioSubmit}
                      disabled={isAnalyzing}
                      className="px-6 py-3 bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-xl hover:scale-105 transition-all disabled:opacity-50"
                    >
                      {isAnalyzing ? '🔍 Analyzing...' : '🚀 Submit Answer'}
                    </button>
                  </div>
                )}
              </div>

              {/* Analysis Results */}
              {analysisResult && (
                <div className="bg-gradient-to-r from-purple-900/30 to-indigo-900/30 p-6 rounded-xl border border-purple-500/30">
                  <h3 className="text-xl font-semibold text-purple-300 mb-4">
                    🤖 AI Analysis Results
                  </h3>
                  
                  <div className="space-y-3">
                    <div>
                      <p className="text-cyan-300 font-medium">Your Answer:</p>
                      <p className="text-white">"{analysisResult.transcript}"</p>
                    </div>
                    
                    <div>
                      <p className="text-cyan-300 font-medium">Similarity Score:</p>
                      <p className="text-2xl font-bold text-white">
                        {analysisResult.similarity_score}%
                      </p>
                    </div>
                    
                    <div>
                      <p className="text-cyan-300 font-medium">AI Feedback:</p>
                      <p className="text-gray-200">{analysisResult.feedback}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Expected Keywords */}
              {currentQuestion.keywords && (
                <div className="bg-gray-900/50 p-4 rounded-lg">
                  <p className="text-cyan-300 font-medium mb-2">Expected Keywords:</p>
                  <div className="flex flex-wrap gap-2">
                    {currentQuestion.keywords.map((keyword, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-cyan-600/20 text-cyan-200 rounded-full text-sm"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex justify-between items-center">
          <button
            onClick={previousQuestion}
            disabled={currentQuestionIndex === 0}
            className="px-6 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white rounded-xl hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            ← Previous
          </button>

          <div className="text-center">
            <p className="text-cyan-200">
              {Object.keys(answers).length}/{totalQuestions} answered
            </p>
          </div>

          <button
            onClick={nextQuestion}
            disabled={!answers[currentQuestion.id]}
            className="px-6 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-xl hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {currentQuestionIndex === totalQuestions - 1 ? 'Finish' : 'Next'} →
          </button>
        </div>
      </div>
    </div>
  );
};

export default LearningPage;