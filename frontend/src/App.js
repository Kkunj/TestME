import React, { useState } from 'react';
import './App.css';

// const API_BASE_URL = 'http://localhost:5000/api';
const API_BASE_URL = '/api';

function App() {
  const [currentStep, setCurrentStep] = useState('upload');
  const [sessionId, setSessionId] = useState('');
  const [fileName, setFileName] = useState('');
  const [mcqs, setMcqs] = useState([]);
  const [answers, setAnswers] = useState({});
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Helper function for API calls
  const apiCall = async (url, options = {}) => {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Request failed');
    }
    
    return await response.json();
  };

  // Step 1: File Upload
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (file.type !== 'application/pdf') {
      setError('Please select a PDF file');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const data = await response.json();
      setSessionId(data.session_id);
      setFileName(data.filename);
      setCurrentStep('processing');
      
      // Immediately start processing
      await processFile(data.session_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Step 2: Process PDF and Generate MCQs
  const processFile = async (sessionId) => {
    setLoading(true);
    try {
      const data = await apiCall(`${API_BASE_URL}/process/${sessionId}`, {
        method: 'POST'
      });
      
      setMcqs(data.mcqs);
      setCurrentStep('quiz');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Step 3: Handle Answer Selection
  const handleAnswerChange = (questionIndex, answer) => {
    setAnswers(prev => ({
      ...prev,
      [questionIndex]: answer
    }));
  };

  // Step 4: Submit Answers
  const submitAnswers = async () => {
    setLoading(true);
    try {
      const data = await apiCall(`${API_BASE_URL}/submit-answers`, {
        method: 'POST',
        body: JSON.stringify({
          session_id: sessionId,
          answers: answers
        })
      });

      setResults(data);
      setCurrentStep('results');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Reset to start over
  const resetApp = () => {
    setCurrentStep('upload');
    setSessionId('');
    setFileName('');
    setMcqs([]);
    setAnswers({});
    setResults(null);
    setError('');
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <h1>TestMe - Question Generator</h1>
          <p>Upload your PDF and get auto-generated multiple choice questions, to test your prep.</p>
        </header>

        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}

        {loading && (
          <div className="alert alert-info">
            <div className="loading-container">
              <div className="spinner"></div>
              Processing...
            </div>
          </div>
        )}

        {/* Step 1: Upload */}
        {currentStep === 'upload' && (
          <div className="card">
            <h2>Upload Your PDF</h2>
            <div className="upload-area">
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="file-input"
                id="file-upload"
                disabled={loading}
              />
              <label htmlFor="file-upload" className="upload-button">
                Choose PDF File
              </label>
              <p className="upload-help">Select a PDF file to generate MCQs</p>
            </div>
          </div>
        )}

        {/* Step 2: Processing */}
        {currentStep === 'processing' && (
          <div className="card text-center">
            <h2>Processing Your PDF</h2>
            <p><strong>File:</strong> {fileName}</p>
            <p>Extracting text, keywords, and generating questions...</p>
          </div>
        )}

        {/* Step 3: Quiz */}
        {currentStep === 'quiz' && (
          <div className="card">
            <h2>Answer the Questions</h2>
            <p className="question-count">Total Questions: {mcqs.length}</p>
            
            <div className="questions-container">
              {mcqs.map((mcq, index) => (
                <div key={index} className="question-block">
                  <h3 className="question-text">
                    {index + 1}. {mcq.question}
                  </h3>
                  <div className="options-container">
                    {Object.entries(mcq.options).map(([key, value]) => (
                      <label key={key} className="option-label">
                        <input
                          type="radio"
                          name={`question-${index}`}
                          value={key}
                          checked={answers[index] === key}
                          onChange={() => handleAnswerChange(index, key)}
                          className="option-input"
                        />
                        <span className="option-text">{key}. {value}</span>
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="submit-container">
              <button
                onClick={submitAnswers}
                disabled={Object.keys(answers).length !== mcqs.length || loading}
                className="submit-button"
              >
                Submit Answers
              </button>
            </div>
          </div>
        )}

        {/* Step 4: Results */}
        {currentStep === 'results' && results && (
          <div className="card">
            <h2>Quiz Results</h2>
            
            <div className="score-card">
              <h3>Score: {results.score.toFixed(1)}%</h3>
              <p>Correct: {results.correct_answers} / {results.total_questions}</p>
            </div>

            <div className="results-container">
              {results.results.map((result, index) => (
                <div key={index} className={`result-item ${result.is_correct ? 'correct' : 'incorrect'}`}>
                  <h4>{index + 1}. {result.question}</h4>
                  <p><strong>Your Answer:</strong> {result.user_answer || 'Not answered'}</p>
                  <p><strong>Correct Answer:</strong> {result.correct_answer}</p>
                  {result.explanation && (
                    <p><strong>Explanation:</strong> {result.explanation}</p>
                  )}
                </div>
              ))}
            </div>

            <div className="reset-container">
              <button onClick={resetApp} className="reset-button">
                Start New Quiz
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;