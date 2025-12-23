from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from rag_system import RAGSystem

app = FastAPI(title="RAG PDF Q&A System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system with your API key
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')
rag_system = RAGSystem(OPENAI_API_KEY)

# Temporary upload directory
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG PDF Q&A System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .spinner {
            animation: spin 1s linear infinite;
            border: 3px solid #f3f4f6;
            border-top: 3px solid #8b5cf6;
            border-radius: 50%;
            width: 40px;
            height: 40px;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
    </style>
</head>
<body class="bg-gradient-to-br from-indigo-100 via-purple-50 to-pink-100 min-h-screen">
    <div class="container mx-auto px-4 py-6 max-w-6xl">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-extrabold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
                📚 RAG PDF Q&A System
            </h1>
            <p class="text-base text-gray-600 max-w-2xl mx-auto">
                Upload any PDF document and get instant, intelligent answers powered by AI
            </p>
        </div>

        <!-- Main Content Grid -->
        <div class="grid lg:grid-cols-3 gap-6">
            <!-- Left Column - Upload & Question -->
            <div class="lg:col-span-2">
                <!-- Upload Card -->
                <div class="bg-white rounded-3xl shadow-xl p-6 mb-6 border border-gray-100">
                    <div class="flex items-center mb-3">
                        <span class="text-2xl mr-2">📄</span>
                        <h2 class="text-xl font-bold text-gray-800">Upload Document</h2>
                    </div>
                    
                    <div class="border-2 border-dashed border-gray-300 rounded-2xl p-6 text-center hover:border-blue-400 hover:bg-blue-50 transition-all duration-300 cursor-pointer" id="dropZone">
                        <input type="file" id="fileInput" accept=".pdf" class="hidden">
                        <label for="fileInput" class="cursor-pointer block">
                            <div id="uploadIcon" class="mb-3">
                                <span class="text-5xl">📤</span>
                            </div>
                            <p class="text-base text-gray-700 font-semibold mb-1" id="uploadText">
                                Drop your PDF here or click to browse
                            </p>
                            <p class="text-sm text-gray-500">
                                Supports PDF files up to 16MB
                            </p>
                        </label>
                    </div>
                    <div id="uploadStatus" class="mt-3 text-sm font-medium"></div>
                </div>

                <!-- Question Card -->
                <div id="questionSection" class="bg-white rounded-3xl shadow-xl p-6 hidden border border-gray-100">
                    <div class="flex items-center mb-3">
                        <span class="text-2xl mr-2">💬</span>
                        <h2 class="text-xl font-bold text-gray-800">Ask a Question</h2>
                    </div>
                    
                    <div class="space-y-3">
                        <textarea 
                            id="questionInput" 
                            rows="3"
                            placeholder="e.g., What are the main conclusions of this document?"
                            class="w-full px-4 py-3 border-2 border-gray-200 rounded-2xl focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none text-base resize-none"
                        ></textarea>
                        <button 
                            id="askButton"
                            class="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white text-base font-semibold rounded-2xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
                        >
                            <span id="askButtonText" class="flex items-center justify-center">
                                <span class="mr-2">⚡</span>
                                Get Answer
                            </span>
                        </button>
                    </div>
                </div>

                <!-- Answer Card -->
                <div id="answerDiv" class="bg-gradient-to-br from-blue-50 to-purple-50 rounded-3xl shadow-xl p-6 mt-6 hidden border border-blue-200">
                    <div class="flex items-start mb-3">
                        <span class="text-2xl mr-2">💡</span>
                        <div class="flex-1">
                            <h3 class="font-bold text-gray-800 text-lg mb-2">Answer</h3>
                            <div class="bg-white rounded-xl p-4 shadow-sm">
                                <p class="text-gray-700 leading-relaxed text-base" id="answerText"></p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Loading Spinner -->
                <div id="loadingSpinner" class="hidden text-center py-8">
                    <div class="spinner mx-auto"></div>
                    <p class="text-gray-700 mt-3 text-base font-medium" id="loadingText">Processing...</p>
                </div>

                <!-- Error Display -->
                <div id="errorDiv" class="bg-red-50 rounded-2xl p-4 mt-6 hidden border-l-4 border-red-500">
                    <div class="flex items-start">
                        <span class="text-xl mr-2">❌</span>
                        <p class="text-red-700 text-sm" id="errorText"></p>
                    </div>
                </div>

                <!-- Reset Button -->
                <button 
                    id="resetButton"
                    class="w-full mt-6 px-6 py-3 border-2 border-gray-300 text-gray-700 font-semibold rounded-2xl hover:bg-gray-50 hover:border-gray-400 transition-all duration-300 hidden"
                >
                    <span class="flex items-center justify-center">
                        <span class="mr-2">🔄</span>
                        Reset & Upload New PDF
                    </span>
                </button>
            </div>

            <!-- Right Column - Instructions -->
            <div class="lg:col-span-1">
                <div class="bg-white rounded-3xl shadow-xl p-5 sticky top-6 border border-gray-100">
                    <div class="flex items-center mb-4">
                        <span class="text-2xl mr-2">ℹ️</span>
                        <h3 class="font-bold text-gray-800 text-lg">How It Works</h3>
                    </div>
                    
                    <div class="space-y-3">
                        <div class="flex items-start p-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
                            <div class="flex-shrink-0 w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold mr-2 text-sm shadow">1</div>
                            <div>
                                <p class="text-gray-800 font-semibold text-sm mb-0.5">Upload PDF</p>
                                <p class="text-gray-600 text-xs">Choose your document (max 16MB)</p>
                            </div>
                        </div>
                        
                        <div class="flex items-start p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl">
                            <div class="flex-shrink-0 w-6 h-6 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold mr-2 text-sm shadow">2</div>
                            <div>
                                <p class="text-gray-800 font-semibold text-sm mb-0.5">AI Processing</p>
                                <p class="text-gray-600 text-xs">Document is analyzed and indexed</p>
                            </div>
                        </div>
                        
                        <div class="flex items-start p-3 bg-gradient-to-r from-green-50 to-teal-50 rounded-xl">
                            <div class="flex-shrink-0 w-6 h-6 bg-green-600 text-white rounded-full flex items-center justify-center font-bold mr-2 text-sm shadow">3</div>
                            <div>
                                <p class="text-gray-800 font-semibold text-sm mb-0.5">Ask Questions</p>
                                <p class="text-gray-600 text-xs">Get intelligent, context-aware answers</p>
                            </div>
                        </div>
                        
                        <div class="flex items-start p-3 bg-gradient-to-r from-orange-50 to-red-50 rounded-xl">
                            <div class="flex-shrink-0 w-6 h-6 bg-orange-600 text-white rounded-full flex items-center justify-center font-bold mr-2 text-sm shadow">4</div>
                            <div>
                                <p class="text-gray-800 font-semibold text-sm mb-0.5">Instant Results</p>
                                <p class="text-gray-600 text-xs">Receive accurate answers in seconds</p>
                            </div>
                        </div>
                    </div>

                    <div class="mt-4 p-3 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl border border-yellow-200">
                        <div class="flex items-start">
                            <span class="text-lg mr-2">💡</span>
                            <div>
                                <p class="text-yellow-800 font-semibold text-xs mb-1">Pro Tips</p>
                                <ul class="text-yellow-700 text-xs space-y-0.5">
                                    <li>• Be specific with your questions</li>
                                    <li>• Ask one question at a time</li>
                                    <li>• Refer to specific sections if needed</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadText = document.getElementById('uploadText');
        const uploadStatus = document.getElementById('uploadStatus');
        const uploadIcon = document.getElementById('uploadIcon');
        const questionSection = document.getElementById('questionSection');
        const questionInput = document.getElementById('questionInput');
        const askButton = document.getElementById('askButton');
        const askButtonText = document.getElementById('askButtonText');
        const errorDiv = document.getElementById('errorDiv');
        const errorText = document.getElementById('errorText');
        const answerDiv = document.getElementById('answerDiv');
        const answerText = document.getElementById('answerText');
        const resetButton = document.getElementById('resetButton');
        const loadingSpinner = document.getElementById('loadingSpinner');
        const loadingText = document.getElementById('loadingText');

        let documentUploaded = false;

        function showError(message) {
            errorText.textContent = message;
            errorDiv.classList.remove('hidden');
            answerDiv.classList.add('hidden');
        }

        function hideError() {
            errorDiv.classList.add('hidden');
        }

        function showLoading(message = 'Processing...') {
            loadingText.textContent = message;
            loadingSpinner.classList.remove('hidden');
            askButton.disabled = true;
            askButtonText.innerHTML = '<span class="mr-2">⏳</span>Processing...';
        }

        function hideLoading() {
            loadingSpinner.classList.add('hidden');
            askButton.disabled = false;
            askButtonText.innerHTML = '<span class="mr-2">⚡</span>Get Answer';
        }

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            hideError();
            answerDiv.classList.add('hidden');
            showLoading('Processing PDF... This may take a moment.');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    uploadIcon.innerHTML = '<span class="text-5xl">✅</span>';
                    uploadText.innerHTML = `<span class="text-green-600">✓ ${file.name}</span>`;
                    uploadStatus.innerHTML = `<span class="text-green-600">✓ PDF processed successfully! Created ${data.num_chunks} text chunks for analysis.</span>`;
                    questionSection.classList.remove('hidden');
                    resetButton.classList.remove('hidden');
                    documentUploaded = true;
                } else {
                    showError(data.detail || 'Failed to upload file');
                }
            } catch (error) {
                showError('Error uploading file: ' + error.message);
            } finally {
                hideLoading();
            }
        });

        askButton.addEventListener('click', async () => {
            const question = questionInput.value.trim();
            if (!question) {
                showError('Please enter a question');
                return;
            }

            hideError();
            answerDiv.classList.add('hidden');
            showLoading('Finding relevant information and generating answer...');

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });

                const data = await response.json();

                if (response.ok) {
                    answerText.textContent = data.answer;
                    answerDiv.classList.remove('hidden');
                    answerDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    showError(data.detail || 'Failed to get answer');
                }
            } catch (error) {
                showError('Error getting answer: ' + error.message);
                console.error('Full error:', error);
            } finally {
                hideLoading();
            }
        });

        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && !askButton.disabled) {
                e.preventDefault();
                askButton.click();
            }
        });

        resetButton.addEventListener('click', async () => {
            try {
                await fetch('/reset', { method: 'POST' });
                
                fileInput.value = '';
                questionInput.value = '';
                uploadIcon.innerHTML = '<span class="text-5xl">📤</span>';
                uploadText.textContent = 'Drop your PDF here or click to browse';
                uploadStatus.innerHTML = '';
                questionSection.classList.add('hidden');
                resetButton.classList.add('hidden');
                answerDiv.classList.add('hidden');
                hideError();
                documentUploaded = false;
            } catch (error) {
                showError('Error resetting: ' + error.message);
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload and processing"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        num_chunks = rag_system.process_pdf(file_path)
        os.remove(file_path)
        
        return {
            "success": True,
            "message": f"PDF processed successfully. Created {num_chunks} chunks.",
            "num_chunks": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Handle question and return answer"""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="No question provided")
    
    if not rag_system.has_document():
        raise HTTPException(status_code=400, detail="Please upload a PDF first")
    
    try:
        answer = rag_system.answer_question(request.question)
        return {
            "success": True,
            "answer": answer
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/reset")
async def reset():
    """Reset the system"""
    rag_system.reset()
    return {"success": True, "message": "System reset successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "has_document": rag_system.has_document()}