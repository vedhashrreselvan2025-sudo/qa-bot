import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from rag_engine import build_vector_store, load_vector_store, answer_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store
    try:
        if os.path.exists("faiss_index"):
            logger.info("Loading existing vector store...")
            vector_store = load_vector_store()
        else:
            logger.info("Building new vector store...")
            vector_store = build_vector_store()
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(
    title="Document Q&A Bot",
    description="Answer questions from documents using RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    sources: list
    confidence: str

@app.get("/")
def root():
    return {"message": "Document Q&A Bot is running!"}

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    result = answer_question(request.question, vector_store)
    return result

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store
    allowed_extensions = [".pdf", ".txt", ".md"]
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are allowed")
    try:
        save_path = os.path.join("docs", filename)
        contents = await file.read()
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"Uploaded: {filename}")
        vector_store = build_vector_store()
        return {"message": f"{filename} uploaded and index rebuilt successfully!"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild")
def rebuild_index():
    global vector_store
    try:
        vector_store = build_vector_store()
        return {"message": "Index rebuilt successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))