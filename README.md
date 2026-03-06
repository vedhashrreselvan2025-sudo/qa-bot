# Document Q&A Bot — RAG Pipeline

AI-powered document question answering with zero hallucination, built for the 30-Hour Hackathon Challenge.

---

## Problem Statement
Organizations have hundreds of documents — policies, contracts, FAQs, manuals — but no intelligent way to query them. This system solves that by letting users ask natural language questions and get precise, cited answers pulled directly from the documents.

---

## How It Works (RAG Pipeline)

1. Ingest — Load all documents from the docs/ folder (PDF, TXT, MD)
2. Chunk — Split documents into 500-character overlapping chunks
3. Embed — Convert each chunk into a vector using HuggingFace all-MiniLM-L6-v2
4. Store — Save all vectors in a FAISS vector database
5. Retrieve — On user question, find top-3 most relevant chunks
6. Generate — Send chunks + question to Groq LLaMA 3.3 70B to generate a clean answer
7. Cite — Return answer + source document + snippet + confidence score

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.14 | Core language |
| FastAPI | REST API server |
| HuggingFace (all-MiniLM-L6-v2) | Text embeddings |
| FAISS | Vector similarity search |
| LangChain | Document loading and chunking |
| Groq (LLaMA 3.3 70B) | Answer generation |
| CUAD Dataset | 500+ real legal contracts |

---

## Project Structure

qa-bot/
├── docs/              <- put your documents here  
├── faiss_index/       <- auto-generated vector index  
├── .env               <- API keys (never share this)  
├── app.py             <- FastAPI server  
├── rag_engine.py      <- RAG engine (core AI logic)  
├── index.html         <- frontend UI  
└── README.md          <- this file  

---

## Setup Instructions

### Step 1 — Install dependencies

pip install fastapi uvicorn pypdf langchain langchain-community faiss-cpu sentence-transformers python-dotenv langchain-huggingface langchain-text-splitters langchain-core groq

---

### Step 2 — Add API keys to .env

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token  
GROQ_API_KEY=your_groq_api_key

---

### Step 3 — Add documents to docs/ folder

Place your .txt, .md, or .pdf files inside the docs/ folder.

---

### Step 4 — Run the server

python -m uvicorn app:app --reload

---

### Step 5 — Open frontend

Open index.html in your browser  
OR go to  
http://127.0.0.1:8000/docs for the API interface.

---

## API Reference

### GET /

Health check — confirms server is running.

Response:

{
  "message": "Document Q&A Bot is running!"
}

---

### POST /ask

Ask a question and get a grounded answer with citations.

Request:

{
  "question": "What is the governing law?"
}

Response:

{
  "answer": "The governing law is the Laws of the State of Delaware as per Section 11.03.",
  "sources": [
    {
      "document": "cuad_small.json",
      "snippet": "Section 11.03. Governing Law; Jurisdiction...",
      "score": 0.9345
    }
  ],
  "confidence": "medium"
}

---

### POST /rebuild

Rebuild the FAISS index when new documents are added to docs/.

---

## Sample Questions & Answers

| Question | Expected Answer |
|----------|----------------|
| What is the governing law? | Laws of the State of Delaware |
| What are the termination clauses? | Either party may terminate with 30 days notice |
| What are the confidentiality obligations? | The receiving party shall not disclose |
| What is the recipe for chocolate cake? | I could not find this in the provided documents |

---

## Evaluation Criteria Coverage

| Criterion | Weight | Status |
|-----------|--------|--------|
| Correctness & Functionality | 40% | Answers grounded in documents only |
| AI/ML Implementation Quality | 30% | HuggingFace + FAISS + Groq LLaMA 3.3 |
| API Design & Engineering | 20% | FastAPI with full validation & error handling |
| Documentation & README | 10% | This file |

---

## What This System Does NOT Do

- It never fabricates answers
- It never uses information outside the provided documents
- If answer is not found, it responds:
"I could not find this in the provided documents. Can you share the relevant document?"

---

## Team

AI & Programming Hackathon — 30 Hour Challenge

Task 1: Document Q&A Bot using RAG