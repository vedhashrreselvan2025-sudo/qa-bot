import os
import json
import logging
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCS_FOLDER = "docs"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_documents():
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        filepath = os.path.join(DOCS_FOLDER, filename)
        try:
            if filename.endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    for article in data["data"]:
                        for para in article.get("paragraphs", []):
                            context = para.get("context", "")
                            if context:
                                documents.append(Document(
                                    page_content=context,
                                    metadata={"source": filename, "title": article.get("title", "")}
                                ))
                logger.info(f"Loaded: {filename}")
            elif filename.endswith(".txt") or filename.endswith(".md"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
                logger.info(f"Loaded: {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    return documents

def build_vector_store():
    logger.info("Loading documents...")
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in docs/ folder!")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    logger.info("Vector store built and saved!")
    return vector_store

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def answer_question(question: str, vector_store):
    if not question or not question.strip():
        return {"answer": "Please provide a valid question.", "sources": [], "confidence": "low"}
    try:
        docs = vector_store.similarity_search_with_score(question, k=3)
        if not docs:
            return {"answer": "I could not find this in the provided documents. Can you share the relevant document?", "sources": [], "confidence": "low"}
        context = "\n\n".join([doc.page_content for doc, _ in docs])
        sources = []
        for doc, score in docs:
            sources.append({
                "document": doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200],
                "score": round(float(score), 4)
            })
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say exactly: "I could not find this in the provided documents. Can you share the relevant document?"
Do not make up any information.

Context:
{context}

Question: {question}

Answer in 2-3 sentences:"""
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        answer = response.choices[0].message.content.strip()
        top_score = docs[0][1]
        if top_score < 0.5:
            confidence = "high"
        elif top_score < 1.0:
            confidence = "medium"
        else:
            confidence = "low"
        return {"answer": answer, "sources": sources, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {"answer": "I could not find this in the provided documents. Can you share the relevant document?", "sources": [], "confidence": "low"}
        