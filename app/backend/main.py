import os
import shutil
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware

from app.backend import config, database
from app.backend.rag import RAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DOCURAG Backend API",
    description="FastAPI backend for AI Document Assistant using local embeddings and Gemini API",
    version="1.1.0"
)

# Enable CORS for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG Pipeline instance
rag_pipeline = RAGPipeline()

# On startup, try to load existing default index and initialize database
@app.on_event("startup")
def startup_event():
    try:
        # Initialize SQLite database
        database.init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing system on startup: {e}")

class ChatMessage(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None

class SourceDetail(BaseModel):
    chunk: str
    source_doc: str
    confidence: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the document assistant")
    session_id: str = Field(..., description="The session ID to store this conversation under")
    history: List[ChatMessage] = Field(default=[], description="Conversation history context")
    top_k: Optional[int] = Field(None, description="Number of source chunks to retrieve")
    gemini_api_key: Optional[str] = Field(None, description="Optional Gemini API key override")

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDetail]

class ConfigRequest(BaseModel):
    gemini_api_key: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/status")
def get_status(session_id: Optional[str] = Query(None)):
    """Retrieve status of the vector index and document for a specific session."""
    has_api_key = bool(config.GEMINI_API_KEY)
    
    if session_id:
        # Load the session index to get accurate count
        rag_pipeline.load_index(session_id)
        
    has_index = rag_pipeline.index is not None
    chunks_count = len(rag_pipeline.chunks)
    documents = rag_pipeline.metadata.get("documents", [])
    
    return {
        "document_loaded": has_index,
        "chunks_count": chunks_count,
        "indexed_documents": documents,
        "gemini_api_key_configured": has_api_key,
        "embedding_model": config.EMBEDDING_MODEL_NAME
    }

@app.post("/configure")
def configure_api_keys(req: ConfigRequest):
    """Allows dynamic configuration of Gemini API Key."""
    if not req.gemini_api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    config.GEMINI_API_KEY = req.gemini_api_key.strip()
    return {"status": "configured", "message": "Gemini API key updated successfully."}

@app.post("/upload")
async def upload_file(
    session_id: str = Query(..., description="Session ID to associate document with"),
    file: UploadFile = File(...),
    chunk_size: int = Query(None, description="Chunk size in characters"),
    chunk_overlap: int = Query(None, description="Chunk overlap in characters")
):
    """Uploads a document, appends chunks to session FAISS index, and generates auto-summary."""
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())
    allowed_extensions = ['.pdf', '.txt', '.docx', '.html', '.htm', '.md']
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(allowed_extensions)}"
        )

    # Resolve settings
    c_size = chunk_size if chunk_size is not None else config.DEFAULT_CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else config.DEFAULT_CHUNK_OVERLAP

    if c_size <= 0:
        raise HTTPException(status_code=400, detail="Chunk size must be greater than zero.")
    if c_overlap >= c_size:
        raise HTTPException(status_code=400, detail="Chunk overlap must be smaller than chunk size.")

    # Save file to upload directory
    file_path = os.path.join(config.UPLOAD_DIR, filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process document
    try:
        # Load and parse text
        text = rag_pipeline.load_document(file_path)
        if not text.strip():
            raise ValueError("No text could be extracted from the document.")

        # Chunk text
        chunks = rag_pipeline.chunk_text(text, chunk_size=c_size, chunk_overlap=c_overlap)
        if not chunks:
            raise ValueError("Document was parsed, but chunking resulted in zero chunks.")

        # 1. Load existing session index (if any) to append to it
        rag_pipeline.load_index(session_id)

        # 2. Append chunks and embeddings
        rag_pipeline.append_to_index(chunks, filename)

        # 3. Save updated session index
        rag_pipeline.save_index(session_id)

        # 4. Auto-generate document 3-line summary via Gemini
        summary = rag_pipeline.generate_summary(text)

        # 5. Auto-generate document smart analytics (word/page counts, topics, suggested questions)
        analytics = rag_pipeline.generate_analytics(text, file_path)

        # 6. Fetch existing session docs list and update database
        existing_docs = database.get_session_document(session_id)
        if existing_docs:
            docs_list = [d.strip() for d in existing_docs.split(",") if d.strip()]
            if filename not in docs_list:
                docs_list.append(filename)
            new_docs_str = ", ".join(docs_list)
        else:
            new_docs_str = filename

        import json
        analytics_json = json.dumps(analytics)
        database.create_session(session_id, document_name=new_docs_str, summary=summary, analytics=analytics_json)

        # Calculate session statistics
        total_chunks = len(rag_pipeline.chunks)
        total_words = sum(len(c.split()) for c in rag_pipeline.chunks)
        estimated_tokens = int(sum(len(c) for c in rag_pipeline.chunks) / 4)

        return {
            "status": "success",
            "message": "Document uploaded and indexed successfully.",
            "filename": filename,
            "chunks_count": len(chunks),
            "total_chunks": total_chunks,
            "total_words": total_words,
            "estimated_tokens": estimated_tokens,
            "summary": summary,
            "analytics": analytics
        }
    except ValueError as ve:
        logger.warning(f"Validation error during processing: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during file processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        file.file.close()

@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """Queries the indexed document using FAISS similarity search and Gemini API."""
    # Setup API Key override if provided
    original_key = config.GEMINI_API_KEY
    if request.gemini_api_key and request.gemini_api_key.strip():
        config.GEMINI_API_KEY = request.gemini_api_key.strip()

    try:
        # Load the session-specific index from disk
        loaded = rag_pipeline.load_index(request.session_id)
        if not loaded or rag_pipeline.index is None:
            raise HTTPException(
                status_code=400, 
                detail="No documents have been indexed for this session yet."
            )

        # Resolve top_k
        top_k = request.top_k if request.top_k is not None else config.DEFAULT_TOP_K
        
        # Convert history to dictionaries
        history_dict = [{"role": msg.role, "content": msg.content} for msg in request.history]

        # 1. Condense the query for vector retrieval using history
        condensed_query = rag_pipeline.condense_query(request.query, history_dict)

        # 2. Retrieve top-k chunks using the condensed query
        retrieved = rag_pipeline.search(condensed_query, top_k=top_k)
        
        # 3. Call Gemini with history and context chunks
        answer, sources = rag_pipeline.generate_answer(request.query, history_dict, retrieved)
        
        # 4. Save interaction to SQLite database
        try:
            # Format sources for DB persistence
            db_sources = [
                {"chunk": item["chunk"], "source_doc": item["source_doc"], "confidence": item["confidence"]}
                for item in retrieved
            ]
            database.add_message(request.session_id, "user", request.query)
            database.add_message(request.session_id, "assistant", answer, db_sources)
        except Exception as dbe:
            logger.error(f"Failed to save messages to SQLite: {dbe}")

        # Format output
        response_sources = [
            SourceDetail(chunk=item["chunk"], source_doc=item["source_doc"], confidence=item["confidence"])
            for item in retrieved
        ]
        return QueryResponse(answer=answer, sources=response_sources)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error querying pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
    finally:
        # Restore original configuration
        config.GEMINI_API_KEY = original_key

@app.get("/sessions")
def get_all_sessions():
    """Returns a list of all active chat sessions stored in the database."""
    try:
        return database.get_all_sessions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sessions: {str(e)}")

@app.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Returns all chat messages for a specific session."""
    try:
        return database.get_session_messages(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch session messages: {str(e)}")

@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Deletes a session, its message logs, and its index files on disk."""
    try:
        # Delete database logs
        database.delete_session(session_id)
        
        # Delete FAISS vector store files if they exist
        index_file = os.path.join(config.INDEX_DIR, f"{session_id}.faiss")
        meta_file = os.path.join(config.INDEX_DIR, f"{session_id}.pkl")
        if os.path.exists(index_file):
            os.remove(index_file)
        if os.path.exists(meta_file):
            os.remove(meta_file)
            
        return {"status": "success", "message": f"Session {session_id} deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

@app.get("/sessions/{session_id}/summary")
def get_session_summary(session_id: str):
    """Gets the generated summary for a session."""
    try:
        summary = database.get_session_summary(session_id)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch summary: {str(e)}")


@app.get("/sessions/{session_id}/analytics")
def get_session_analytics(session_id: str):
    """Gets the generated analytics (statistics, topics, suggested questions) for a session."""
    try:
        import json
        analytics_str = database.get_session_analytics(session_id)
        if analytics_str:
            return json.loads(analytics_str)
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")
