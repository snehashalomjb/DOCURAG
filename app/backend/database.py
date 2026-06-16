import sqlite3
import json
import os
import logging
from typing import List, Dict, Any, Optional
from app.backend import config

logger = logging.getLogger(__name__)

def get_db_connection() -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if it doesn't already exist and runs migrations."""
    db_dir = os.path.dirname(config.DATABASE_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
        
    logger.info(f"Initializing SQLite database at {config.DATABASE_PATH}...")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    document_name TEXT,
                    summary TEXT,
                    analytics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Run table migrations to add columns if they don't exist
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [row[1] for row in cursor.fetchall()]
            if "summary" not in columns:
                cursor.execute("ALTER TABLE sessions ADD COLUMN summary TEXT")
                logger.info("Database migration: Added 'summary' column to sessions table.")
            if "document_name" not in columns:
                cursor.execute("ALTER TABLE sessions ADD COLUMN document_name TEXT")
                logger.info("Database migration: Added 'document_name' column to sessions table.")
            if "analytics" not in columns:
                cursor.execute("ALTER TABLE sessions ADD COLUMN analytics TEXT")
                logger.info("Database migration: Added 'analytics' column to sessions table.")
            
            # Create messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,  -- JSON list of retrieved chunks
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
        raise RuntimeError(f"Database initialization failed: {e}")

def create_session(
    session_id: str, 
    document_name: Optional[str] = None, 
    summary: Optional[str] = None,
    analytics: Optional[str] = None
):
    """Inserts a new session record or updates the active document, summary, and analytics for an existing one."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # UPSERT syntax preserving existing values on conflict if inputs are None
        cursor.execute("""
            INSERT INTO sessions (session_id, document_name, summary, analytics) 
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET 
                document_name = CASE WHEN excluded.document_name IS NOT NULL THEN excluded.document_name ELSE sessions.document_name END,
                summary = CASE WHEN excluded.summary IS NOT NULL THEN excluded.summary ELSE sessions.summary END,
                analytics = CASE WHEN excluded.analytics IS NOT NULL THEN excluded.analytics ELSE sessions.analytics END
        """, (session_id, document_name, summary, analytics))
        conn.commit()

def get_session_document(session_id: str) -> Optional[str]:
    """Retrieves the document names associated with a session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT document_name FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["document_name"] if row else None

def get_session_summary(session_id: str) -> Optional[str]:
    """Retrieves the summary associated with a session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT summary FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["summary"] if row else None

def get_session_analytics(session_id: str) -> Optional[str]:
    """Retrieves the analytics JSON string associated with a session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        row = cursor.execute("SELECT analytics FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row["analytics"] if row else None

def add_message(session_id: str, role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None):
    """Persists a chat message associated with a session to the database."""
    sources_json = json.dumps(sources) if sources is not None else None
    
    # First, make sure session exists
    create_session(session_id)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (session_id, role, content, sources) 
            VALUES (?, ?, ?, ?)
        """, (session_id, role, content, sources_json))
        conn.commit()

def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves all chat messages for a specific session sorted by creation order."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT role, content, sources, created_at 
            FROM messages 
            WHERE session_id = ? 
            ORDER BY id ASC
        """, (session_id,)).fetchall()
        
        messages = []
        for row in rows:
            sources_list = json.loads(row["sources"]) if row["sources"] else []
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "sources": sources_list,
                "created_at": row["created_at"]
            })
        return messages

def get_all_sessions() -> List[Dict[str, Any]]:
    """Returns a list of all active sessions in the database sorted by creation date."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT session_id, document_name, summary, analytics, created_at 
            FROM sessions 
            ORDER BY created_at DESC
        """).fetchall()
        return [dict(row) for row in rows]

def delete_session(session_id: str):
    """Deletes a session and all its associated chat history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
