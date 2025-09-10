import sqlite3
import json
from typing import List, Dict, Optional
from datetime import datetime
import uuid

class ConversationDB:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.init_db() 
    
    def init_db(self):
        """Initialize the database with a simple messages table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # create table for message data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')

        # create table for session data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # for each session, create new session id, insert into db
        cursor.execute(
        "INSERT INTO sessions (id) VALUES (?)",
        (session_id,)
        )   

        conn.commit()
        conn.close()
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the session"""
        message_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO messages (id, session_id, role, content) VALUES (?, ?, ?, ?)",
            (message_id, session_id, role, content)
        )

        # Update session last_activity
        cursor.execute(
            "UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = ?",
            (session_id,)
        )

        conn.commit()
        conn.close()

    # get all messages for a specific session
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )

        messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
        conn.close()

        return messages
    
    # check if a session exists
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?",
            (session_id,)
        )

        exists = cursor.fetchone()[0] > 0
        conn.close()

        return exists

# single instance of db
_db: Optional[ConversationDB] = None

def get_db() -> ConversationDB:
    global _db
    if _db is None:
        _db = ConversationDB()
    return _db