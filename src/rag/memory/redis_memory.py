"""
Simple Redis Conversational Memory for RAG System
"""

import json
import redis
from typing import List, Dict, Any
from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, CONVERSATION_HISTORY_LIMIT

class ConversationMemory:
    """Simple Redis-based conversation memory to store last 5 conversations."""
    
    def __init__(self):
        self.limit = 5  # Store last 5 conversations
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            self.redis.ping()  # Test connection
        except:
            self.redis = None
    
    def add_conversation(self, session_id: str, user_msg: str, assistant_msg: str):
        """Add a conversation to memory."""
        if not self.redis:
            return
        
        key = f"chat:{session_id}"
        conversation = {"user": user_msg, "assistant": assistant_msg}
        
        try:
            # Add to list and keep only last 5
            self.redis.lpush(key, json.dumps(conversation))
            self.redis.ltrim(key, 0, self.limit - 1)
            self.redis.expire(key, 86400)  # Expire in 24 hours
        except:
            pass
    
    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for session."""
        if not self.redis:
            return []
        
        key = f"chat:{session_id}"
        try:
            conversations = self.redis.lrange(key, 0, -1)
            return [json.loads(conv) for conv in reversed(conversations)]
        except:
            return []
    
    def get_context(self, session_id: str) -> str:
        """Get formatted conversation history as context."""
        history = self.get_history(session_id)
        if not history:
            return ""
        
        context = "Previous conversation:\n"
        for conv in history:
            context += f"User: {conv['user']}\nAssistant: {conv['assistant']}\n\n"
        context += "Current question:\n"
        return context

# Global instance
conversation_memory = ConversationMemory()