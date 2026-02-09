"""Server-side conversation session management."""
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationSession:
    """A single conversation session with message history."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict[str, str]] = []  # [{"role": "user"|"assistant", "content": "..."}]
        self.scope: Optional[Dict] = None  # Current book/series scope
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append({"role": role, "content": content})
        self.last_activity = datetime.now()

    def get_recent_history(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """Get the most recent turns of conversation.

        A "turn" is a user message + assistant response pair.

        Args:
            max_turns: Maximum number of turns to return

        Returns:
            List of recent messages (up to max_turns * 2 messages)
        """
        # Each turn is 2 messages (user + assistant)
        max_messages = max_turns * 2
        return self.history[-max_messages:]

    def format_history_for_prompt(self, max_turns: int = 5) -> str:
        """Format recent conversation history as a string for inclusion in prompts.

        Args:
            max_turns: Maximum number of turns to include

        Returns:
            Formatted conversation history string, or empty string if no history
        """
        recent = self.get_recent_history(max_turns)
        if not recent:
            return ""

        parts = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")

        return "\n".join(parts)

    def clear(self):
        """Clear the conversation history and scope."""
        self.history = []
        self.scope = None
        self.last_activity = datetime.now()


class SessionManager:
    """Manages multiple conversation sessions."""

    def __init__(self, max_sessions: int = 100, max_history_per_session: int = 50):
        """
        Args:
            max_sessions: Maximum number of concurrent sessions before eviction
            max_history_per_session: Maximum messages to keep per session
        """
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions
        self.max_history = max_history_per_session

    def create_session(self) -> ConversationSession:
        """Create a new conversation session.

        Returns:
            The new session
        """
        # Evict oldest sessions if at capacity
        if len(self.sessions) >= self.max_sessions:
            self._evict_oldest()

        session_id = str(uuid.uuid4())
        session = ConversationSession(session_id)
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get an existing session by ID."""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Get an existing session or create a new one.

        Args:
            session_id: Optional session ID to look up

        Returns:
            The session (existing or newly created)
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to a session.

        Args:
            session_id: Session to add to
            role: "user" or "assistant"
            content: Message content

        Returns:
            True if message was added, False if session not found
        """
        session = self.get_session(session_id)
        if session is None:
            return False

        session.add_message(role, content)

        # Trim if history is too long
        if len(session.history) > self.max_history:
            session.history = session.history[-self.max_history:]

        return True

    def clear_session(self, session_id: str) -> bool:
        """Clear a session's history but keep it alive.

        Returns:
            True if session was found and cleared
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        session.clear()
        logger.info(f"Cleared session {session_id}")
        return True

    def delete_session(self, session_id: str):
        """Delete a session entirely."""
        self.sessions.pop(session_id, None)

    def _evict_oldest(self):
        """Remove the oldest session by last_activity."""
        if not self.sessions:
            return
        oldest_id = min(
            self.sessions,
            key=lambda sid: self.sessions[sid].last_activity,
        )
        del self.sessions[oldest_id]
        logger.info(f"Evicted oldest session {oldest_id}")
