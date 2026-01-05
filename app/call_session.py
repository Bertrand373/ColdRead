"""
Coachd Call Session Manager
Tracks active Twilio call sessions and manages real-time state
"""

import asyncio
import uuid
import json
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CallStatus(Enum):
    INITIATING = "initiating"
    AGENT_RINGING = "agent_ringing"
    AGENT_CONNECTED = "agent_connected"
    CLIENT_RINGING = "client_ringing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CallSession:
    """Represents an active call session"""
    session_id: str
    agent_phone: str
    client_phone: Optional[str] = None
    agent_call_sid: Optional[str] = None
    client_call_sid: Optional[str] = None
    conference_sid: Optional[str] = None
    recording_sid: Optional[str] = None
    recording_url: Optional[str] = None
    status: CallStatus = CallStatus.INITIATING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    transcript: list = field(default_factory=list)
    guidance_history: list = field(default_factory=list)
    websocket_clients: set = field(default_factory=set)
    client_context: Optional[dict] = None  # Quick Prep context from agent
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_phone": self.agent_phone,
            "client_phone": self.client_phone,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.get_duration(),
            "transcript_length": len(self.transcript),
            "recording_url": self.recording_url
        }
    
    def get_duration(self) -> Optional[int]:
        if self.started_at:
            end = self.ended_at or datetime.utcnow()
            return int((end - self.started_at).total_seconds())
        return None


class CallSessionManager:
    """Manages all active call sessions"""
    
    def __init__(self):
        self._sessions: Dict[str, CallSession] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, agent_phone: str, client_context: dict = None, client_phone: str = None) -> CallSession:
        """Create a new call session"""
        session_id = str(uuid.uuid4())[:8]
        
        # Normalize client_phone if provided
        if client_phone:
            if not client_phone.startswith("+"):
                digits = client_phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
                client_phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
        
        session = CallSession(
            session_id=session_id,
            agent_phone=agent_phone,
            client_phone=client_phone,
            client_context=client_context
        )
        
        async with self._lock:
            self._sessions[session_id] = session
        
        logger.info(f"Created session {session_id} for agent {agent_phone}, client: {client_phone}" + 
                   (f" with context: {list(client_context.keys())}" if client_context else ""))
        return session
    
    async def get_session(self, session_id: str) -> Optional[CallSession]:
        """Get a session by ID"""
        return self._sessions.get(session_id)
    
    async def update_session(self, session_id: str, **kwargs) -> Optional[CallSession]:
        """Update session properties"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        async with self._lock:
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
        
        # Broadcast status update to connected clients
        await self._broadcast_to_session(session_id, {
            "type": "session_update",
            "data": session.to_dict()
        })
        
        logger.info(f"Updated session {session_id}: {list(kwargs.keys())}")
        return session
    
    async def add_transcript(self, session_id: str, text: str, speaker: str = "unknown", is_final: bool = False):
        """Add transcript entry to session"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        entry = {
            "text": text,
            "speaker": speaker,
            "timestamp": datetime.utcnow().isoformat(),
            "is_final": is_final
        }
        
        async with self._lock:
            session.transcript.append(entry)
        
        await self._broadcast_to_session(session_id, {
            "type": "transcript",
            "data": entry
        })
    
    async def add_guidance(self, session_id: str, guidance: dict):
        """Add AI guidance to session history"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        entry = {
            **guidance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with self._lock:
            session.guidance_history.append(entry)
        
        await self._broadcast_to_session(session_id, {
            "type": "guidance",
            "data": entry
        })
    
    async def register_websocket(self, session_id: str, websocket: Any):
        """Register a WebSocket client for a session"""
        session = self._sessions.get(session_id)
        if session:
            async with self._lock:
                session.websocket_clients.add(websocket)
            logger.info(f"WebSocket registered for session {session_id}")
    
    async def unregister_websocket(self, session_id: str, websocket: Any):
        """Unregister a WebSocket client"""
        session = self._sessions.get(session_id)
        if session:
            async with self._lock:
                session.websocket_clients.discard(websocket)
            logger.info(f"WebSocket unregistered for session {session_id}")
    
    async def _broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast a message to all WebSocket clients for a session"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        message_str = json.dumps(message)
        
        dead_clients = set()
        for ws in session.websocket_clients:
            try:
                await ws.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                dead_clients.add(ws)
        
        if dead_clients:
            async with self._lock:
                session.websocket_clients -= dead_clients
    
    async def end_session(self, session_id: str):
        """Mark a session as completed"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        async with self._lock:
            session.status = CallStatus.COMPLETED
            session.ended_at = datetime.utcnow()
        
        await self._broadcast_to_session(session_id, {
            "type": "call_ended",
            "data": session.to_dict()
        })
        
        logger.info(f"Session {session_id} ended. Duration: {session.get_duration()}s")
    
    async def get_active_sessions(self) -> list:
        """Get all active sessions"""
        return [
            s.to_dict() for s in self._sessions.values() 
            if s.status not in [CallStatus.COMPLETED, CallStatus.FAILED]
        ]


# Global session manager instance
session_manager = CallSessionManager()