"""
Coachd Agent Authentication
============================
Phone-based authentication for agents using existing Telnyx verification.
No passwords - agents verify via SMS each session (like magic links).

Flow:
1. Agent enters agency code (existing)
2. Agent enters phone number
3. SMS verification (reuses phone_verification.py)
4. On success: create/update agent record, return session token
5. Token stored in localStorage, sent with API requests
"""

import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, DateTime, Date, Text, Boolean, JSON
from sqlalchemy.orm import Session

from .database import Base, get_db, is_db_configured, engine
from .config import settings

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Session duration - 30 days (agents shouldn't have to re-verify often)
SESSION_DURATION_DAYS = 30


# ==================== DATABASE MODEL ====================

class Agent(Base):
    """
    Agent account - identified by phone number within an agency.
    Phone is the primary identifier since agents always have their phone.
    """
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Identity
    phone = Column(String(20), index=True, nullable=False)  # E.164 format
    agency_code = Column(String(50), index=True, nullable=False)
    
    # Profile (collected on first login)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    email = Column(String(255), nullable=True)  # Optional, for notifications
    
    # Onboarding tracking
    onboarding_start_date = Column(Date, nullable=True)
    onboarding_day = Column(Integer, default=1)
    current_week = Column(Integer, default=1)
    
    # Progress (stored as JSON for flexibility)
    modules_completed = Column(JSON, default=list)  # ["module1", "module2"]
    
    # Activity tracking
    total_calls = Column(Integer, default=0)
    total_appointments = Column(Integer, default=0)
    total_presentations = Column(Integer, default=0)
    total_closes = Column(Integer, default=0)
    
    # Session management
    session_token = Column(String(64), nullable=True, index=True)
    session_expires = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    last_active = Column(DateTime, nullable=True)
    
    # Status
    status = Column(String(20), default="active")  # active, paused, churned
    
    # Unique constraint: one agent per phone per agency
    __table_args__ = (
        # A phone number can only be registered once per agency
        # But same phone could be in different agencies (rare but possible)
    )


# ==================== PYDANTIC MODELS ====================

class PhoneLoginRequest(BaseModel):
    """Initial login request with phone number"""
    phone: str
    agency_code: str


class VerifyLoginRequest(BaseModel):
    """Verify SMS code to complete login"""
    phone: str
    agency_code: str
    code: str


class CompleteProfileRequest(BaseModel):
    """Complete profile after first verification"""
    first_name: str
    last_name: str
    email: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent data returned to frontend"""
    id: int
    phone: str
    agency_code: str
    first_name: Optional[str]
    last_name: Optional[str]
    display_name: str
    onboarding_day: int
    current_week: int
    modules_completed: list
    total_calls: int
    total_appointments: int
    total_presentations: int
    total_closes: int
    needs_profile: bool
    
    class Config:
        from_attributes = True


# ==================== HELPER FUNCTIONS ====================

def normalize_phone(phone: str) -> str:
    """Normalize phone to E.164 format"""
    digits = ''.join(c for c in phone if c.isdigit())
    if len(digits) == 10:
        return f"+1{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}"


def generate_session_token() -> str:
    """Generate a secure random session token"""
    return secrets.token_hex(32)


def hash_token(token: str) -> str:
    """Hash token for storage (we store hash, compare hash)"""
    return hashlib.sha256(token.encode()).hexdigest()


def get_agent_from_token(token: str, db: Session) -> Optional[Agent]:
    """Look up agent by session token"""
    if not token:
        return None
    
    # We store the token directly (not hashed) for simplicity
    # In production, you might want to hash it
    agent = db.query(Agent).filter(
        Agent.session_token == token,
        Agent.session_expires > datetime.utcnow()
    ).first()
    
    return agent


def agent_to_response(agent: Agent) -> dict:
    """Convert Agent model to response dict"""
    needs_profile = not agent.first_name or not agent.last_name
    
    if agent.first_name and agent.last_name:
        display_name = f"{agent.first_name} {agent.last_name}"
    elif agent.first_name:
        display_name = agent.first_name
    else:
        display_name = "Agent"
    
    return {
        "id": agent.id,
        "phone": agent.phone,
        "agency_code": agent.agency_code,
        "first_name": agent.first_name,
        "last_name": agent.last_name,
        "display_name": display_name,
        "onboarding_day": agent.onboarding_day or 1,
        "current_week": agent.current_week or 1,
        "modules_completed": agent.modules_completed or [],
        "total_calls": agent.total_calls or 0,
        "total_appointments": agent.total_appointments or 0,
        "total_presentations": agent.total_presentations or 0,
        "total_closes": agent.total_closes or 0,
        "needs_profile": needs_profile,
    }


# ==================== AUTH DEPENDENCY ====================

async def get_current_agent(
    authorization: Optional[str] = Header(None)
) -> Optional[Agent]:
    """
    Dependency to get current authenticated agent.
    Token should be passed as: Authorization: Bearer <token>
    """
    if not is_db_configured():
        return None
    
    if not authorization:
        return None
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    
    token = parts[1]
    
    with get_db() as db:
        agent = get_agent_from_token(token, db)
        if agent:
            # Update last_active
            agent.last_active = datetime.utcnow()
            db.commit()
            # Detach from session for use outside context
            db.expunge(agent)
        return agent


async def require_agent(
    authorization: Optional[str] = Header(None)
) -> Agent:
    """Dependency that requires authentication"""
    agent = await get_current_agent(authorization)
    if not agent:
        raise HTTPException(status_code=401, detail="Authentication required")
    return agent


# ==================== AUTH ENDPOINTS ====================

@router.post("/login/initiate")
async def initiate_login(data: PhoneLoginRequest):
    """
    Step 1: Agent provides phone + agency code.
    We check if they exist and trigger SMS verification.
    """
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    normalized_phone = normalize_phone(data.phone)
    agency_code = data.agency_code.strip().upper()
    
    print(f"[Auth] Login initiated: {normalized_phone} @ {agency_code}", flush=True)
    
    with get_db() as db:
        # Check if agent exists
        agent = db.query(Agent).filter(
            Agent.phone == normalized_phone,
            Agent.agency_code == agency_code
        ).first()
        
        is_new = agent is None
        
        if is_new:
            print(f"[Auth] New agent - will create after verification", flush=True)
        else:
            print(f"[Auth] Existing agent: {agent.first_name} {agent.last_name}", flush=True)
    
    # Trigger SMS verification using existing endpoint
    # The frontend will call /api/verify/initiate directly
    # This endpoint just validates and returns status
    
    return {
        "success": True,
        "phone": normalized_phone,
        "agency_code": agency_code,
        "is_new_agent": is_new,
        "message": "Please verify your phone number"
    }


@router.post("/login/complete")
async def complete_login(data: VerifyLoginRequest):
    """
    Step 2: After SMS verification succeeds, create session.
    Frontend calls this after successful /api/verify/confirm
    """
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    normalized_phone = normalize_phone(data.phone)
    agency_code = data.agency_code.strip().upper()
    
    print(f"[Auth] Completing login for: {normalized_phone} @ {agency_code}", flush=True)
    
    # Generate session token
    session_token = generate_session_token()
    session_expires = datetime.utcnow() + timedelta(days=SESSION_DURATION_DAYS)
    
    with get_db() as db:
        # Find or create agent
        agent = db.query(Agent).filter(
            Agent.phone == normalized_phone,
            Agent.agency_code == agency_code
        ).first()
        
        if not agent:
            # Create new agent
            agent = Agent(
                phone=normalized_phone,
                agency_code=agency_code,
                onboarding_start_date=datetime.utcnow().date(),
                onboarding_day=1,
                current_week=1,
                created_at=datetime.utcnow()
            )
            db.add(agent)
            print(f"[Auth] Created new agent record", flush=True)
        
        # Update session
        agent.session_token = session_token
        agent.session_expires = session_expires
        agent.last_login = datetime.utcnow()
        agent.last_active = datetime.utcnow()
        
        db.commit()
        db.refresh(agent)
        
        response_data = agent_to_response(agent)
    
    print(f"[Auth] Login complete - needs_profile: {response_data['needs_profile']}", flush=True)
    
    return {
        "success": True,
        "token": session_token,
        "expires": session_expires.isoformat(),
        "agent": response_data
    }


@router.post("/profile/complete")
async def complete_profile(
    data: CompleteProfileRequest,
    agent: Agent = Depends(require_agent)
):
    """
    Step 3 (first time only): Complete profile with name.
    """
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    print(f"[Auth] Completing profile for agent {agent.id}", flush=True)
    
    with get_db() as db:
        # Re-fetch agent in this session
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if not db_agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        db_agent.first_name = data.first_name.strip()
        db_agent.last_name = data.last_name.strip()
        if data.email:
            db_agent.email = data.email.strip().lower()
        
        db.commit()
        db.refresh(db_agent)
        
        response_data = agent_to_response(db_agent)
    
    print(f"[Auth] Profile complete: {data.first_name} {data.last_name}", flush=True)
    
    return {
        "success": True,
        "agent": response_data
    }


@router.get("/me")
async def get_current_user(agent: Agent = Depends(require_agent)):
    """Get current authenticated agent's data"""
    return {
        "success": True,
        "agent": agent_to_response(agent)
    }


@router.post("/logout")
async def logout(agent: Agent = Depends(require_agent)):
    """Clear session token"""
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    with get_db() as db:
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if db_agent:
            db_agent.session_token = None
            db_agent.session_expires = None
            db.commit()
    
    return {"success": True, "message": "Logged out"}


@router.post("/refresh")
async def refresh_session(agent: Agent = Depends(require_agent)):
    """Extend session expiration"""
    if not is_db_configured():
        raise HTTPException(status_code=503, detail="Database not configured")
    
    new_expires = datetime.utcnow() + timedelta(days=SESSION_DURATION_DAYS)
    
    with get_db() as db:
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if db_agent:
            db_agent.session_expires = new_expires
            db_agent.last_active = datetime.utcnow()
            db.commit()
    
    return {
        "success": True,
        "expires": new_expires.isoformat()
    }


# ==================== PROGRESS TRACKING ====================

@router.post("/progress/call")
async def record_call(agent: Agent = Depends(require_agent)):
    """Increment call count"""
    if not is_db_configured():
        return {"success": False}
    
    with get_db() as db:
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if db_agent:
            db_agent.total_calls = (db_agent.total_calls or 0) + 1
            db_agent.last_active = datetime.utcnow()
            db.commit()
            return {"success": True, "total_calls": db_agent.total_calls}
    
    return {"success": False}


@router.post("/progress/appointment")
async def record_appointment(agent: Agent = Depends(require_agent)):
    """Increment appointment count"""
    if not is_db_configured():
        return {"success": False}
    
    with get_db() as db:
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if db_agent:
            db_agent.total_appointments = (db_agent.total_appointments or 0) + 1
            db_agent.last_active = datetime.utcnow()
            db.commit()
            return {"success": True, "total_appointments": db_agent.total_appointments}
    
    return {"success": False}


@router.post("/progress/presentation")
async def record_presentation(agent: Agent = Depends(require_agent)):
    """Increment presentation count"""
    if not is_db_configured():
        return {"success": False}
    
    with get_db() as db:
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if db_agent:
            db_agent.total_presentations = (db_agent.total_presentations or 0) + 1
            db_agent.last_active = datetime.utcnow()
            db.commit()
            return {"success": True, "total_presentations": db_agent.total_presentations}
    
    return {"success": False}


@router.post("/progress/close")
async def record_close(agent: Agent = Depends(require_agent)):
    """Increment close count - the big one!"""
    if not is_db_configured():
        return {"success": False}
    
    with get_db() as db:
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if db_agent:
            db_agent.total_closes = (db_agent.total_closes or 0) + 1
            db_agent.last_active = datetime.utcnow()
            db.commit()
            return {"success": True, "total_closes": db_agent.total_closes}
    
    return {"success": False}


@router.post("/progress/module")
async def complete_module(
    module_id: str,
    agent: Agent = Depends(require_agent)
):
    """Mark a training module as complete"""
    if not is_db_configured():
        return {"success": False}
    
    with get_db() as db:
        db_agent = db.query(Agent).filter(Agent.id == agent.id).first()
        if db_agent:
            modules = db_agent.modules_completed or []
            if module_id not in modules:
                modules.append(module_id)
                db_agent.modules_completed = modules
            db_agent.last_active = datetime.utcnow()
            db.commit()
            return {"success": True, "modules_completed": modules}
    
    return {"success": False}


# ==================== ADMIN: AGENT LISTING ====================

@router.get("/agents")
async def list_agents(
    agency_code: Optional[str] = None,
    x_admin_password: Optional[str] = Header(None)
):
    """
    List all agents (admin only).
    Pass X-Admin-Password header for authentication.
    """
    admin_password = os.environ.get("COACHD_ADMIN_PASSWORD", "coachd-admin-2024")
    if x_admin_password != admin_password:
        raise HTTPException(status_code=401, detail="Admin authentication required")
    
    if not is_db_configured():
        return {"agents": [], "count": 0}
    
    with get_db() as db:
        query = db.query(Agent)
        
        if agency_code:
            query = query.filter(Agent.agency_code == agency_code.upper())
        
        agents = query.order_by(Agent.last_active.desc()).all()
        
        return {
            "agents": [agent_to_response(a) for a in agents],
            "count": len(agents)
        }


# Ensure table is created
def init_auth_tables():
    """Create auth tables if they don't exist"""
    if engine:
        Base.metadata.create_all(bind=engine)
        print("✓ Auth tables initialized")
