"""
Coachd Power Dialer Routes
API endpoints for Power Dialer - appointment booking tool
2-step connection (Agent → Client), no AI/transcription
"""

import logging
import uuid
import re
import requests
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from .config import settings
from .telnyx_bridge import (
    is_telnyx_configured,
    get_telnyx_api_key,
    get_telnyx_app_id,
    get_telnyx_phone,
    add_client_to_conference,
    hangup_call,
    end_conference
)
from .usage_tracker import log_telnyx_usage
from .database import (
    is_db_configured,
    create_dialer_session,
    add_dialer_contacts,
    get_next_dialer_contact,
    update_dialer_contact_disposition,
    create_dialer_appointment,
    update_dialer_session_stats,
    end_dialer_session,
    get_dialer_session_summary,
    get_upcoming_appointments
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/dialer", tags=["dialer"])


# ============ IN-MEMORY SESSION STATE ============

class DialerCallState:
    """Real-time state for an active dialer session"""
    def __init__(self, session_id: str, agent_phone: str):
        self.session_id = session_id
        self.agent_phone = agent_phone
        self.agent_call_sid: Optional[str] = None
        self.client_call_sid: Optional[str] = None
        self.conference_name: str = f"dialer_{session_id}"
        
        # Current contact
        self.current_contact_id: Optional[int] = None
        self.current_contact_name: Optional[str] = None
        self.current_contact_phone: Optional[str] = None
        
        # Call state
        self.agent_connected: bool = False
        self.client_connected: bool = False
        self.call_started_at: Optional[datetime] = None
        
        # Session stats
        self.current_index: int = 0
        self.total_contacts: int = 0
        self.appointments: int = 0
        self.session_started_at: datetime = datetime.utcnow()
        
        # Contact list (in-memory)
        self._contacts: List[Dict] = []
        
        # WebSocket connections
        self._websockets: List[WebSocket] = []
    
    def get_call_duration(self) -> int:
        if self.call_started_at:
            return int((datetime.utcnow() - self.call_started_at).total_seconds())
        return 0
    
    def get_session_duration(self) -> int:
        return int((datetime.utcnow() - self.session_started_at).total_seconds())
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "agent_connected": self.agent_connected,
            "client_connected": self.client_connected,
            "current_contact": {
                "id": self.current_contact_id,
                "name": self.current_contact_name,
                "phone": self.current_contact_phone
            } if self.current_contact_id else None,
            "call_duration": self.get_call_duration(),
            "session_duration": self.get_session_duration(),
            "progress": {
                "current": self.current_index,
                "total": self.total_contacts,
                "appointments": self.appointments
            }
        }
    
    async def broadcast(self, message: Dict):
        """Send message to all connected WebSockets"""
        for ws in self._websockets[:]:
            try:
                await ws.send_json(message)
            except:
                self._websockets.remove(ws)


# Session store
_dialer_sessions: Dict[str, DialerCallState] = {}


def get_dialer_session(session_id: str) -> Optional[DialerCallState]:
    return _dialer_sessions.get(session_id)


def remove_dialer_session(session_id: str):
    if session_id in _dialer_sessions:
        del _dialer_sessions[session_id]


# ============ PYDANTIC MODELS ============

class Contact(BaseModel):
    name: str
    phone: str
    sponsor: Optional[str] = None

class StartSessionRequest(BaseModel):
    agent_phone: str
    list_type: str
    contacts: List[Contact]
    agency_code: Optional[str] = None

class DialNextRequest(BaseModel):
    session_id: str

class DispositionRequest(BaseModel):
    session_id: str
    contact_id: int
    disposition: str
    scheduled_date: Optional[str] = None
    scheduled_time: Optional[str] = None
    notes: Optional[str] = None

class EndSessionRequest(BaseModel):
    session_id: str

class SkipContactRequest(BaseModel):
    session_id: str


# ============ HELPERS ============

def normalize_phone(phone: str) -> str:
    """Normalize phone to E.164"""
    digits = re.sub(r'\D', '', phone)
    if len(digits) == 10:
        return f"+1{digits}"
    elif len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}"


def texml_response(content: str) -> Response:
    return Response(content=content, media_type="application/xml")


def initiate_dialer_agent_call(agent_phone: str, session_id: str) -> Dict:
    """
    Call the agent's phone for Power Dialer.
    Simplified version - no audio streaming, just DTMF gate.
    """
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    from_number = get_telnyx_phone()
    
    # Status callback for call events
    status_callback = f"{settings.base_url}/api/dialer/status?session_id={session_id}&type=agent"
    
    try:
        response = requests.post(
            f"https://api.telnyx.com/v2/texml/calls/{app_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "To": agent_phone,
                "From": from_number,
                "Url": f"{settings.base_url}/api/dialer/agent-answered?session_id={session_id}",
                "Method": "POST",
                "StatusCallback": status_callback,
                "StatusCallbackEvent": "initiated ringing answered completed",
                "Timeout": 30
            }
        )
        
        if response.status_code in [200, 201]:
            response_json = response.json()
            data = response_json.get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            if not call_control_id:
                call_control_id = response_json.get("call_sid", "") or response_json.get("sid", "")
            
            print(f"[Dialer] Agent call initiated: {call_control_id}", flush=True)
            return {"success": True, "call_control_id": call_control_id}
        else:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                if "errors" in error_data:
                    error_msg = error_data["errors"][0].get("detail", error_msg)
            except:
                pass
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.error(f"Failed to initiate dialer agent call: {e}")
        return {"success": False, "error": str(e)}


# ============ WEBSOCKET ============

@router.websocket("/ws/{session_id}")
async def dialer_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time call status updates"""
    await websocket.accept()
    
    state = get_dialer_session(session_id)
    if not state:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    state._websockets.append(websocket)
    
    try:
        await websocket.send_json({"type": "session_state", "data": state.to_dict()})
        
        while True:
            try:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_json({"type": "pong"})
            except:
                break
    finally:
        if websocket in state._websockets:
            state._websockets.remove(websocket)


# ============ API ENDPOINTS ============

@router.post("/start-session")
async def start_session(data: StartSessionRequest):
    """Start a new Power Dialer session"""
    if not is_telnyx_configured():
        raise HTTPException(status_code=503, detail="Telnyx not configured")
    
    if not data.contacts:
        raise HTTPException(status_code=400, detail="No contacts provided")
    
    agent_phone = normalize_phone(data.agent_phone)
    session_id = f"dialer_{uuid.uuid4().hex[:12]}"
    
    # Create state
    state = DialerCallState(session_id, agent_phone)
    state.total_contacts = len(data.contacts)
    state._contacts = [
        {"id": i+1, "name": c.name, "phone": normalize_phone(c.phone), "sponsor": c.sponsor}
        for i, c in enumerate(data.contacts)
    ]
    _dialer_sessions[session_id] = state
    
    # Persist to DB
    if is_db_configured():
        create_dialer_session(
            session_id=session_id,
            agent_phone=agent_phone,
            list_type=data.list_type,
            total_contacts=len(data.contacts),
            agency_code=data.agency_code
        )
        add_dialer_contacts(session_id, [
            {"name": c.name, "phone": normalize_phone(c.phone), "sponsor": c.sponsor}
            for c in data.contacts
        ])
    
    return {
        "success": True,
        "session_id": session_id,
        "total_contacts": len(data.contacts),
        "list_type": data.list_type
    }


@router.post("/dial-next")
async def dial_next(data: DialNextRequest):
    """Dial the next contact"""
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.current_index >= len(state._contacts):
        return {"success": False, "complete": True, "message": "No more contacts"}
    
    contact = state._contacts[state.current_index]
    
    # Update state
    state.current_contact_id = contact["id"]
    state.current_contact_name = contact["name"]
    state.current_contact_phone = contact["phone"]
    state.agent_connected = False
    state.client_connected = False
    state.call_started_at = None
    
    # Call agent
    result = initiate_dialer_agent_call(state.agent_phone, data.session_id)
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to call"))
    
    state.agent_call_sid = result.get("call_control_id")
    
    await state.broadcast({"type": "dialing_agent", "contact": contact})
    
    return {"success": True, "contact": contact, "message": "Calling your phone..."}


@router.post("/skip")
async def skip_contact(data: SkipContactRequest):
    """Skip current contact"""
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.current_contact_id and is_db_configured():
        update_dialer_contact_disposition(state.current_contact_id, "skipped")
    
    if state.agent_call_sid:
        hangup_call(state.agent_call_sid)
    if state.client_call_sid:
        hangup_call(state.client_call_sid)
    
    state.current_index += 1
    state.agent_call_sid = None
    state.client_call_sid = None
    state.agent_connected = False
    state.client_connected = False
    
    await state.broadcast({"type": "contact_skipped"})
    return {"success": True}


@router.post("/end-call")
async def end_call(data: EndSessionRequest):
    """End current call"""
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    duration = state.get_call_duration()
    
    if state.agent_call_sid:
        hangup_call(state.agent_call_sid)
    if state.client_call_sid:
        hangup_call(state.client_call_sid)
    
    end_conference(data.session_id)
    
    if duration > 0:
        log_telnyx_usage(
            call_duration_seconds=duration,
            session_id=data.session_id,
            call_control_id=state.agent_call_sid,
            is_dual_channel=True
        )
    
    state.agent_call_sid = None
    state.client_call_sid = None
    state.agent_connected = False
    state.client_connected = False
    
    await state.broadcast({"type": "call_ended", "duration": duration})
    
    return {
        "success": True,
        "call_duration": duration,
        "contact": {"id": state.current_contact_id, "name": state.current_contact_name}
    }


@router.post("/disposition")
async def save_disposition(data: DispositionRequest):
    """Save disposition and create appointment if needed"""
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if is_db_configured():
        update_dialer_contact_disposition(
            contact_id=data.contact_id,
            disposition=data.disposition,
            duration_seconds=state.get_call_duration(),
            notes=data.notes
        )
        
        if data.disposition == "appointment" and data.scheduled_date:
            scheduled_date = datetime.fromisoformat(data.scheduled_date)
            create_dialer_appointment(
                contact_id=data.contact_id,
                session_id=data.session_id,
                contact_name=state.current_contact_name or "",
                contact_phone=state.current_contact_phone or "",
                scheduled_date=scheduled_date,
                scheduled_time=data.scheduled_time or "12:00",
                agent_phone=state.agent_phone
            )
            state.appointments += 1
        
        update_dialer_session_stats(data.session_id)
    
    state.current_index += 1
    state.current_contact_id = None
    state.current_contact_name = None
    state.current_contact_phone = None
    
    has_more = state.current_index < state.total_contacts
    next_contact = state._contacts[state.current_index] if has_more else None
    
    return {
        "success": True,
        "has_more": has_more,
        "next_contact": next_contact,
        "stats": {
            "current": state.current_index,
            "total": state.total_contacts,
            "appointments": state.appointments
        }
    }


@router.post("/end-session")
async def end_session_route(data: EndSessionRequest):
    """End entire session"""
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if state.agent_call_sid:
        hangup_call(state.agent_call_sid)
    if state.client_call_sid:
        hangup_call(state.client_call_sid)
    
    end_conference(data.session_id)
    
    stats = {
        "calls_made": state.current_index,
        "appointments": state.appointments,
        "duration": state.get_session_duration()
    }
    
    if is_db_configured():
        end_dialer_session(data.session_id)
        db_stats = get_dialer_session_summary(data.session_id)
        if db_stats:
            stats = db_stats
    
    await state.broadcast({"type": "session_ended", "stats": stats})
    remove_dialer_session(data.session_id)
    
    return {"success": True, "summary": stats}


@router.get("/session/{session_id}")
async def get_session_status(session_id: str):
    """Get session status"""
    state = get_dialer_session(session_id)
    if state:
        return state.to_dict()
    
    if is_db_configured():
        summary = get_dialer_session_summary(session_id)
        if summary:
            return summary
    
    raise HTTPException(status_code=404, detail="Session not found")


@router.get("/appointments")
async def list_appointments(agency_code: Optional[str] = None, days: int = 7):
    """Get upcoming appointments"""
    if not is_db_configured():
        return {"appointments": []}
    return {"appointments": get_upcoming_appointments(agency_code, days)}


# ============ TELNYX WEBHOOKS ============

@router.post("/agent-answered")
async def dialer_agent_answered(request: Request):
    """Agent answered - DTMF gate (no streaming)"""
    session_id = request.query_params.get("session_id", "unknown")
    print(f"[Dialer] Agent answered: {session_id}", flush=True)
    
    state = get_dialer_session(session_id)
    if not state:
        return texml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response><Hangup /></Response>""")
    
    contact_name = state.current_contact_name or "the contact"
    
    # Simple DTMF gate - no audio streaming needed for dialer
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather action="{settings.base_url}/api/dialer/agent-ready?session_id={session_id}" method="POST" numDigits="1" timeout="30">
        <Say voice="Polly.Joanna">Press 1 to dial {contact_name}.</Say>
    </Gather>
    <Say voice="Polly.Joanna">No response. Goodbye.</Say>
    <Hangup />
</Response>"""
    return texml_response(texml)


@router.post("/agent-ready")
async def dialer_agent_ready(request: Request):
    """Agent pressed 1 - put in conference and dial client"""
    session_id = request.query_params.get("session_id", "unknown")
    
    form = await request.form()
    digits = form.get("Digits", "")
    
    print(f"[Dialer] Agent ready: {session_id}, digits: {digits}", flush=True)
    
    state = get_dialer_session(session_id)
    if not state:
        return texml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response><Hangup /></Response>""")
    
    if digits != "1":
        return texml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response><Say voice="Polly.Joanna">Cancelled.</Say><Hangup /></Response>""")
    
    state.agent_connected = True
    await state.broadcast({"type": "agent_connected"})
    
    # Dial client using existing telnyx_bridge function
    if state.current_contact_phone:
        # Use dialer-specific webhook for client
        result = dial_client_for_dialer(
            state.current_contact_phone, 
            session_id, 
            state.agent_phone
        )
        if result.get("success"):
            state.client_call_sid = result.get("call_control_id")
            await state.broadcast({"type": "dialing_client"})
    
    conference_name = f"dialer_{session_id}"
    silence_url = f"{settings.base_url}/api/telnyx/silence"
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Dialing. Please hold.</Say>
    <Dial>
        <Conference
            startConferenceOnEnter="true"
            endConferenceOnExit="true"
            beep="false"
            waitUrl="{silence_url}">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    return texml_response(texml)


def dial_client_for_dialer(client_phone: str, session_id: str, agent_caller_id: str) -> Dict:
    """Dial client with dialer-specific webhook"""
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    from_number = agent_caller_id if agent_caller_id else get_telnyx_phone()
    
    status_callback = f"{settings.base_url}/api/dialer/status?session_id={session_id}&type=client"
    
    try:
        response = requests.post(
            f"https://api.telnyx.com/v2/texml/calls/{app_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "To": client_phone,
                "From": from_number,
                "Url": f"{settings.base_url}/api/dialer/client-answered?session_id={session_id}",
                "Method": "POST",
                "StatusCallback": status_callback,
                "StatusCallbackEvent": "initiated ringing answered completed",
                "Timeout": 25
            }
        )
        
        if response.status_code in [200, 201]:
            response_json = response.json()
            data = response_json.get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            if not call_control_id:
                call_control_id = response_json.get("call_sid", "") or response_json.get("sid", "")
            
            print(f"[Dialer] Client call initiated: {call_control_id}", flush=True)
            return {"success": True, "call_control_id": call_control_id}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/client-answered")
async def dialer_client_answered(request: Request):
    """Client answered - join conference (no streaming)"""
    session_id = request.query_params.get("session_id", "unknown")
    print(f"[Dialer] Client answered: {session_id}", flush=True)
    
    state = get_dialer_session(session_id)
    if state:
        state.client_connected = True
        state.call_started_at = datetime.utcnow()
        await state.broadcast({"type": "client_connected"})
    
    conference_name = f"dialer_{session_id}"
    silence_url = f"{settings.base_url}/api/telnyx/silence"
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Conference
            startConferenceOnEnter="false"
            endConferenceOnExit="false"
            beep="false"
            waitUrl="{silence_url}">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    return texml_response(texml)


@router.post("/status")
async def dialer_status_callback(request: Request):
    """Call status updates"""
    session_id = request.query_params.get("session_id", "unknown")
    call_type = request.query_params.get("type", "unknown")
    
    form = await request.form()
    status = form.get("CallStatus", "unknown")
    
    print(f"[Dialer] Status: {session_id}, {call_type}, {status}", flush=True)
    
    state = get_dialer_session(session_id)
    if state and call_type == "client" and status in ("no-answer", "busy", "failed", "canceled"):
        state.client_connected = False
        await state.broadcast({"type": "client_unavailable", "reason": status})
    
    return {"received": True}
