"""
Coachd Power Dialer Routes
API endpoints for Power Dialer - appointment booking tool
2-step connection (Agent → Client), no AI/transcription

ARCHITECTURE: Agent connects ONCE at session start, stays in conference.
Clients get dialed in/out one at a time. Agent only hangs up at session end.
"""

import logging
import uuid
import re
import json
import base64
import requests
import anthropic
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, WebSocket, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from .config import settings
from .telnyx_bridge import (
    is_telnyx_configured,
    get_telnyx_api_key,
    get_telnyx_app_id,
    get_telnyx_phone,
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
        
        # Call state - AGENT STAYS CONNECTED
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
        """Duration of current client call"""
        if self.call_started_at and self.client_connected:
            return int((datetime.utcnow() - self.call_started_at).total_seconds())
        return 0
    
    def get_session_duration(self) -> int:
        """Total session duration"""
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

class SessionIdRequest(BaseModel):
    session_id: str

class DispositionRequest(BaseModel):
    session_id: str
    contact_id: int
    disposition: str
    scheduled_date: Optional[str] = None
    scheduled_time: Optional[str] = None
    notes: Optional[str] = None


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
    Call the agent's phone for Power Dialer session start.
    Agent will stay connected for the entire session.
    """
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    from_number = get_telnyx_phone()
    
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


def dial_client_into_conference(client_phone: str, session_id: str, agent_caller_id: str) -> Dict:
    """Dial a client into the existing conference where agent is waiting"""
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
    """
    Start a new Power Dialer session.
    This creates the session and calls the AGENT (one-time connection).
    Agent stays on the line for the entire session.
    """
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
    
    # Call the agent - they'll connect once and stay on
    result = initiate_dialer_agent_call(agent_phone, session_id)
    
    if not result.get("success"):
        remove_dialer_session(session_id)
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to call agent"))
    
    state.agent_call_sid = result.get("call_control_id")
    
    return {
        "success": True,
        "session_id": session_id,
        "total_contacts": len(data.contacts),
        "list_type": data.list_type,
        "message": "Calling your phone - answer to start dialing"
    }


@router.post("/dial-next")
async def dial_next(data: SessionIdRequest):
    """
    Dial the next contact into the conference.
    Agent is already connected and waiting.
    """
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not state.agent_connected:
        raise HTTPException(status_code=400, detail="Agent not connected yet")
    
    # If there's still a client connected, hang them up first
    if state.client_call_sid:
        hangup_call(state.client_call_sid)
        state.client_call_sid = None
        state.client_connected = False
    
    if state.current_index >= len(state._contacts):
        return {"success": False, "complete": True, "message": "No more contacts"}
    
    contact = state._contacts[state.current_index]
    
    # Update state
    state.current_contact_id = contact["id"]
    state.current_contact_name = contact["name"]
    state.current_contact_phone = contact["phone"]
    state.client_connected = False
    state.call_started_at = None
    
    # Dial client into the existing conference
    result = dial_client_into_conference(
        contact["phone"], 
        data.session_id, 
        state.agent_phone
    )
    
    if not result.get("success"):
        # Don't fail the session, just report the error
        await state.broadcast({"type": "dial_failed", "contact": contact, "error": result.get("error")})
        return {"success": False, "error": result.get("error"), "contact": contact}
    
    state.client_call_sid = result.get("call_control_id")
    
    await state.broadcast({"type": "dialing_client", "contact": contact})
    
    return {"success": True, "contact": contact, "message": f"Dialing {contact['name']}..."}


@router.post("/skip")
async def skip_contact(data: SessionIdRequest):
    """Skip current contact and move to next"""
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Log skip disposition
    if state.current_contact_id and is_db_configured():
        update_dialer_contact_disposition(state.current_contact_id, "skipped")
    
    # Hang up client only (agent stays)
    if state.client_call_sid:
        hangup_call(state.client_call_sid)
        state.client_call_sid = None
    
    state.current_index += 1
    state.client_connected = False
    state.current_contact_id = None
    state.current_contact_name = None
    state.current_contact_phone = None
    
    await state.broadcast({"type": "contact_skipped"})
    return {"success": True}


@router.post("/end-call")
async def end_call(data: SessionIdRequest):
    """
    End current CLIENT call only.
    Agent stays connected for next call.
    """
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    duration = state.get_call_duration()
    
    # Hang up CLIENT only - agent stays in conference
    if state.client_call_sid:
        hangup_call(state.client_call_sid)
        state.client_call_sid = None
    
    # Log usage for this call segment
    if duration > 0:
        log_telnyx_usage(
            call_duration_seconds=duration,
            session_id=data.session_id,
            call_control_id=state.client_call_sid,
            is_dual_channel=False
        )
    
    state.client_connected = False
    state.call_started_at = None
    
    await state.broadcast({"type": "call_ended", "duration": duration})
    
    return {
        "success": True,
        "call_duration": duration,
        "contact": {"id": state.current_contact_id, "name": state.current_contact_name},
        "agent_still_connected": state.agent_connected
    }


@router.post("/disposition")
async def save_disposition(data: DispositionRequest):
    """Save disposition for current contact"""
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    duration = state.get_call_duration()
    
    if is_db_configured():
        update_dialer_contact_disposition(
            contact_id=data.contact_id,
            disposition=data.disposition,
            duration_seconds=duration,
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
    
    # Move to next contact
    state.current_index += 1
    state.current_contact_id = None
    state.current_contact_name = None
    state.current_contact_phone = None
    
    has_more = state.current_index < state.total_contacts
    next_contact = state._contacts[state.current_index] if has_more else None
    
    await state.broadcast({
        "type": "disposition_saved",
        "disposition": data.disposition,
        "has_more": has_more
    })
    
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
async def end_session_route(data: SessionIdRequest):
    """
    End entire dialing session.
    This is the ONLY time the agent gets disconnected.
    """
    state = get_dialer_session(data.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Now we hang up the AGENT (ending their session)
    if state.agent_call_sid:
        hangup_call(state.agent_call_sid)
    if state.client_call_sid:
        hangup_call(state.client_call_sid)
    
    end_conference(data.session_id)
    
    # Log total session usage
    session_duration = state.get_session_duration()
    if session_duration > 0:
        log_telnyx_usage(
            call_duration_seconds=session_duration,
            session_id=data.session_id,
            call_control_id=state.agent_call_sid,
            is_dual_channel=False
        )
    
    stats = {
        "calls_made": state.current_index,
        "appointments": state.appointments,
        "duration": session_duration
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
    """
    Agent answered the session call.
    Put them in conference where they'll STAY for the entire session.
    """
    session_id = request.query_params.get("session_id", "unknown")
    print(f"[Dialer] Agent answered session: {session_id}", flush=True)
    
    state = get_dialer_session(session_id)
    if not state:
        return texml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response><Hangup /></Response>""")
    
    contact_count = state.total_contacts
    first_contact = state._contacts[0]["name"] if state._contacts else "your contacts"
    
    # DTMF gate - agent presses 1 to confirm they're ready
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Gather action="{settings.base_url}/api/dialer/agent-ready?session_id={session_id}" method="POST" numDigits="1" timeout="30">
        <Say voice="Polly.Joanna">Power Dialer ready. {contact_count} contacts loaded. Press 1 to start dialing {first_contact}.</Say>
    </Gather>
    <Say voice="Polly.Joanna">No response. Session cancelled.</Say>
    <Hangup />
</Response>"""
    return texml_response(texml)


@router.post("/agent-ready")
async def dialer_agent_ready(request: Request):
    """
    Agent pressed 1 - put them in PERSISTENT conference.
    They stay here until session ends.
    """
    session_id = request.query_params.get("session_id", "unknown")
    
    form = await request.form()
    digits = form.get("Digits", "")
    
    print(f"[Dialer] Agent ready: {session_id}, digits: {digits}", flush=True)
    
    state = get_dialer_session(session_id)
    if not state:
        return texml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response><Hangup /></Response>""")
    
    if digits != "1":
        remove_dialer_session(session_id)
        return texml_response("""<?xml version="1.0" encoding="UTF-8"?>
<Response><Say voice="Polly.Joanna">Session cancelled.</Say><Hangup /></Response>""")
    
    state.agent_connected = True
    await state.broadcast({"type": "agent_connected", "message": "Ready to dial"})
    
    conference_name = f"dialer_{session_id}"
    silence_url = f"{settings.base_url}/api/telnyx/silence"
    
    # Agent joins conference and STAYS
    # endConferenceOnExit="true" - if agent hangs up, conference ends
    # startConferenceOnEnter="true" - conference starts when agent joins
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Connected. Click dial to start calling.</Say>
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


@router.post("/client-answered")
async def dialer_client_answered(request: Request):
    """
    Client answered - join them to the conference where agent is waiting.
    Client's exit does NOT end the conference.
    """
    session_id = request.query_params.get("session_id", "unknown")
    print(f"[Dialer] Client answered: {session_id}", flush=True)
    
    state = get_dialer_session(session_id)
    if state:
        state.client_connected = True
        state.call_started_at = datetime.utcnow()
        await state.broadcast({
            "type": "client_connected",
            "contact": {
                "name": state.current_contact_name,
                "phone": state.current_contact_phone
            }
        })
    
    conference_name = f"dialer_{session_id}"
    silence_url = f"{settings.base_url}/api/telnyx/silence"
    
    # Client joins conference but does NOT end it when they leave
    # endConferenceOnExit="false" - agent stays when client hangs up
    # startConferenceOnEnter="false" - conference already started by agent
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
    """Handle call status updates from Telnyx"""
    session_id = request.query_params.get("session_id", "unknown")
    call_type = request.query_params.get("type", "unknown")
    
    form = await request.form()
    status = form.get("CallStatus", "unknown")
    
    print(f"[Dialer] Status: {session_id}, {call_type}, {status}", flush=True)
    
    state = get_dialer_session(session_id)
    if not state:
        return {"received": True}
    
    if call_type == "client":
        if status in ("no-answer", "busy", "failed", "canceled"):
            state.client_connected = False
            state.client_call_sid = None
            await state.broadcast({"type": "client_unavailable", "reason": status})
        elif status == "completed":
            # Client hung up - agent stays connected
            state.client_connected = False
            state.client_call_sid = None
            await state.broadcast({"type": "client_disconnected"})
    
    elif call_type == "agent":
        if status == "completed":
            # Agent hung up - end the whole session
            state.agent_connected = False
            await state.broadcast({"type": "agent_disconnected"})
            # Clean up session since agent left
            if is_db_configured():
                end_dialer_session(session_id)
            remove_dialer_session(session_id)
    
    return {"received": True}


# ============ CLAUDE VISION IMAGE PARSING ============

@router.post("/parse-image")
async def parse_image_contacts(image: UploadFile = File(...)):
    """
    Use Claude Vision to extract contacts from a photo of a list.
    Returns structured contact data (name, phone, optional sponsor).
    """
    try:
        # Read and encode image
        image_data = await image.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine media type
        content_type = image.content_type or 'image/jpeg'
        if 'png' in content_type:
            media_type = 'image/png'
        elif 'gif' in content_type:
            media_type = 'image/gif'
        elif 'webp' in content_type:
            media_type = 'image/webp'
        else:
            media_type = 'image/jpeg'
        
        # Call Claude Vision
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": """Extract all contacts (names and phone numbers) from this image.

Return ONLY a JSON array with this exact format, no other text:
[
  {"name": "John Smith", "phone": "+14045551234", "sponsor": "Jane Doe"},
  {"name": "Maria Garcia", "phone": "+16785559876", "sponsor": null}
]

Rules:
- Extract every name and phone number visible
- Format phone as +1XXXXXXXXXX (US format with country code)
- If there's a sponsor/referrer mentioned, include it
- If no sponsor, set sponsor to null
- Return empty array [] if no contacts found
- Return ONLY the JSON array, no explanations"""
                        }
                    ]
                }
            ]
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        
        # Handle if Claude wrapped it in markdown code blocks
        if '```' in response_text:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                response_text = json_match.group(1)
        
        contacts = json.loads(response_text)
        
        # Validate and clean contacts
        valid_contacts = []
        for i, contact in enumerate(contacts):
            if isinstance(contact, dict) and 'name' in contact and 'phone' in contact:
                phone = re.sub(r'\D', '', contact.get('phone', ''))
                if len(phone) >= 10:
                    if len(phone) == 10:
                        phone = f"+1{phone}"
                    elif len(phone) == 11 and phone.startswith('1'):
                        phone = f"+{phone}"
                    else:
                        phone = f"+{phone}"
                    
                    valid_contacts.append({
                        "id": i + 1,
                        "name": contact.get('name', '').strip(),
                        "phone": phone,
                        "sponsor": contact.get('sponsor') if contact.get('sponsor') else None
                    })
        
        print(f"[Dialer] Parsed {len(valid_contacts)} contacts from image", flush=True)
        return {"success": True, "contacts": valid_contacts}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        return {"success": False, "contacts": [], "error": "Could not parse contacts from image"}
    except Exception as e:
        logger.error(f"Image parsing failed: {e}")
        return {"success": False, "contacts": [], "error": str(e)}
