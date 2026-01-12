"""
Coachd Telnyx Routes
All Telnyx webhook handlers and API endpoints for TeXML integration
"""

import logging
import uuid
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from .config import settings
from .telnyx_bridge import (
    is_telnyx_configured,
    initiate_agent_call,
    generate_agent_conference_texml,
    generate_client_conference_texml,
    generate_inbound_texml,
    add_client_to_conference,
    end_conference,
    hangup_call,
    get_recording_url,
    _decode_client_state
)
from .call_session import session_manager, CallStatus
from .usage_tracker import log_telnyx_usage, get_dual_channel_cost_breakdown

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/telnyx", tags=["telnyx"])


# ============ PYDANTIC MODELS ============

class ClientContext(BaseModel):
    """Quick Prep context from agent"""
    product: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    weight: Optional[int] = None
    tobacco: Optional[str] = None
    married: Optional[str] = None
    kids: Optional[str] = None
    kidsCount: Optional[int] = None
    income: Optional[str] = None
    issue: Optional[str] = None
    budget: Optional[int] = None

class StartCallRequest(BaseModel):
    agent_phone: str
    client_phone: Optional[str] = None
    client_context: Optional[ClientContext] = None

class DialClientRequest(BaseModel):
    session_id: str
    client_phone: str
    agent_caller_id: Optional[str] = None

class EndCallRequest(BaseModel):
    session_id: str


# ============ HELPER FUNCTIONS ============

def texml_response(content: str) -> Response:
    """Return TeXML with proper content type"""
    return Response(content=content, media_type="application/xml")


# ============ API ENDPOINTS (Called by Frontend) ============

@router.post("/start-call")
async def start_call(data: StartCallRequest):
    """
    Start a new coaching session.
    App calls the agent, then auto-dials client if provided.
    """
    if not is_telnyx_configured():
        raise HTTPException(status_code=503, detail="Telnyx not configured")
    
    # Normalize agent phone number
    agent_phone = data.agent_phone
    if not agent_phone.startswith("+"):
        digits = agent_phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        agent_phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
    
    # Normalize client phone if provided
    client_phone = data.client_phone
    if client_phone:
        if not client_phone.startswith("+"):
            digits = client_phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
            client_phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
    
    # Extract client context if provided
    client_context = data.client_context.dict() if data.client_context else None
    
    # Create session with context AND client_phone for auto-dial
    session = await session_manager.create_session(
        agent_phone, 
        client_context=client_context,
        client_phone=client_phone
    )
    
    # Call the agent
    result = initiate_agent_call(agent_phone, session.session_id)
    
    if not result["success"]:
        await session_manager.update_session(session.session_id, status=CallStatus.FAILED)
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to start call"))
    
    await session_manager.update_session(
        session.session_id,
        agent_call_sid=result["call_control_id"],
        status=CallStatus.AGENT_RINGING
    )
    
    return {
        "success": True,
        "session_id": session.session_id,
        "message": "Calling your phone now. Answer to connect."
    }


@router.post("/dial-client")
async def dial_client(data: DialClientRequest):
    """Add client to existing conference"""
    session = await session_manager.get_session(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Normalize phone number
    client_phone = data.client_phone
    if not client_phone.startswith("+"):
        digits = client_phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
        client_phone = f"+1{digits}" if len(digits) == 10 else f"+{digits}"
    
    result = add_client_to_conference(client_phone, data.session_id, data.agent_caller_id)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to dial client"))
    
    await session_manager.update_session(
        data.session_id,
        client_phone=client_phone,
        client_call_sid=result["call_control_id"],
        status=CallStatus.CLIENT_RINGING
    )
    
    return {"success": True, "message": f"Dialing {client_phone}..."}


@router.post("/end-call")
async def end_call(data: EndCallRequest):
    """End a call session"""
    session = await session_manager.get_session(data.session_id)
    
    # Log usage with dual-channel tracking
    if session and session.started_at:
        duration = session.get_duration() or 0
        
        # Get individual leg durations if available
        agent_duration = getattr(session, 'agent_duration', None) or duration
        client_duration = getattr(session, 'client_duration', None) or duration
        
        log_telnyx_usage(
            call_duration_seconds=duration,
            agency_code=getattr(session, 'agency_code', None),
            session_id=data.session_id,
            call_control_id=session.agent_call_sid,
            is_dual_channel=True,  # All calls now use dual-channel
            agent_duration_seconds=agent_duration,
            client_duration_seconds=client_duration
        )
    
    end_conference(data.session_id)
    await session_manager.end_session(data.session_id)
    return {"success": True, "message": "Call ended"}


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.to_dict()


@router.get("/status")
async def get_status():
    """Check if Telnyx is configured"""
    return {
        "configured": is_telnyx_configured(),
        "phone_number": settings.telnyx_phone_number if is_telnyx_configured() else None,
        "base_url": settings.base_url
    }


# ============ TELNYX WEBHOOKS (Called by Telnyx) ============

@router.post("/incoming")
async def incoming_call(request: Request):
    """
    Handle inbound calls to the Telnyx number.
    Agent calls this number to start coaching.
    """
    try:
        form = await request.form()
        caller = form.get("From", "unknown")
        call_sid = form.get("CallSid", str(uuid.uuid4()))
        
        logger.info(f"Incoming call from {caller}")
        
        # Create a session for this inbound call
        session = await session_manager.create_session(caller)
        session_id = session.session_id
        
        await session_manager.update_session(
            session_id,
            agent_call_sid=call_sid,
            status=CallStatus.AGENT_CONNECTED,
            started_at=datetime.utcnow()
        )
        
        # Return TeXML to put caller in conference with streaming
        texml = generate_inbound_texml(session_id)
        return texml_response(texml)
        
    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        texml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Sorry, there was an error. Please try again.</Say>
    <Hangup />
</Response>"""
        return texml_response(texml)


@router.post("/agent-answered")
async def agent_answered(request: Request):
    """
    Webhook: Agent has answered the outbound call.
    Return TeXML to put them in conference with streaming.
    If client_phone was provided, automatically dial the client.
    """
    session_id = request.query_params.get("session_id", "unknown")
    
    logger.info(f"Agent answered for session {session_id}")
    
    # Get the session to check for client_phone
    session = await session_manager.get_session(session_id)
    
    await session_manager.update_session(
        session_id,
        status=CallStatus.AGENT_CONNECTED,
        started_at=datetime.utcnow()
    )
    
    # If client phone was provided (click-to-call mode), auto-dial them
    if session and session.client_phone:
        logger.info(f"Auto-dialing client {session.client_phone} for session {session_id}")
        
        # Broadcast client_dialing to frontend
        await session_manager._broadcast_to_session(session_id, {
            "type": "client_dialing",
            "message": "Dialing client..."
        })
        
        # Dial the client using agent's phone as caller ID
        result = add_client_to_conference(
            session.client_phone, 
            session_id, 
            agent_caller_id=session.agent_phone
        )
        
        if result["success"]:
            await session_manager.update_session(
                session_id,
                client_call_sid=result["call_control_id"],
                status=CallStatus.CLIENT_RINGING
            )
            logger.info(f"Client dial initiated: {result['call_control_id']}")
        else:
            logger.error(f"Failed to dial client: {result.get('error')}")
            # Broadcast error to frontend
            await session_manager._broadcast_to_session(session_id, {
                "type": "error",
                "message": f"Failed to dial client: {result.get('error', 'Unknown error')}"
            })
    
    texml = generate_agent_conference_texml(session_id)
    
    # Log the TeXML being returned
    print(f"[TeXML] Returning for session {session_id}:", flush=True)
    print(texml, flush=True)
    
    return texml_response(texml)


@router.post("/client-answered")
async def client_answered(request: Request):
    """
    Webhook: Client has answered.
    Return TeXML to put them in the conference.
    Broadcast client_connected to frontend.
    """
    session_id = request.query_params.get("session_id", "unknown")
    
    logger.info(f"Client answered for session {session_id}")
    
    await session_manager.update_session(session_id, status=CallStatus.IN_PROGRESS)
    
    # Broadcast client_connected to frontend - this triggers transition to guidance panel
    await session_manager._broadcast_to_session(session_id, {
        "type": "client_connected",
        "message": "Client connected"
    })
    
    texml = generate_client_conference_texml(session_id)
    return texml_response(texml)


@router.post("/call-status")
async def call_status(request: Request):
    """Webhook: Call status updates"""
    try:
        form = await request.form()
        status = form.get("CallStatus")
        session_id = request.query_params.get("session_id")
        
        logger.info(f"Call status: {status} for session {session_id}")
        
        if session_id and status == "completed":
            session = await session_manager.get_session(session_id)
            if session:
                await session_manager.end_session(session_id)
    except Exception as e:
        logger.error(f"Error handling call status: {e}")
    
    return Response(content="", status_code=200)


@router.post("/conference-status")
async def conference_status(request: Request):
    """Webhook: Conference events"""
    try:
        form = await request.form()
        event = form.get("StatusCallbackEvent")
        conference_sid = form.get("ConferenceSid")
        session_id = request.query_params.get("session_id")
        
        logger.info(f"Conference event: {event} for session {session_id}")
        
        if session_id and conference_sid:
            await session_manager.update_session(session_id, conference_sid=conference_sid)
        
    except Exception as e:
        logger.error(f"Error handling conference status: {e}")
    
    return Response(content="", status_code=200)


@router.get("/ringback")
@router.post("/ringback")
async def ringback_audio():
    """
    Returns TeXML that plays US ringback tone while agent waits.
    Traditional 440Hz + 480Hz tone, 2 sec on, 4 sec off.
    When client joins, this stops automatically - the natural phone experience.
    """
    ringback_file = f"{settings.base_url}/static/ringback.wav"
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play loop="0">{ringback_file}</Play>
</Response>"""
    
    return Response(content=texml, media_type="application/xml")


@router.post("/recording-complete")
async def recording_complete(request: Request):
    """Webhook: Recording finished"""
    try:
        form = await request.form()
        recording_sid = form.get("RecordingSid")
        recording_status = form.get("RecordingStatus")
        session_id = request.query_params.get("session_id")
        
        logger.info(f"Recording complete: {recording_sid} for session {session_id}")
        
        if session_id and recording_sid and recording_status == "completed":
            recording_url = get_recording_url(recording_sid)
            await session_manager.update_session(
                session_id,
                recording_sid=recording_sid,
                recording_url=recording_url
            )
    except Exception as e:
        logger.error(f"Error handling recording: {e}")
    
    return Response(content="", status_code=200)


@router.post("/webhook")
async def main_webhook(request: Request):
    """
    Main webhook handler for Call Control API events.
    Telnyx can send various event types here.
    """
    try:
        body = await request.json()
        data = body.get("data", {})
        event_type = data.get("event_type", "")
        payload = data.get("payload", {})
        
        logger.info(f"Webhook event: {event_type}")
        
        # Extract session from client_state
        client_state = payload.get("client_state", "")
        state_data = _decode_client_state(client_state) if client_state else {}
        session_id = state_data.get("session_id")
        
        if event_type == "call.hangup" and session_id:
            session = await session_manager.get_session(session_id)
            if session:
                await session_manager.end_session(session_id)
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
    
    return Response(content="", status_code=200)