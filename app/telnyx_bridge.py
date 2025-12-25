"""
Coachd Telnyx Bridge - Dual-Channel Click-to-Call
==================================================
Agent clicks start → System calls agent → Agent answers → System calls client
→ Separate audio streams → Client audio to Deepgram → 100% speaker accuracy
"""

import logging
import requests
import os
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


def get_telnyx_phone() -> str:
    """Get Telnyx phone number"""
    return os.environ.get("TELNYX_PHONE_NUMBER", "") or settings.telnyx_phone_number


def get_telnyx_api_key() -> str:
    """Get Telnyx API key"""
    return os.environ.get("TELNYX_API_KEY", "") or settings.telnyx_api_key


def get_telnyx_app_id() -> str:
    """Get Telnyx App ID"""
    return os.environ.get("TELNYX_APP_ID", "") or settings.telnyx_app_id


def is_telnyx_configured() -> bool:
    """Check if Telnyx is properly configured"""
    return bool(get_telnyx_api_key() and get_telnyx_phone())


def normalize_phone(phone: str) -> str:
    """Normalize phone number to E.164 format"""
    if not phone:
        return ""
    if phone.startswith("+"):
        return phone
    digits = phone.replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
    return f"+1{digits}" if len(digits) == 10 else f"+{digits}"


def _extract_error(response) -> str:
    """Extract error message from Telnyx API response"""
    try:
        error_data = response.json()
        if "errors" in error_data:
            return error_data["errors"][0].get("detail", "Unknown error")
        elif "error" in error_data:
            return error_data["error"]
    except:
        pass
    return f"HTTP {response.status_code}"


def initiate_click_to_call(agent_phone: str, client_phone: str, session_id: str) -> dict:
    """
    Start click-to-call: Call agent first, then client when agent answers.
    """
    from_number = get_telnyx_phone()
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    
    agent_phone = normalize_phone(agent_phone)
    client_phone = normalize_phone(client_phone)
    
    logger.info(f"[Telnyx] Click-to-call: agent={agent_phone}, client={client_phone}")
    print(f"[Telnyx] Click-to-call: agent={agent_phone}, client={client_phone}", flush=True)
    
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    try:
        webhook_url = (
            f"{settings.base_url}/api/telnyx/agent-answered"
            f"?session_id={session_id}"
            f"&client_phone={client_phone}"
            f"&agent_phone={agent_phone}"
        )
        
        response = requests.post(
            f"https://api.telnyx.com/v2/texml/calls/{app_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "To": agent_phone,
                "From": from_number,
                "Url": webhook_url,
                "Method": "POST"
            }
        )
        
        if response.status_code in [200, 201]:
            data = response.json().get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            
            logger.info(f"[Telnyx] Agent call initiated: {call_control_id}")
            
            return {
                "success": True,
                "call_control_id": call_control_id,
                "session_id": session_id,
                "agent_phone": agent_phone,
                "client_phone": client_phone
            }
        else:
            error_msg = _extract_error(response)
            logger.error(f"[Telnyx] Agent call failed: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.error(f"[Telnyx] Exception: {e}")
        return {"success": False, "error": str(e)}


def dial_client(client_phone: str, agent_caller_id: str, session_id: str) -> dict:
    """
    Dial client. Uses Telnyx number as caller ID.
    (To use agent's number, verify it in Telnyx Portal → Verified Numbers)
    """
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    
    client_phone = normalize_phone(client_phone)
    # Always use Telnyx number - agent number requires verification
    from_number = get_telnyx_phone()
    
    logger.info(f"[Telnyx] Dialing client {client_phone} from {from_number}")
    print(f"[Telnyx] Dialing client {client_phone} from {from_number}", flush=True)
    
    try:
        webhook_url = f"{settings.base_url}/api/telnyx/client-answered?session_id={session_id}"
        
        response = requests.post(
            f"https://api.telnyx.com/v2/texml/calls/{app_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "To": client_phone,
                "From": from_number,
                "Url": webhook_url,
                "Method": "POST"
            }
        )
        
        if response.status_code in [200, 201]:
            data = response.json().get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            
            logger.info(f"[Telnyx] Client call initiated: {call_control_id}")
            return {"success": True, "call_control_id": call_control_id}
        else:
            error_msg = _extract_error(response)
            logger.error(f"[Telnyx] Client dial failed: {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.error(f"[Telnyx] Exception: {e}")
        return {"success": False, "error": str(e)}


def generate_agent_answered_texml(session_id: str) -> str:
    """
    TeXML when agent answers: Stream agent audio, join conference.
    Agent audio is NOT sent to Deepgram (recording only).
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    agent_stream_url = f"{stream_url}/ws/telnyx/stream/agent/{session_id}"
    conference_name = f"coachd_{session_id}"
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Connecting you now.</Say>
    <Start>
        <Stream url="{agent_stream_url}" track="inbound_track" />
    </Start>
    <Dial>
        <Conference 
            startConferenceOnEnter="true"
            endConferenceOnExit="true"
            waitUrl=""
            beep="false">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    
    print(f"[TeXML] Agent answered - streaming to: {agent_stream_url}", flush=True)
    return texml


def generate_client_answered_texml(session_id: str) -> str:
    """
    TeXML when client answers: Stream client audio to Deepgram, join conference.
    This is where objection detection happens.
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    client_stream_url = f"{stream_url}/ws/telnyx/stream/client/{session_id}"
    conference_name = f"coachd_{session_id}"
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{client_stream_url}" track="inbound_track" />
    </Start>
    <Dial>
        <Conference 
            startConferenceOnEnter="true"
            endConferenceOnExit="true"
            waitUrl=""
            beep="false">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    
    print(f"[TeXML] Client answered - streaming to: {client_stream_url}", flush=True)
    return texml


def hangup_call(call_control_id: str) -> dict:
    """Hang up a call"""
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    try:
        response = requests.post(
            f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup",
            headers={
                "Authorization": f"Bearer {get_telnyx_api_key()}",
                "Content-Type": "application/json"
            }
        )
        return {"success": response.status_code in [200, 201]}
    except Exception as e:
        logger.error(f"Failed to hangup: {e}")
        return {"success": False, "error": str(e)}


def end_conference(session_id: str) -> dict:
    """End conference (TeXML handles cleanup)"""
    logger.info(f"Ending conference: {session_id}")
    return {"success": True, "session_id": session_id}