"""
Coachd Telnyx Bridge (TeXML Version)
Handles 3-way conference calls with real-time audio streaming
Uses TeXML API for outbound calls and webhook responses
"""

import logging
import requests
import base64
import json
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


def is_telnyx_configured() -> bool:
    """Check if Telnyx is properly configured"""
    return bool(settings.telnyx_api_key and settings.telnyx_phone_number)


def _encode_client_state(data: dict) -> str:
    """Encode client state for Telnyx callbacks"""
    return base64.b64encode(json.dumps(data).encode()).decode()


def _decode_client_state(encoded: str) -> dict:
    """Decode client state from Telnyx callbacks"""
    try:
        return json.loads(base64.b64decode(encoded).decode())
    except Exception:
        return {}


def initiate_agent_call(agent_phone: str, session_id: str) -> dict:
    """
    Call the agent using Telnyx API.
    When agent answers, Telnyx hits our webhook which returns TeXML instructions.
    """
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    try:
        response = requests.post(
            f"https://api.telnyx.com/v2/texml/calls/{settings.telnyx_app_id}",
            headers={
                "Authorization": f"Bearer {settings.telnyx_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "to": agent_phone,
                "from": settings.telnyx_phone_number,
                "url": f"{settings.base_url}/api/telnyx/agent-answered?session_id={session_id}",
                "method": "POST"
            }
        )
        
        if response.status_code in [200, 201]:
            data = response.json().get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            
            logger.info(f"Initiated agent call: {call_control_id} for session {session_id}")
            
            return {
                "success": True,
                "call_control_id": call_control_id,
                "session_id": session_id
            }
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                if "errors" in error_data:
                    error_msg = error_data["errors"][0].get("detail", error_msg)
                elif "error" in error_data:
                    error_msg = error_data["error"]
            except:
                error_msg = f"HTTP {response.status_code}"
            
            logger.error(f"Telnyx API error: {response.status_code} - {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.error(f"Failed to initiate agent call: {e}")
        return {"success": False, "error": str(e)}


def generate_agent_conference_texml(session_id: str) -> str:
    """
    Generate TeXML to put the agent into the conference with streaming.
    Called when agent answers the call.
    """
    conference_name = f"coachd_{session_id}"
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Coachd ready. Dial your client now.</Say>
    <Start>
        <Stream url="{stream_url}/ws/telnyx/stream/{session_id}" track="both_tracks" />
    </Start>
    <Dial>
        <Conference 
            startConferenceOnEnter="true"
            endConferenceOnExit="false"
            record="record-from-start"
            recordingStatusCallback="{settings.base_url}/api/telnyx/recording-complete?session_id={session_id}"
            statusCallback="{settings.base_url}/api/telnyx/conference-status?session_id={session_id}"
            statusCallbackEvent="start end join leave">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    
    return texml


def generate_client_conference_texml(session_id: str) -> str:
    """
    Generate TeXML to put the client into the conference.
    Called when client answers the call.
    """
    conference_name = f"coachd_{session_id}"
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Conference
            startConferenceOnEnter="false"
            endConferenceOnExit="true">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    
    return texml


def generate_inbound_texml(session_id: str) -> str:
    """
    Generate TeXML for inbound calls (agent calls the Telnyx number).
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    conference_name = f"coachd_{session_id}"
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Coachd ready.</Say>
    <Start>
        <Stream url="{stream_url}/ws/telnyx/stream/{session_id}" track="both_tracks" />
    </Start>
    <Dial>
        <Conference 
            startConferenceOnEnter="true"
            endConferenceOnExit="true"
            record="record-from-start">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    
    return texml


def add_client_to_conference(client_phone: str, session_id: str, agent_caller_id: str = None) -> dict:
    """
    Call the client and add them to the conference.
    """
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    from_number = agent_caller_id if agent_caller_id else settings.telnyx_phone_number
    
    try:
        response = requests.post(
            f"https://api.telnyx.com/v2/texml/calls/{settings.telnyx_app_id}",
            headers={
                "Authorization": f"Bearer {settings.telnyx_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "to": client_phone,
                "from": from_number,
                "url": f"{settings.base_url}/api/telnyx/client-answered?session_id={session_id}",
                "method": "POST"
            }
        )
        
        if response.status_code in [200, 201]:
            data = response.json().get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            
            logger.info(f"Added client to conference: {call_control_id}")
            
            return {
                "success": True,
                "call_control_id": call_control_id,
                "client_phone": client_phone
            }
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                if "errors" in error_data:
                    error_msg = error_data["errors"][0].get("detail", error_msg)
            except:
                error_msg = f"HTTP {response.status_code}"
            
            logger.error(f"Telnyx API error: {response.status_code} - {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.error(f"Failed to add client to conference: {e}")
        return {"success": False, "error": str(e)}


def hangup_call(call_control_id: str) -> dict:
    """Hang up a specific call"""
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    try:
        response = requests.post(
            f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup",
            headers={
                "Authorization": f"Bearer {settings.telnyx_api_key}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code in [200, 201]:
            logger.info(f"Hung up call: {call_control_id}")
            return {"success": True}
        else:
            return {"success": False, "error": "Failed to hangup"}
            
    except Exception as e:
        logger.error(f"Failed to hangup call: {e}")
        return {"success": False, "error": str(e)}


def end_conference(session_id: str) -> dict:
    """
    End the conference (simplified - TeXML handles cleanup automatically)
    """
    logger.info(f"Ending conference for session: {session_id}")
    return {"success": True, "session_id": session_id}


def get_recording_url(recording_id: str) -> Optional[str]:
    """Get the URL for a recording"""
    if not is_telnyx_configured() or not recording_id:
        return None
    
    try:
        response = requests.get(
            f"https://api.telnyx.com/v2/recordings/{recording_id}",
            headers={
                "Authorization": f"Bearer {settings.telnyx_api_key}"
            }
        )
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            return data.get("download_urls", {}).get("mp3")
        return None
        
    except Exception as e:
        logger.error(f"Failed to get recording URL: {e}")
        return None


# Legacy function names for compatibility
def join_conference(call_control_id: str, session_id: str) -> dict:
    """Legacy - TeXML handles this via webhook responses"""
    return {"success": True, "message": "Handled by TeXML webhooks"}