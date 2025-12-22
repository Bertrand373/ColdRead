"""
Coachd Telnyx Bridge (TeXML Version)
Handles 3-way conference calls with real-time audio streaming
Uses TeXML API for outbound calls and webhook responses
"""

import logging
import requests
import base64
import json
import os
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


def get_telnyx_phone() -> str:
    """Get Telnyx phone number directly from environment"""
    phone = os.environ.get("TELNYX_PHONE_NUMBER", "")
    if not phone:
        phone = settings.telnyx_phone_number
    return phone


def get_telnyx_api_key() -> str:
    """Get Telnyx API key directly from environment"""
    key = os.environ.get("TELNYX_API_KEY", "")
    if not key:
        key = settings.telnyx_api_key
    return key


def get_telnyx_app_id() -> str:
    """Get Telnyx App ID directly from environment"""
    app_id = os.environ.get("TELNYX_APP_ID", "")
    if not app_id:
        app_id = settings.telnyx_app_id
    return app_id


def is_telnyx_configured() -> bool:
    """Check if Telnyx is properly configured"""
    return bool(get_telnyx_api_key() and get_telnyx_phone())


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
    from_number = get_telnyx_phone()
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    
    logger.info(f"[Telnyx] Initiating call to {agent_phone} from {from_number}")
    
    if not is_telnyx_configured():
        logger.error(f"[Telnyx] Not configured")
        return {"success": False, "error": "Telnyx not configured"}
    
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
                "Url": f"{settings.base_url}/api/telnyx/agent-answered?session_id={session_id}",
                "Method": "POST"
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
    Generate TeXML for agent call with audio streaming.
    Called when agent answers the call.
    
    Architecture for native 3-way:
    - Agent answers our call
    - We start streaming audio to capture everything
    - Agent uses phone's native 3-way to call client  
    - Client audio comes through mixed on agent's connection
    - We transcribe the mixed audio stream
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    full_stream_url = f"{stream_url}/ws/telnyx/stream/{session_id}"
    
    print(f"[TeXML] Generating for session {session_id}", flush=True)
    print(f"[TeXML] base_url: {settings.base_url}", flush=True)
    print(f"[TeXML] stream_url: {full_stream_url}", flush=True)
    
    # Use Record to keep call alive
    # - maxLength="7200" = 2 hours max
    # - timeout="3600" = 1 hour silence timeout
    # - playBeep="false" = no beep
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{full_stream_url}" track="both_tracks" />
    </Start>
    <Record maxLength="7200" timeout="3600" playBeep="false" />
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
    
    from_number = agent_caller_id if agent_caller_id else get_telnyx_phone()
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    
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
                "Url": f"{settings.base_url}/api/telnyx/client-answered?session_id={session_id}",
                "Method": "POST"
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
                "Authorization": f"Bearer {get_telnyx_api_key()}",
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
                "Authorization": f"Bearer {get_telnyx_api_key()}"
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