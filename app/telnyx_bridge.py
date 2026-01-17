"""
Coachd Telnyx Bridge (TeXML Version)
Handles 3-way conference calls with real-time audio streaming
Uses TeXML API for outbound calls and webhook responses

DUAL-CHANNEL CLICK-TO-CALL ARCHITECTURE:
- Agent answers → Joins conference + agent audio stream starts
- Client answers → Joins SAME conference + client audio stream starts  
- Both are bridged via conference, can hear each other
- Separate audio streams enable speaker-accurate transcription
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
    StatusCallback handles call end events (hangup, no-answer, busy, failed).
    """
    from_number = get_telnyx_phone()
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    
    logger.info(f"[Telnyx] Initiating call to {agent_phone} from {from_number}")
    
    if not is_telnyx_configured():
        logger.error(f"[Telnyx] Not configured")
        return {"success": False, "error": "Telnyx not configured"}
    
    # Status callback to detect hangup, no-answer, busy, failed
    status_callback = f"{settings.base_url}/api/telnyx/call-status?session_id={session_id}&party=agent"
    
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
                "Method": "POST",
                "StatusCallback": status_callback,
                "StatusCallbackEvent": "initiated ringing answered completed",
                "Timeout": 30  # 30 second ring timeout for agent
            }
        )
        
        if response.status_code in [200, 201]:
            response_json = response.json()
            print(f"[Telnyx] Full response: {response_json}", flush=True)
            data = response_json.get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            
            # Also check top level for call_sid (TeXML format)
            if not call_control_id:
                call_control_id = response_json.get("call_sid", "") or response_json.get("sid", "")
            
            print(f"[Telnyx] Extracted call_control_id: {call_control_id}", flush=True)
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


def generate_agent_dtmf_texml(session_id: str) -> str:
    """
    Generate TeXML for agent DTMF gate.
    Called when agent answers the call.
    
    CRITICAL: Stream MUST start here (on call answer) - Telnyx only starts
    streams from direct answer webhook responses, not from Gather action responses.
    
    - Start streaming agent audio immediately (before DTMF)
    - Wait for agent to press 1 before dialing client
    - 30 second timeout for DTMF input (gives agent time to get ready)
    - On success (press 1): redirect to /agent-ready endpoint
    - On timeout/no input: hang up with message
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    agent_stream_url = f"{stream_url}/ws/telnyx/stream/agent/{session_id}"
    dtmf_action_url = f"{settings.base_url}/api/telnyx/agent-ready?session_id={session_id}"
    
    print(f"[TeXML] Generating AGENT DTMF TeXML for session {session_id}", flush=True)
    print(f"[TeXML] Agent stream: {agent_stream_url}", flush=True)
    print(f"[TeXML] DTMF action URL: {dtmf_action_url}", flush=True)
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{agent_stream_url}" track="inbound_track" />
    </Start>
    <Gather action="{dtmf_action_url}" method="POST" numDigits="1" timeout="30">
        <Say voice="Polly.Joanna" language="en-US">Coachd ready. Press 1 to dial your client.</Say>
    </Gather>
    <Say voice="Polly.Joanna" language="en-US">No response received. Goodbye.</Say>
    <Hangup />
</Response>"""
    
    print(f"[TeXML] Agent DTMF TeXML:\n{texml}", flush=True)
    return texml


def generate_agent_conference_texml(session_id: str) -> str:
    """
    Generate TeXML to put the agent into a conference.
    Called AFTER agent presses 1 (DTMF gate passed).
    
    CRITICAL: Must re-start the audio stream here because the previous stream
    from DTMF phase gets killed when <Dial> takes over.
    
    - Re-start streaming agent's audio for transcription
    - Join conference so they can hear/talk to client
    - startConferenceOnEnter="true" - conference starts when agent joins
    - endConferenceOnExit="true" - conference ends if agent hangs up
    - waitUrl points to silence endpoint (no hold music)
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    conference_name = f"coachd_{session_id}"
    silence_url = f"{settings.base_url}/api/telnyx/silence"
    agent_stream_url = f"{stream_url}/ws/telnyx/stream/agent/{session_id}"
    
    print(f"[TeXML] Generating AGENT CONFERENCE TeXML for session {session_id}", flush=True)
    print(f"[TeXML] Conference: {conference_name}", flush=True)
    print(f"[TeXML] Agent stream (re-started): {agent_stream_url}", flush=True)
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna" language="en-US">Calling.</Say>
    <Start>
        <Stream url="{agent_stream_url}" track="inbound_track" />
    </Start>
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
    
    print(f"[TeXML] Agent Conference TeXML:\n{texml}", flush=True)
    return texml


def generate_client_conference_texml(session_id: str) -> str:
    """
    Generate TeXML to put the client into the conference with streaming.
    Called when client answers the call.
    
    - Start streaming client's audio (inbound_track = client speaking)
    - Join same conference as agent
    - startConferenceOnEnter="false" - don't start new conference, join existing
    - endConferenceOnExit="true" - end conference if client hangs up
    - waitUrl points to silence endpoint (no hold music)
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    conference_name = f"coachd_{session_id}"
    silence_url = f"{settings.base_url}/api/telnyx/silence"
    # Use separate stream endpoint for client audio
    client_stream_url = f"{stream_url}/ws/telnyx/stream/client/{session_id}"
    
    print(f"[TeXML] Generating CLIENT TeXML for session {session_id}", flush=True)
    print(f"[TeXML] Conference: {conference_name}", flush=True)
    print(f"[TeXML] Client stream: {client_stream_url}", flush=True)
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{client_stream_url}" track="inbound_track" />
    </Start>
    <Dial>
        <Conference
            startConferenceOnEnter="false"
            endConferenceOnExit="true"
            beep="false"
            waitUrl="{silence_url}">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    
    print(f"[TeXML] Client TeXML:\n{texml}", flush=True)
    return texml


def generate_inbound_texml(session_id: str) -> str:
    """
    Generate TeXML for inbound calls (agent calls the Telnyx number).
    This is for the legacy flow where agent dials in.
    """
    stream_url = settings.base_url.replace("https://", "wss://").replace("http://", "ws://")
    conference_name = f"coachd_{session_id}"
    
    texml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Coachd ready.</Say>
    <Start>
        <Stream url="{stream_url}/ws/telnyx/stream/agent/{session_id}" track="inbound_track" />
    </Start>
    <Dial>
        <Conference 
            startConferenceOnEnter="true"
            endConferenceOnExit="true"
            beep="false">
            {conference_name}
        </Conference>
    </Dial>
</Response>"""
    
    return texml


def add_client_to_conference(client_phone: str, session_id: str, agent_caller_id: str = None) -> dict:
    """
    Call the client and add them to the conference.
    Uses agent's phone number as caller ID so client sees familiar number.
    StatusCallback handles call end events (hangup, no-answer, busy, failed).
    """
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    from_number = agent_caller_id if agent_caller_id else get_telnyx_phone()
    api_key = get_telnyx_api_key()
    app_id = get_telnyx_app_id()
    
    logger.info(f"[Telnyx] Dialing client {client_phone} from {from_number} for session {session_id}")
    
    # Status callback to detect hangup, no-answer, busy, failed
    status_callback = f"{settings.base_url}/api/telnyx/call-status?session_id={session_id}&party=client"
    
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
                "Method": "POST",
                "StatusCallback": status_callback,
                "StatusCallbackEvent": "initiated ringing answered completed",
                "Timeout": 25  # 25 second ring timeout for client
            }
        )
        
        if response.status_code in [200, 201]:
            response_json = response.json()
            print(f"[Telnyx] Client dial response: {response_json}", flush=True)
            data = response_json.get("data", {})
            call_control_id = data.get("call_control_id", "") or data.get("call_sid", "")
            
            # Also check top level for call_sid (TeXML format)
            if not call_control_id:
                call_control_id = response_json.get("call_sid", "") or response_json.get("sid", "")
            
            print(f"[Telnyx] Extracted client call_control_id: {call_control_id}", flush=True)
            logger.info(f"Client call initiated: {call_control_id} for session {session_id}")
            
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
            
            logger.error(f"Telnyx API error dialing client: {response.status_code} - {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.error(f"Failed to dial client: {e}")
        return {"success": False, "error": str(e)}


def hangup_call(call_control_id: str) -> dict:
    """
    Hang up a specific TeXML call.
    TeXML calls use a different API than standard Call Control.
    """
    if not is_telnyx_configured():
        return {"success": False, "error": "Telnyx not configured"}
    
    if not call_control_id:
        logger.warning("No call_control_id provided for hangup")
        return {"success": False, "error": "No call ID"}
    
    app_id = get_telnyx_app_id()
    
    try:
        # TeXML calls use /texml/calls/{call_sid}/update with Status=completed
        response = requests.post(
            f"https://api.telnyx.com/v2/texml/calls/{call_control_id}/update",
            headers={
                "Authorization": f"Bearer {get_telnyx_api_key()}",
                "Content-Type": "application/json"
            },
            json={
                "Status": "completed"
            }
        )
        
        print(f"[Hangup] TeXML hangup {call_control_id}: {response.status_code}", flush=True)
        
        if response.status_code in [200, 201, 204]:
            logger.info(f"Hung up TeXML call: {call_control_id}")
            return {"success": True}
        else:
            # Try fallback to standard Call Control API
            logger.warning(f"TeXML hangup failed ({response.status_code}), trying Call Control API")
            fallback = requests.post(
                f"https://api.telnyx.com/v2/calls/{call_control_id}/actions/hangup",
                headers={
                    "Authorization": f"Bearer {get_telnyx_api_key()}",
                    "Content-Type": "application/json"
                }
            )
            print(f"[Hangup] Fallback hangup {call_control_id}: {fallback.status_code}", flush=True)
            
            if fallback.status_code in [200, 201, 204]:
                return {"success": True}
            else:
                error_detail = ""
                try:
                    error_detail = response.text[:200]
                except:
                    pass
                logger.error(f"Both hangup methods failed for {call_control_id}: {error_detail}")
                return {"success": False, "error": f"Failed to hangup: {response.status_code}"}
            
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