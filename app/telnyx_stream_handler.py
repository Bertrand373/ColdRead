"""
Coachd Telnyx Stream Handler
Receives audio from Telnyx media streams, processes through Deepgram with diarization,
and broadcasts transcripts + guidance to the frontend.

Speaker Identification Strategy:
- Both speakers start as "unknown" - transcripts shown but no guidance
- Listen for agent introduction: "this is [name] with Globe Life/Liberty National"
- When intro detected → THAT speaker = Agent, the OTHER = Client
- Only trigger guidance on CLIENT speech AFTER roles are locked
- This handles edge cases where either party speaks first
"""

import asyncio
import base64
import json
import time
import audioop
from typing import Optional, Callable, Dict, Set, List
from fastapi import WebSocket

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from .config import settings
from .rag_engine import get_rag_engine, CallContext
from .call_state_machine import CallStateMachine
from .call_session import session_manager
from .usage_tracker import log_deepgram_usage

import logging
logger = logging.getLogger(__name__)


class TelnyxStreamHandler:
    """
    Handles Telnyx media stream WebSocket connections.
    Converts μ-law audio to PCM, sends to Deepgram with diarization,
    and broadcasts results to the session's frontend clients.
    
    Speaker Identification Strategy:
    - Both speakers start as "unknown"
    - Listen for agent introduction: "this is [name] with Globe/Liberty"
    - When detected, THAT speaker = Agent, the OTHER = Client
    - Only trigger guidance AFTER roles are locked in
    """
    
    # Deepgram configuration
    SAMPLE_RATE = 8000  # Telnyx sends 8kHz audio
    DEEPGRAM_SAMPLE_RATE = 8000  # Match Telnyx rate
    
    # Speaker identification
    SPEAKER_CLIENT = "client"
    SPEAKER_AGENT = "agent"
    SPEAKER_UNKNOWN = "unknown"
    
    # Agent introduction patterns (Globe Life specific)
    AGENT_INTRO_PATTERNS = [
        "this is",  # "this is [name] with Globe Life"
        "my name is",  # "my name is [name] and"
        "i'm calling from",  # "I'm calling from Globe Life"
        "with globe life",
        "with liberty national",
        "your insurance company",
    ]
    
    # Guidance configuration
    GUIDANCE_COOLDOWN_SECONDS = 3.0
    MIN_WORDS_FOR_GUIDANCE = 8
    GUIDANCE_TIMEOUT_SECONDS = 8.0
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self._loop = None
        
        # Speaker tracking - wait for agent introduction before locking roles
        self._speaker_map: Dict[int, str] = {}  # Maps Deepgram speaker ID -> role
        self._roles_locked = False  # True once we've identified agent
        self._agent_speaker_id: Optional[int] = None
        self._pending_transcripts: List[dict] = []  # Buffer until roles locked
        
        # Transcript buffer per speaker
        self._client_buffer = ""
        self._agent_buffer = ""
        self._full_transcript = []
        
        # Guidance state
        self._last_guidance_time = 0
        self._generating_guidance = False
        self._rag_engine = None
        self._state_machine = None
        
        # Usage tracking
        self._total_audio_bytes = 0
        self._session_start_time = None
        self._agency = None
        
        logger.info(f"[TelnyxStream] Handler created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize Deepgram connection with diarization"""
        logger.info(f"[TelnyxStream] Starting for session {self.session_id}")
        
        self._loop = asyncio.get_running_loop()
        self._session_start_time = time.time()
        
        # Get session info for agency context
        session = await session_manager.get_session(self.session_id)
        if session:
            self._agency = getattr(session, 'agency', None)
        
        # Initialize RAG engine (optional - continue without it)
        try:
            self._rag_engine = get_rag_engine()
        except Exception as e:
            logger.warning(f"[TelnyxStream] RAG engine not available: {e}")
            self._rag_engine = None
        
        # Initialize state machine (optional - continue without it)
        try:
            self._state_machine = CallStateMachine(session_id=self.session_id)
        except Exception as e:
            logger.warning(f"[TelnyxStream] State machine not available: {e}")
            self._state_machine = None
        
        # Initialize Deepgram (optional - continue without it for basic call flow)
        if not settings.deepgram_api_key:
            logger.warning("[TelnyxStream] No Deepgram API key - transcription disabled")
            self.is_running = True
            await self._broadcast_to_frontend({
                "type": "ready",
                "message": "Call connected (transcription unavailable)"
            })
            return True
        
        try:
            self.deepgram = DeepgramClient(settings.deepgram_api_key)
            
            # Use ASYNC live client for async context
            self.connection = self.deepgram.listen.asynclive.v("1")
            print(f"[TelnyxStream] Created Deepgram ASYNC client", flush=True)
            
            # Register event handlers
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            # Configure with diarization enabled
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                interim_results=True,
                utterance_end_ms=1000,
                # DIARIZATION - identify different speakers
                diarize=True,
                # Audio format from Telnyx (after μ-law to PCM conversion)
                encoding="linear16",
                sample_rate=self.DEEPGRAM_SAMPLE_RATE,
                channels=1
            )
            
            print(f"[TelnyxStream] Starting Deepgram async connection...", flush=True)
            
            # Async start
            await self.connection.start(options)
            print(f"[TelnyxStream] Deepgram async connection started", flush=True)
            
            self.is_running = True
            logger.info(f"[TelnyxStream] Deepgram connected with diarization")
            
            # Notify frontend that we're ready
            await self._broadcast_to_frontend({
                "type": "ready",
                "message": "Live transcription active"
            })
            
            return True
                
        except Exception as e:
            logger.error(f"[TelnyxStream] Error starting Deepgram: {e} - continuing without transcription")
            print(f"[TelnyxStream] Deepgram error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.deepgram = None
            self.connection = None
            self.is_running = True
            await self._broadcast_to_frontend({
                "type": "ready",
                "message": "Call connected (transcription unavailable)"
            })
            return True
    
    async def handle_telnyx_message(self, message: dict):
        """
        Process incoming Telnyx WebSocket message.
        Telnyx sends JSON with base64-encoded μ-law audio.
        """
        event = message.get("event")
        
        if event == "connected":
            logger.info(f"[TelnyxStream] Telnyx stream connected")
            print(f"[TelnyxStream] Telnyx stream connected - full message: {message}", flush=True)
            
        elif event == "start":
            # Stream starting - contains metadata
            stream_sid = message.get("stream_sid")
            logger.info(f"[TelnyxStream] Stream started: {stream_sid}")
            print(f"[TelnyxStream] Stream started - full message: {message}", flush=True)
            
        elif event == "media":
            # Audio data
            media = message.get("media", {})
            payload = media.get("payload")
            
            if payload:
                try:
                    # Decode base64 μ-law audio
                    ulaw_audio = base64.b64decode(payload)
                    self._total_audio_bytes += len(ulaw_audio)
                    
                    # Convert μ-law to linear PCM (16-bit)
                    pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
                    
                    # Send to Deepgram if connected (async)
                    if self.connection and self.is_running:
                        await self.connection.send(pcm_audio)
                        
                        # Log occasionally
                        if self._total_audio_bytes % 16000 < 200:  # ~every second
                            print(f"[TelnyxStream] Sent to Deepgram: {self._total_audio_bytes} total bytes", flush=True)
                    else:
                        # Got audio but no Deepgram connection
                        if self._total_audio_bytes % 32000 < 400:
                            print(f"[TelnyxStream] Audio received (no Deepgram): {self._total_audio_bytes} bytes", flush=True)
                    
                except Exception as e:
                    logger.error(f"[TelnyxStream] Error processing audio: {e}")
                    print(f"[TelnyxStream] Audio error: {e}", flush=True)
                    
        elif event == "stop":
            logger.info(f"[TelnyxStream] Stream stopped")
            print(f"[TelnyxStream] Stream stopped - full message: {message}", flush=True)
            await self.stop()
        else:
            # Unknown event
            print(f"[TelnyxStream] Unknown event: {event} - message: {message}", flush=True)
    
    def _on_open(self, *args, **kwargs):
        """Deepgram connection opened"""
        logger.info(f"[TelnyxStream] Deepgram connection open")
        print(f"[TelnyxStream] Deepgram WebSocket OPEN - ready for audio", flush=True)
    
    def _on_transcript(self, *args, **kwargs):
        """Handle transcript from Deepgram (called from Deepgram's thread)"""
        try:
            result = kwargs.get('result') or (args[1] if len(args) > 1 else None)
            if not result:
                return
            
            channel = result.channel
            alternatives = channel.alternatives
            
            if not alternatives:
                return
            
            alt = alternatives[0]
            transcript = alt.transcript
            is_final = result.is_final
            
            if not transcript:
                return
            
            # Log transcript
            print(f"[TelnyxStream] TRANSCRIPT ({'FINAL' if is_final else 'interim'}): {transcript[:50]}...", flush=True)
            
            # Get speaker from diarization
            words = alt.words if hasattr(alt, 'words') else []
            speaker_id = None
            
            if words:
                # Get speaker from first word with speaker info
                for word in words:
                    if hasattr(word, 'speaker') and word.speaker is not None:
                        speaker_id = word.speaker
                        break
            
            # Identify speaker - pass transcript for intro detection
            speaker_role = self._identify_speaker(speaker_id, transcript if is_final else "")
            
            # If roles are locked and this is a different speaker, get their role
            if self._roles_locked and speaker_id is not None:
                speaker_role = self._get_other_speaker_role(speaker_id)
            
            # Schedule async broadcast (thread-safe)
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._process_transcript(transcript, is_final, speaker_role, speaker_id),
                    self._loop
                )
                
        except Exception as e:
            logger.error(f"[TelnyxStream] Error in transcript handler: {e}")
            print(f"[TelnyxStream] Transcript error: {e}", flush=True)
    
    def _identify_speaker(self, speaker_id: Optional[int], transcript: str = "") -> str:
        """
        Identify speaker role based on agent introduction detection.
        
        Strategy:
        - Both speakers start as "unknown"
        - When we detect agent intro phrase, THAT speaker becomes Agent
        - The other speaker ID becomes Client
        - Only return definitive roles after locking
        """
        if speaker_id is None:
            return self.SPEAKER_UNKNOWN
        
        # If roles already locked, use the map
        if self._roles_locked:
            return self._speaker_map.get(speaker_id, self.SPEAKER_UNKNOWN)
        
        # Roles not locked yet - check if this transcript contains agent introduction
        if transcript and self._check_for_agent_intro(transcript):
            # This speaker is the agent!
            self._agent_speaker_id = speaker_id
            self._speaker_map[speaker_id] = self.SPEAKER_AGENT
            self._roles_locked = True
            
            logger.info(f"[TelnyxStream] AGENT IDENTIFIED! Speaker {speaker_id} introduced themselves")
            logger.info(f"[TelnyxStream] Roles locked - guidance now active")
            
            # Notify frontend that we've identified the agent
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_to_frontend({
                        "type": "speakers_identified",
                        "message": "Agent identified - live coaching active"
                    }),
                    self._loop
                )
            
            return self.SPEAKER_AGENT
        
        # Not locked yet, return unknown
        return self.SPEAKER_UNKNOWN
    
    def _check_for_agent_intro(self, transcript: str) -> bool:
        """
        Check if transcript contains agent introduction patterns.
        Globe Life agents always introduce themselves:
        - "this is [name] with Globe Life/Liberty National"
        - "my name is [name]"
        """
        text_lower = transcript.lower()
        
        # Must contain at least one intro pattern
        has_intro = False
        for pattern in self.AGENT_INTRO_PATTERNS:
            if pattern in text_lower:
                has_intro = True
                break
        
        if not has_intro:
            return False
        
        # Additional validation: should mention company or have "this is" + "with"
        is_company_intro = (
            "globe life" in text_lower or
            "liberty national" in text_lower or
            "insurance company" in text_lower or
            ("this is" in text_lower and "with" in text_lower) or
            ("my name is" in text_lower)
        )
        
        return is_company_intro
    
    def _get_other_speaker_role(self, speaker_id: int) -> str:
        """
        Once agent is identified, any OTHER speaker is the client.
        """
        if not self._roles_locked:
            return self.SPEAKER_UNKNOWN
        
        if speaker_id == self._agent_speaker_id:
            return self.SPEAKER_AGENT
        
        # Different speaker ID = Client
        if speaker_id not in self._speaker_map:
            self._speaker_map[speaker_id] = self.SPEAKER_CLIENT
            logger.info(f"[TelnyxStream] Speaker {speaker_id} identified as CLIENT")
        
        return self._speaker_map[speaker_id]
    
    async def _process_transcript(self, transcript: str, is_final: bool, speaker: str, speaker_id: Optional[int] = None):
        """Process transcript and trigger guidance if needed"""
        
        # Broadcast transcript to frontend (always, even before roles locked)
        await self._broadcast_to_frontend({
            "type": "transcript",
            "text": transcript,
            "is_final": is_final,
            "speaker": speaker,
            "roles_locked": self._roles_locked
        })
        
        # Also add to session for history
        await session_manager.add_transcript(
            self.session_id, 
            transcript, 
            speaker=speaker, 
            is_final=is_final
        )
        
        # Buffer transcripts
        if is_final:
            self._full_transcript.append({
                "speaker": speaker,
                "speaker_id": speaker_id,
                "text": transcript,
                "timestamp": time.time()
            })
            
            # Only process for guidance if roles are locked
            if self._roles_locked:
                if speaker == self.SPEAKER_CLIENT:
                    self._client_buffer += " " + transcript
                    
                    # Feed to state machine
                    if self._state_machine:
                        self._state_machine.add_transcript(transcript, is_final=True)
                    
                    # Check for guidance triggers on CLIENT speech only
                    await self._check_guidance_trigger(transcript)
                    
                elif speaker == self.SPEAKER_AGENT:
                    self._agent_buffer += " " + transcript
    
    async def _check_guidance_trigger(self, transcript: str):
        """Check if we should generate guidance based on client speech"""
        
        if self._generating_guidance:
            return
        
        now = time.time()
        if now - self._last_guidance_time < self.GUIDANCE_COOLDOWN_SECONDS:
            return
        
        # Check for hot trigger keywords (objections)
        hot_triggers = [
            "can't afford", "too expensive", "not interested", "no money",
            "think about it", "talk to my", "spouse", "wife", "husband",
            "call back", "busy", "not a good time", "don't need",
            "already have", "too much", "let me think", "send information",
            "how much", "what's the cost", "what's the price"
        ]
        
        transcript_lower = transcript.lower()
        triggered = False
        trigger_phrase = None
        
        for trigger in hot_triggers:
            if trigger in transcript_lower:
                triggered = True
                trigger_phrase = trigger
                break
        
        # Also trigger on sufficient word count
        word_count = len(self._client_buffer.split())
        if not triggered and word_count >= self.MIN_WORDS_FOR_GUIDANCE:
            triggered = True
        
        if triggered:
            logger.info(f"[TelnyxStream] Guidance triggered by: {trigger_phrase or 'word count'}")
            await self._generate_guidance(transcript, trigger_phrase)
    
    async def _generate_guidance(self, trigger_text: str, trigger_phrase: Optional[str] = None):
        """Generate AI guidance based on conversation context"""
        
        if not self._rag_engine:
            return
        
        self._generating_guidance = True
        self._last_guidance_time = time.time()
        
        try:
            # Build context
            context = CallContext(
                call_type="presentation",  # Default, could be passed from session
                product="life insurance",
                recent_transcript=self._client_buffer[-500:],  # Last 500 chars
                client_profile={}
            )
            
            # Notify frontend that guidance is starting
            await self._broadcast_to_frontend({
                "type": "guidance_start",
                "trigger": trigger_phrase or "conversation"
            })
            
            # Generate guidance with streaming
            guidance_text = ""
            
            async for chunk in self._rag_engine.get_guidance_stream(
                trigger_text,
                context,
                agency=self._agency
            ):
                guidance_text += chunk
                await self._broadcast_to_frontend({
                    "type": "guidance_chunk",
                    "text": chunk
                })
            
            # Send completion
            await self._broadcast_to_frontend({
                "type": "guidance_complete",
                "full_text": guidance_text
            })
            
            # Add to session history
            await session_manager.add_guidance(self.session_id, {
                "trigger": trigger_phrase or trigger_text[:50],
                "response": guidance_text
            })
            
            logger.info(f"[TelnyxStream] Guidance generated: {len(guidance_text)} chars")
            
        except asyncio.TimeoutError:
            logger.warning(f"[TelnyxStream] Guidance generation timed out")
            await self._broadcast_to_frontend({
                "type": "guidance_error",
                "message": "Guidance timed out"
            })
        except Exception as e:
            logger.error(f"[TelnyxStream] Error generating guidance: {e}")
        finally:
            self._generating_guidance = False
    
    async def _broadcast_to_frontend(self, message: dict):
        """Send message to all frontend WebSocket clients for this session"""
        await session_manager._broadcast_to_session(self.session_id, message)
    
    def _on_error(self, *args, **kwargs):
        """Handle Deepgram error"""
        error = kwargs.get('error') or (args[1] if len(args) > 1 else "Unknown error")
        logger.error(f"[TelnyxStream] Deepgram error: {error}")
        print(f"[TelnyxStream] Deepgram ERROR: {error}", flush=True)
        # Don't set is_running to False - let audio continue flowing
        # The error might be recoverable
    
    def _on_close(self, *args, **kwargs):
        """Handle Deepgram connection close"""
        logger.info(f"[TelnyxStream] Deepgram connection closed")
        print(f"[TelnyxStream] Deepgram WebSocket CLOSED", flush=True)
        # Only stop if we initiated the close
        # self.is_running = False
    
    async def stop(self):
        """Stop the stream handler and log usage"""
        logger.info(f"[TelnyxStream] Stopping session {self.session_id}")
        print(f"[TelnyxStream] Stopping session {self.session_id}", flush=True)
        self.is_running = False
        
        # Log Deepgram usage
        if self._total_audio_bytes > 0 and self._session_start_time:
            # Calculate duration (8kHz, 16-bit mono = 16000 bytes/sec)
            bytes_per_second = self.SAMPLE_RATE * 2  # 2 bytes per sample
            duration_seconds = self._total_audio_bytes / bytes_per_second
            
            cost = log_deepgram_usage(
                duration_seconds=duration_seconds,
                agency_code=self._agency,
                session_id=self.session_id,
                model='nova-2'
            )
            logger.info(f"[TelnyxStream] Logged usage: {duration_seconds:.1f}s = ${cost:.4f}")
            print(f"[TelnyxStream] Logged usage: {duration_seconds:.1f}s = ${cost:.4f}", flush=True)
        
        # Close Deepgram connection (async)
        if self.connection:
            try:
                await asyncio.wait_for(
                    self.connection.finish(),
                    timeout=3.0
                )
                print(f"[TelnyxStream] Deepgram connection closed", flush=True)
            except Exception as e:
                logger.error(f"[TelnyxStream] Error closing Deepgram: {e}")
            finally:
                self.connection = None
        
        # Notify frontend
        await self._broadcast_to_frontend({
            "type": "stream_ended"
        })


# Active stream handlers
_active_handlers: Dict[str, TelnyxStreamHandler] = {}


async def get_or_create_handler(session_id: str) -> TelnyxStreamHandler:
    """Get existing handler or create new one for session"""
    if session_id not in _active_handlers:
        handler = TelnyxStreamHandler(session_id)
        if await handler.start():
            _active_handlers[session_id] = handler
        else:
            raise Exception("Failed to start stream handler")
    return _active_handlers[session_id]


async def remove_handler(session_id: str):
    """Remove and stop handler for session"""
    if session_id in _active_handlers:
        handler = _active_handlers.pop(session_id)
        await handler.stop()