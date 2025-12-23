"""
Coachd Real-Time Engine
Deepgram streaming + live AI guidance

Production-ready with:
- Thread-safe async callbacks
- Timeout on AI guidance generation
- Proper error handling
- OPTIMIZED: Reduced latency for real-time feel
- BILLING: Complete usage tracking for Deepgram and Claude
- V3: Phrase-based objection detection with buying signal blocking
- V4: Globe Life locked scripts with Sonnet contextualization
  - Scripts are verbatim from Globe Life training docs
  - Sonnet fills in context slots (names, family, details from transcript)
  - Never changes script structure - just personalizes
  - Makes new agents sound like senior agents
"""

import asyncio
import time
import sys
import uuid
from typing import Optional, Callable
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from .config import settings
from .rag_engine import get_rag_engine, CallContext
from .call_state_machine import CallStateMachine
from .usage_tracker import log_deepgram_usage
from .objection_detector import detect_objection, has_any_trigger, DetectionResult


class RealtimeTranscriber:
    """Handles real-time audio transcription via Deepgram"""
    
    # ============ LATENCY-OPTIMIZED CONFIGURATION ============
    GUIDANCE_TIMEOUT_SECONDS = 8.0   # Max time for AI guidance generation
    GUIDANCE_COOLDOWN_SECONDS = 3.0  # Cooldown for word-count triggers
    MIN_WORDS_FOR_GUIDANCE = 12      # Minimum words before non-trigger guidance
    
    # Objection cooldown - don't re-trigger same objection type within this window
    OBJECTION_COOLDOWN_SECONDS = 10.0
    
    # Agent identification window - look for intro in first N seconds
    AGENT_ID_WINDOW_SECONDS = 90.0
    
    # Audio format constants for duration calculation
    SAMPLE_RATE = 16000  # 16kHz
    BYTES_PER_SAMPLE = 2  # 16-bit linear PCM = 2 bytes per sample
    CHANNELS = 1  # Mono
    
    # Agent intro patterns - if speaker says these, they're the agent
    AGENT_INTRO_PATTERNS = [
        "globe life",
        "liberty national",
        "your insurance company",
        "this is calling from",
        "calling about your policy",
        "calling about your coverage",
        "home office asked",
        "home office has asked",
        "with globe life",
        "with liberty national",
    ]
    
    # Agent quote patterns - if objection contains these, agent is quoting client
    AGENT_QUOTE_PATTERNS = [
        "you're saying",
        "you are saying",
        "i hear that you",
        "you mentioned",
        "so if you",
        "what you're telling me",
        "sounds like you",
        "it sounds like you",
        "you said",
        "you feel like",
        "you think that",
        "i understand you",
        "i understand that you",
    ]
    
    def __init__(self, on_transcript: Callable, on_guidance: Callable):
        self.on_transcript = on_transcript
        self.on_guidance = on_guidance
        self.connection = None
        self.transcript_buffer = ""
        self.last_guidance_time = 0
        self.call_context = CallContext()  # Legacy fallback
        self.state_machine = None  # V2: Full state tracking
        self.rag_engine = None
        self.is_running = False
        self._loop = None
        self._generating_guidance = False  # Prevent concurrent guidance generation
        self._agency = None  # Track agency for RAG context and billing
        self._call_type = "presentation"  # Track call type for objection routing
        
        # Duplicate prevention: track interim triggers to skip matching finals
        self._pending_interim_text = None
        self._pending_interim_time = 0
        
        # V4: Track handled objections to prevent rapid re-triggers
        self._last_objection_type = None
        self._last_objection_time = 0
        
        # V4: Speaker identification
        self._agent_speaker_id = None  # None until identified
        self._call_start_time = None   # Track when call started
        
        # ============ USAGE TRACKING ============
        self._session_id = str(uuid.uuid4())  # Unique session ID for tracking
        self._session_start_time = None  # Track when session started
        self._total_audio_bytes = 0  # Track total audio bytes received
        self._audio_duration_seconds = 0.0  # Calculated audio duration
        
        print(f"[RT] RealtimeTranscriber initialized (session={self._session_id[:8]}, cooldown={self.GUIDANCE_COOLDOWN_SECONDS}s, min_words={self.MIN_WORDS_FOR_GUIDANCE})", flush=True)
        
        # Initialize Deepgram client
        if settings.deepgram_api_key:
            try:
                self.deepgram = DeepgramClient(settings.deepgram_api_key)
                print(f"[RT] Deepgram client created", flush=True)
            except Exception as e:
                print(f"[RT] ERROR creating Deepgram client: {e}", flush=True)
                self.deepgram = None
        else:
            self.deepgram = None
            print("[RT] WARNING: DEEPGRAM_API_KEY not set - transcription disabled", flush=True)
    
    @property
    def session_id(self) -> str:
        """Get the session ID for this transcription session"""
        return self._session_id
        
    async def start(self) -> bool:
        """Start the Deepgram live transcription"""
        print(f"[RT] start() called", flush=True)
        
        if not self.deepgram:
            print("[RT] ERROR: Cannot start - no Deepgram client", flush=True)
            await self.on_transcript({
                "type": "error",
                "message": "Deepgram API key not configured"
            })
            return False
        
        # CRITICAL: Capture the event loop for thread-safe callbacks
        self._loop = asyncio.get_running_loop()
        print(f"[RT] Event loop captured", flush=True)
        
        # Reset usage tracking for new session
        self._session_start_time = time.time()
        self._total_audio_bytes = 0
        self._audio_duration_seconds = 0.0
            
        # Initialize RAG engine (non-critical - can run without it)
        try:
            self.rag_engine = get_rag_engine()
            print(f"[RT] RAG engine initialized", flush=True)
        except Exception as e:
            print(f"[RT] WARNING: RAG engine not available: {e}", flush=True)
            self.rag_engine = None
        
        try:
            print(f"[RT] Creating Deepgram live connection...", flush=True)
            
            # Create live transcription connection
            self.connection = self.deepgram.listen.live.v("1")
            print(f"[RT] Connection object created: {type(self.connection)}", flush=True)
            
            # Set up event handlers
            self.connection.on(LiveTranscriptionEvents.Transcript, self._handle_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self._handle_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._handle_close)
            print(f"[RT] Event handlers registered", flush=True)
            
            # Configure live transcription options
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                interim_results=True,
                endpointing=300,
                sample_rate=self.SAMPLE_RATE,
                encoding="linear16",
                channels=self.CHANNELS,
                diarize=True,  # V4: Enable speaker diarization
            )
            print(f"[RT] Options configured (diarize=True), starting connection...", flush=True)
            
            # Track call start time for agent identification window
            self._call_start_time = time.time()
            
            # Start the connection in an executor to prevent blocking
            try:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.connection.start(options)
                    ),
                    timeout=8.0
                )
                print(f"[RT] connection.start() returned: {result}", flush=True)
            except asyncio.TimeoutError:
                print(f"[RT] ERROR: connection.start() timed out after 8 seconds", flush=True)
                await self.on_transcript({
                    "type": "error",
                    "message": "Deepgram connection timed out"
                })
                return False
            
            # Give it a moment to fully establish
            await asyncio.sleep(0.3)
            
            if self.connection:
                self.is_running = True
                print(f"[RT] SUCCESS: Deepgram connection started", flush=True)
                return True
            else:
                print(f"[RT] ERROR: Connection is None after start", flush=True)
                await self.on_transcript({
                    "type": "error",
                    "message": "Failed to connect to transcription service"
                })
                return False
                
        except Exception as e:
            print(f"[RT] ERROR: Deepgram connection error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            await self.on_transcript({
                "type": "error",
                "message": f"Transcription service error: {str(e)}"
            })
            return False
        
    async def send_audio(self, audio_data: bytes):
        """Send audio chunk to Deepgram (non-blocking)"""
        if not self.connection or not self.is_running:
            return
        
        # Track audio bytes for usage calculation
        self._total_audio_bytes += len(audio_data)
            
        try:
            # Run in executor to prevent blocking the event loop
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, self.connection.send, audio_data),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            print(f"[RT] WARNING: send_audio timed out, stopping", flush=True)
            self.is_running = False
        except Exception as e:
            print(f"[RT] Error sending audio: {e}", flush=True)
            self.is_running = False
            
    async def stop(self):
        """Stop the transcription and log usage"""
        print(f"[RT] stop() called", flush=True)
        self.is_running = False
        
        # Calculate and log Deepgram usage
        self._log_session_usage()
        
        if self.connection:
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.connection.finish),
                    timeout=3.0
                )
                print(f"[RT] connection.finish() completed", flush=True)
            except asyncio.TimeoutError:
                print(f"[RT] WARNING: connection.finish() timed out", flush=True)
            except Exception as e:
                print(f"[RT] Error finishing connection: {e}", flush=True)
            finally:
                self.connection = None
    
    def _log_session_usage(self):
        """Calculate and log Deepgram usage for this session"""
        if self._total_audio_bytes == 0:
            print(f"[RT] No audio data to log for session {self._session_id[:8]}", flush=True)
            return
        
        # Calculate audio duration from bytes
        bytes_per_second = self.SAMPLE_RATE * self.BYTES_PER_SAMPLE * self.CHANNELS
        self._audio_duration_seconds = self._total_audio_bytes / bytes_per_second
        
        # Also calculate wall-clock duration for comparison
        wall_clock_seconds = 0.0
        if self._session_start_time:
            wall_clock_seconds = time.time() - self._session_start_time
        
        print(f"[RT] Session {self._session_id[:8]} usage: "
              f"audio={self._audio_duration_seconds:.1f}s ({self._total_audio_bytes:,} bytes), "
              f"wall={wall_clock_seconds:.1f}s, "
              f"agency={self._agency}", flush=True)
        
        # Log to database for billing
        cost = log_deepgram_usage(
            duration_seconds=self._audio_duration_seconds,
            agency_code=self._agency,
            session_id=self._session_id,
            model='nova-2'
        )
        
        print(f"[RT] Logged Deepgram usage: {self._audio_duration_seconds:.1f}s = ${cost:.4f}", flush=True)
    
    def _schedule_async(self, coro):
        """Schedule a coroutine to run in the main event loop (thread-safe)"""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            print(f"[RT] WARNING: Cannot schedule async - no running loop", flush=True)
    
    def _handle_transcript(self, *args, **kwargs):
        """Handle incoming transcripts from Deepgram (called from Deepgram's thread)"""
        try:
            result = kwargs.get('result') or (args[1] if len(args) > 1 else None)
            
            if result:
                alternatives = result.channel.alternatives
                if alternatives:
                    transcript = alternatives[0].transcript
                    is_final = result.is_final
                    
                    # V4: Extract speaker ID if diarization is enabled
                    speaker_id = None
                    words = alternatives[0].words if hasattr(alternatives[0], 'words') else []
                    if words and len(words) > 0 and hasattr(words[0], 'speaker'):
                        speaker_id = words[0].speaker
                    
                    if transcript:
                        # Send transcript to client (thread-safe)
                        self._schedule_async(self.on_transcript({
                            "type": "transcript",
                            "text": transcript,
                            "is_final": is_final,
                            "speaker": speaker_id
                        }))
                        
                        # V4: Speaker identification and filtering
                        if not self._process_speaker(transcript, speaker_id):
                            # This is agent speech - skip objection detection
                            # But still buffer for context
                            if is_final:
                                self.transcript_buffer += " " + transcript
                                if self.state_machine:
                                    self.state_machine.add_transcript(transcript, is_final=True)
                            return
                        
                        # Check for trigger keywords on BOTH interim AND final
                        # This lets us react MID-SENTENCE to objections
                        self._check_for_guidance_trigger(transcript, is_final)
                        
                        # Buffer final transcripts for context
                        if is_final:
                            self.transcript_buffer += " " + transcript
                            # Feed to state machine for full tracking
                            if self.state_machine:
                                self.state_machine.add_transcript(transcript, is_final=True)
                                
        except Exception as e:
            print(f"[RT] Error handling transcript: {e}", flush=True)
    
    def _process_speaker(self, transcript: str, speaker_id: int) -> bool:
        """
        Process speaker identification.
        Returns True if this is CLIENT speech (should check for objections).
        Returns False if this is AGENT speech (should ignore for objections).
        
        If diarization doesn't return speaker IDs, falls back to pattern matching only.
        """
        text_lower = transcript.lower()
        now = time.time()
        
        # If we haven't identified the agent yet, look for intro patterns
        if self._agent_speaker_id is None and self._call_start_time:
            time_since_start = now - self._call_start_time
            
            # Only look for agent intro in the first 90 seconds
            if time_since_start <= self.AGENT_ID_WINDOW_SECONDS:
                for pattern in self.AGENT_INTRO_PATTERNS:
                    if pattern in text_lower:
                        # Only set agent ID if we have a valid speaker_id
                        if speaker_id is not None:
                            self._agent_speaker_id = speaker_id
                            print(f"[RT] 🎤 AGENT IDENTIFIED: Speaker {speaker_id} (said '{pattern}')", flush=True)
                        else:
                            print(f"[RT] 🎤 Agent intro detected but no speaker ID available", flush=True)
                        return False  # This is agent speech regardless
        
        # If we've identified the agent AND have a speaker_id, filter their speech
        if self._agent_speaker_id is not None and speaker_id is not None:
            if speaker_id == self._agent_speaker_id:
                return False  # Agent speech - ignore for objections
        
        # This is client speech (or we can't determine speaker)
        # The AGENT_QUOTE_PATTERNS in _check_for_guidance_trigger act as backup filter
        return True
    
    def _is_agent_quoting(self, transcript: str) -> bool:
        """Check if this transcript contains agent quoting the client."""
        text_lower = transcript.lower()
        for pattern in self.AGENT_QUOTE_PATTERNS:
            if pattern in text_lower:
                return True
        return False
    
    def _check_for_guidance_trigger(self, latest_transcript: str, is_final: bool = True):
        """Check if we should trigger guidance generation using phrase-based detection.
        
        V4: 
        - Objections only (no word-count fallback)
        - Filters out agent quotes
        - Prevents rapid re-triggers of same objection type
        """
        # Don't generate if already generating
        if self._generating_guidance:
            return
        
        now = time.time()
        
        # Quick check first - if no potential triggers, stay silent
        if not has_any_trigger(latest_transcript):
            return
        
        # V4: Check if agent is quoting the client (backup filter)
        if self._is_agent_quoting(latest_transcript):
            print(f"[RT] 🔇 Agent quote detected - ignoring: '{latest_transcript[:40]}...'", flush=True)
            return
        
        # Run full phrase detection
        detection = detect_objection(latest_transcript, self._call_type)
        
        # If buying signal detected, DO NOT fire objection handling
        if detection.is_buying_signal:
            print(f"[RT] ✅ Buying signal detected - staying silent", flush=True)
            return
        
        # If no objection detected, stay silent
        if not detection.detected:
            return
        
        # V4: OBJECTION TYPE COOLDOWN - Don't re-trigger same objection type
        if self._last_objection_type == detection.objection_type:
            time_since_last = now - self._last_objection_time
            if time_since_last < self.OBJECTION_COOLDOWN_SECONDS:
                print(f"[RT] ⏳ Same objection type '{detection.objection_type}' within {time_since_last:.1f}s - skipping", flush=True)
                return
        
        # Basic time cooldown (any objection)
        if now - self.last_guidance_time < 2.0:
            return
        
        print(f"[RT] 🎯 OBJECTION DETECTED: {detection.objection_type} - '{latest_transcript[:50]}...'", flush=True)
        
        # Track this objection type
        self._last_objection_type = detection.objection_type
        self._last_objection_time = now
        
        # Update transcript buffer with triggering text
        if latest_transcript.strip():
            self.transcript_buffer = latest_transcript
        
        # Set timestamp immediately to prevent race condition
        self.last_guidance_time = now
        
        # Schedule guidance generation with detection result
        self._schedule_async(self._generate_guidance_v3(detection, latest_transcript))
                   
    def _handle_error(self, *args, **kwargs):
        """Handle Deepgram errors"""
        error = kwargs.get('error') or (args[1] if len(args) > 1 else "Unknown error")
        print(f"[RT] Deepgram error: {error}", flush=True)
        
        self.is_running = False
        
        self._schedule_async(self.on_transcript({
            "type": "error",
            "message": str(error)
        }))
        
    def _handle_close(self, *args, **kwargs):
        """Handle connection close"""
        self.is_running = False
        print("[RT] Deepgram connection closed", flush=True)
        
        # Log usage on unexpected close
        self._log_session_usage()
    
    # ============ V3: INTELLIGENT GUIDANCE GENERATION ============
    
    async def _generate_guidance_v3(self, detection: DetectionResult, trigger_text: str):
        """
        Generate guidance using the V4 system.
        
        Routes:
        1. Phone + known objection → Globe Life rebuttal (instant, no AI)
        2. Presentation objection → Bridge phrase + Contextualized Globe Life script
           - Script template is locked verbatim from Globe Life docs
           - Sonnet fills in context slots from transcript (names, family, details)
           - Makes new agents sound like senior agents
        """
        if not self.rag_engine:
            print(f"[RT] No RAG engine, skipping guidance", flush=True)
            return
        
        if self._generating_guidance:
            print(f"[RT] Already generating guidance, skipping", flush=True)
            return
            
        self._generating_guidance = True
        
        try:
            # ============ ROUTE 1: Phone call with Globe Life rebuttal ============
            if detection.phone_rebuttal and self._call_type == "phone":
                print(f"[RT] 📞 Phone rebuttal - serving Globe Life script instantly", flush=True)
                
                # Send instant rebuttal (no AI needed)
                await self.on_guidance({
                    "type": "guidance_start",
                    "chunk": detection.phone_rebuttal,
                    "full_text": detection.phone_rebuttal,
                    "is_complete": False
                })
                
                await self.on_guidance({
                    "type": "guidance_complete",
                    "guidance": detection.phone_rebuttal,
                    "trigger": trigger_text,
                    "objection_type": detection.objection_type,
                    "instant": True
                })
                
                # Record in state machine
                if self.state_machine:
                    self.state_machine.record_objection(detection.objection_type, trigger_text)
                    self.state_machine.record_guidance(detection.phone_rebuttal, trigger_text)
                
                return
            
            # ============ ROUTE 2 & 3: Presentation - Bridge + AI ============
            
            # Send bridge phrase IMMEDIATELY (if available)
            if detection.bridge_phrase:
                print(f"[RT] 🌉 Sending bridge phrase: '{detection.bridge_phrase}'", flush=True)
                await self.on_guidance({
                    "type": "bridge",
                    "text": detection.bridge_phrase,
                    "objection_type": detection.objection_type
                })
            
            # Record objection in state machine
            if self.state_machine:
                self.state_machine.record_objection(detection.objection_type, trigger_text)
            
            # Notify frontend of objection type (for UI updates)
            if detection.objection_type == "price":
                await self.on_guidance({
                    "type": "price_objection",
                    "trigger": trigger_text[:100]
                })
            
            # Generate AI guidance
            await self._stream_ai_guidance(detection, trigger_text)
            
        except Exception as e:
            print(f"[RT] Error in guidance generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            self._generating_guidance = False
            self.transcript_buffer = ""
    
    async def _stream_ai_guidance(self, detection: DetectionResult, trigger_text: str):
        """
        Stream contextualized Globe Life script.
        
        V4 Approach:
        - Get locked Globe Life script template for this objection
        - Send to Sonnet to fill in context slots from transcript
        - Never changes script structure, just personalizes
        - Fallback to vanilla script if AI fails
        """
        full_text = ""
        first_chunk = True
        batch_buffer = ""
        last_send_time = time.time()
        BATCH_INTERVAL = 0.08  # 80ms batching for smooth streaming
        
        try:
            # Get full transcript for context extraction
            transcript = ""
            down_close_level = 0
            
            if self.state_machine:
                state = self.state_machine.get_state_for_claude()
                transcript = state.get("recent_transcript", "")
                down_close_level = state.get("down_close_level", 0)
            
            # If no transcript, use trigger text
            if not transcript:
                transcript = self.transcript_buffer or trigger_text
            
            print(f"[RT] 📜 Contextualizing {detection.objection_type} script (down_close={down_close_level}, transcript={len(transcript)} chars)", flush=True)
            
            # Stream contextualized script
            guidance_stream = self.rag_engine.contextualize_script(
                objection_type=detection.objection_type,
                transcript=transcript,
                down_close_level=down_close_level,
                agency=self._agency,
                session_id=self._session_id
            )
            
            for chunk in guidance_stream:
                if chunk:
                    full_text += chunk
                    batch_buffer += chunk
                    
                    now = time.time()
                    
                    if first_chunk or (now - last_send_time) >= BATCH_INTERVAL:
                        if first_chunk:
                            print(f"[RT] First guidance chunk received, streaming...", flush=True)
                        
                        await self.on_guidance({
                            "type": "guidance_start" if first_chunk else "guidance_chunk",
                            "chunk": batch_buffer,
                            "full_text": full_text,
                            "is_complete": False
                        })
                        
                        batch_buffer = ""
                        last_send_time = now
                        first_chunk = False
            
            # Send remaining buffer
            if batch_buffer:
                await self.on_guidance({
                    "type": "guidance_chunk",
                    "chunk": batch_buffer,
                    "full_text": full_text,
                    "is_complete": False
                })
            
            # Send completion
            if full_text:
                print(f"[RT] Guidance complete ({len(full_text)} chars)", flush=True)
                
                if self.state_machine:
                    self.state_machine.record_guidance(full_text, trigger_text)
                
                await self.on_guidance({
                    "type": "guidance_complete",
                    "guidance": full_text,
                    "trigger": trigger_text,
                    "objection_type": detection.objection_type
                })
                
        except Exception as e:
            print(f"[RT] Error streaming guidance, using fallback: {e}", flush=True)
            import traceback
            traceback.print_exc()
            
            # Fallback to vanilla script
            try:
                fallback = self.rag_engine.get_fallback_script(
                    detection.objection_type, 
                    down_close_level if 'down_close_level' in dir() else 0
                )
                await self.on_guidance({
                    "type": "guidance_start",
                    "chunk": fallback,
                    "full_text": fallback,
                    "is_complete": False
                })
                await self.on_guidance({
                    "type": "guidance_complete",
                    "guidance": fallback,
                    "trigger": trigger_text,
                    "objection_type": detection.objection_type,
                    "fallback": True
                })
            except Exception as fallback_error:
                print(f"[RT] Fallback also failed: {fallback_error}", flush=True)
    
    # ============ LEGACY GUIDANCE (word-count triggered) ============
    
    async def _generate_guidance(self):
        """Generate AI guidance based on transcript buffer with timeout (legacy path)"""
        if not self.rag_engine:
            print(f"[RT] No RAG engine, skipping guidance", flush=True)
            return
        if not self.transcript_buffer.strip():
            print(f"[RT] Empty transcript buffer, skipping guidance", flush=True)
            return
        
        if self._generating_guidance:
            print(f"[RT] Already generating guidance, skipping", flush=True)
            return
            
        self._generating_guidance = True
        self.last_guidance_time = time.time()
        
        print(f"[RT] Starting guidance generation for: '{self.transcript_buffer[:60]}...'", flush=True)
        
        try:
            try:
                result = await asyncio.wait_for(
                    self._do_generate_guidance(),
                    timeout=self.GUIDANCE_TIMEOUT_SECONDS
                )
                
                if result:
                    await self.on_guidance(result)
                    
            except asyncio.TimeoutError:
                print(f"[RT] WARNING: Guidance generation timed out after {self.GUIDANCE_TIMEOUT_SECONDS}s", flush=True)
            
            self.transcript_buffer = ""
            
        except Exception as e:
            print(f"[RT] Guidance generation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            self._generating_guidance = False
    
    async def _do_generate_guidance(self) -> Optional[dict]:
        """Stream guidance tokens to client (legacy path for word-count triggers)"""
        try:
            full_text = ""
            first_chunk = True
            batch_buffer = ""
            last_send_time = time.time()
            BATCH_INTERVAL = 0.08
            
            print(f"[RT] Calling RAG engine for guidance...", flush=True)
            
            # Check for price objection and notify frontend
            if self.rag_engine.detect_price_objection(self.transcript_buffer):
                print(f"[RT] 💰 Price objection detected", flush=True)
                await self.on_guidance({
                    "type": "price_objection",
                    "trigger": self.transcript_buffer[-100:]
                })
                if self.state_machine:
                    self.state_machine.record_objection("price", self.transcript_buffer[-200:])
            
            # Use V2 with full state if state machine exists
            if self.state_machine:
                call_state = self.state_machine.get_state_for_claude()
                print(f"[RT] Using V2 guidance", flush=True)
                
                guidance_stream = self.rag_engine.generate_guidance_stream_v2(
                    call_state,
                    self.transcript_buffer,
                    agency=self._agency,
                    session_id=self._session_id
                )
            else:
                print(f"[RT] Using legacy guidance (no state machine)", flush=True)
                guidance_stream = self.rag_engine.generate_guidance_stream(
                    self.transcript_buffer,
                    self.call_context,
                    agency=self._agency,
                    session_id=self._session_id
                )
            
            # Stream guidance
            for chunk in guidance_stream:
                if chunk:
                    full_text += chunk
                    batch_buffer += chunk
                    
                    now = time.time()
                    
                    if first_chunk or (now - last_send_time) >= BATCH_INTERVAL:
                        if first_chunk:
                            print(f"[RT] First guidance chunk received, streaming to client...", flush=True)
                        
                        await self.on_guidance({
                            "type": "guidance_start" if first_chunk else "guidance_chunk",
                            "chunk": batch_buffer,
                            "full_text": full_text,
                            "is_complete": False
                        })
                        
                        batch_buffer = ""
                        last_send_time = now
                        first_chunk = False
            
            if batch_buffer:
                await self.on_guidance({
                    "type": "guidance_chunk",
                    "chunk": batch_buffer,
                    "full_text": full_text,
                    "is_complete": False
                })
            
            if full_text:
                print(f"[RT] Guidance complete ({len(full_text)} chars)", flush=True)
                
                if self.state_machine:
                    self.state_machine.record_guidance(
                        full_text,
                        self.transcript_buffer[-100:] if len(self.transcript_buffer) > 100 else self.transcript_buffer
                    )
                
                await self.on_guidance({
                    "type": "guidance_complete",
                    "guidance": full_text,
                    "trigger": self.transcript_buffer[-100:] if len(self.transcript_buffer) > 100 else self.transcript_buffer
                })
                return None
            else:
                print(f"[RT] WARNING: No guidance text generated", flush=True)
            
            return None
            
        except Exception as e:
            print(f"[RT] Error in guidance generation: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return None
            
    def update_context(self, context_data: dict):
        """Update call context with client information"""
        # Update legacy context (fallback)
        if "call_type" in context_data:
            self.call_context.call_type = context_data["call_type"]
            self._call_type = context_data["call_type"]  # Track for objection routing
            print(f"[RT] Call type set to: {self._call_type}", flush=True)
        if "current_product" in context_data:
            self.call_context.current_product = context_data["current_product"]
        if "client_age" in context_data:
            self.call_context.client_age = context_data["client_age"]
        if "client_occupation" in context_data:
            self.call_context.client_occupation = context_data["client_occupation"]
        if "client_family" in context_data:
            self.call_context.client_family = context_data["client_family"]
        if "agency" in context_data:
            self._agency = context_data["agency"]
            print(f"[RT] Agency set to: {self._agency} for session {self._session_id[:8]}", flush=True)
        
        # Create or update state machine (V2)
        if self.state_machine is None:
            self.state_machine = CallStateMachine(
                session_id=self._session_id,
                agency=self._agency or "default",
                call_type=context_data.get("call_type", "phone")
            )
            print(f"[RT] State machine created for session {self._session_id[:8]}", flush=True)
        
        # Update state machine with client info
        if self.state_machine:
            self.state_machine.update_client_profile(
                age=context_data.get("client_age"),
                occupation=context_data.get("client_occupation"),
                family=context_data.get("client_family"),
                budget=context_data.get("client_budget")
            )
    
    def apply_down_close(self):
        """Agent clicked down-close button - reduce coverage and generate contextual guidance"""
        if self.state_machine:
            result = self.state_machine.do_down_close(reason="price")
            level = result.get('level', 0)
            label = result.get('label', '')
            print(f"[RT] Down-close applied: level {level}, {label}", flush=True)
            
            # Generate contextual guidance for this down-close
            self._schedule_async(self._generate_down_close_guidance(level, label))
        else:
            print(f"[RT] Down-close requested but no state machine", flush=True)
    
    async def _generate_down_close_guidance(self, level: int, label: str):
        """Generate contextual down-close guidance using Claude"""
        try:
            if not self.rag_engine:
                self.rag_engine = get_rag_engine()
            
            full_text = ""
            first_chunk = True
            batch_buffer = ""
            last_send_time = time.time()
            BATCH_INTERVAL = 0.08
            
            call_state = self.state_machine.get_state_for_claude()
            
            down_close_prompt = f"""The client just said they can't afford the coverage. The agent clicked to reduce coverage.

DOWN-CLOSE LEVEL: {level} - {label}
CURRENT COVERAGE: {call_state.get('coverage_summary', 'Unknown')}

Based on the conversation so far, give the agent EXACTLY what to say to present this reduced coverage option.
- Reference something specific from the conversation (family, job, concerns mentioned)
- State the new coverage amount naturally
- Keep it to 2-3 sentences max
- Make it feel personal, not scripted"""

            print(f"[RT] Generating contextual down-close guidance...", flush=True)
            
            for chunk in self.rag_engine.generate_guidance_stream_v2(
                call_state,
                down_close_prompt,
                agency=self._agency,
                session_id=self._session_id,
                use_fast_model=True  # Use Haiku for speed
            ):
                if chunk:
                    full_text += chunk
                    batch_buffer += chunk
                    
                    now = time.time()
                    if first_chunk or (now - last_send_time) >= BATCH_INTERVAL:
                        await self.on_guidance({
                            "type": "guidance_start" if first_chunk else "guidance_chunk",
                            "chunk": batch_buffer,
                            "full_text": full_text,
                            "is_complete": False
                        })
                        batch_buffer = ""
                        last_send_time = now
                        first_chunk = False
            
            if batch_buffer:
                await self.on_guidance({
                    "type": "guidance_chunk",
                    "chunk": batch_buffer,
                    "full_text": full_text,
                    "is_complete": False
                })
            
            if full_text:
                print(f"[RT] Down-close guidance complete ({len(full_text)} chars)", flush=True)
                self.state_machine.record_guidance(full_text, f"down_close_level_{level}")
                await self.on_guidance({
                    "type": "guidance_complete",
                    "guidance": full_text,
                    "trigger": f"down_close_level_{level}"
                })
                
        except Exception as e:
            print(f"[RT] Error generating down-close guidance: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    def get_session_stats(self) -> dict:
        """Get current session statistics for debugging/monitoring"""
        bytes_per_second = self.SAMPLE_RATE * self.BYTES_PER_SAMPLE * self.CHANNELS
        current_duration = self._total_audio_bytes / bytes_per_second if self._total_audio_bytes > 0 else 0
        
        return {
            "session_id": self._session_id,
            "agency": self._agency,
            "call_type": self._call_type,
            "audio_bytes": self._total_audio_bytes,
            "audio_duration_seconds": current_duration,
            "is_running": self.is_running,
            "transcript_buffer_words": len(self.transcript_buffer.split())
        }