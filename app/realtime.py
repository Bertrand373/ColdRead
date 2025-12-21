"""
Coachd Real-Time Engine
Deepgram streaming + live AI guidance

Production-ready with:
- Thread-safe async callbacks
- Timeout on AI guidance generation
- Proper error handling
- OPTIMIZED: Reduced latency for real-time feel
- BILLING: Complete usage tracking for Deepgram and Claude
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


class RealtimeTranscriber:
    """Handles real-time audio transcription via Deepgram"""
    
    # ============ LATENCY-OPTIMIZED CONFIGURATION ============
    # Target: ~800ms to first visible guidance text
    #
    # Timing chain:
    #   Deepgram interim transcript → 200-400ms
    #   Hot trigger detection       → 0ms (checked on interim)
    #   RAG search                  → 200-300ms
    #   Claude first token          → 400ms
    #   WebSocket to browser        → 50ms
    #   Total first visible:        → ~800-1000ms
    #
    GUIDANCE_TIMEOUT_SECONDS = 8.0   # Max time for AI guidance generation
    GUIDANCE_COOLDOWN_SECONDS = 3.0  # Cooldown for word-count triggers
    # Note: Hot triggers use 1.0s cooldown (hardcoded in _check_for_guidance_trigger)
    MIN_WORDS_FOR_GUIDANCE = 12      # Minimum words before non-trigger guidance
    
    # Audio format constants for duration calculation
    SAMPLE_RATE = 16000  # 16kHz
    BYTES_PER_SAMPLE = 2  # 16-bit linear PCM = 2 bytes per sample
    CHANNELS = 1  # Mono
    
    def __init__(self, on_transcript: Callable, on_guidance: Callable):
        self.on_transcript = on_transcript
        self.on_guidance = on_guidance
        self.connection = None
        self.transcript_buffer = ""
        self.last_guidance_time = 0
        self.last_trigger_text = ""  # Track last trigger to prevent semantic duplicates
        self.call_context = CallContext()  # Legacy fallback
        self.state_machine = None  # V2: Full state tracking
        self.rag_engine = None
        self.is_running = False
        self._loop = None
        self._generating_guidance = False  # Prevent concurrent guidance generation
        self._agency = None  # Track agency for RAG context and billing
        
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
            # Start conservative - get connection working first
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                interim_results=True,
                endpointing=300,
                sample_rate=self.SAMPLE_RATE,
                encoding="linear16",
                channels=self.CHANNELS
            )
            print(f"[RT] Options configured, starting connection...", flush=True)
            
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
        # Formula: bytes / (sample_rate * bytes_per_sample * channels) = seconds
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
                    
                    if transcript:
                        # Send transcript to client (thread-safe)
                        self._schedule_async(self.on_transcript({
                            "type": "transcript",
                            "text": transcript,
                            "is_final": is_final
                        }))
                        
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
    
    def _is_semantic_duplicate(self, new_text: str) -> bool:
        """Check if new trigger text is semantically similar to the last trigger"""
        if not self.last_trigger_text:
            return False
        
        # Reset tracking if it's been more than 3 seconds since last trigger
        # This allows the same objection to re-trigger if client repeats it
        # (meaning agent's response didn't land and they need a different approach)
        time_since_last = time.time() - self.last_guidance_time
        if time_since_last > 3.0:
            if self.last_trigger_text:
                print(f"[RT] Resetting duplicate tracking (>{time_since_last:.1f}s since last trigger)", flush=True)
            self.last_trigger_text = ""
            return False
        
        # Simple heuristic: if the new text starts the same way or is a substring
        new_lower = new_text.lower().strip()
        last_lower = self.last_trigger_text.lower().strip()
        
        # Check if one is a prefix/extension of the other
        if new_lower.startswith(last_lower[:30]) or last_lower.startswith(new_lower[:30]):
            return True
        
        # Check word overlap (if >70% words match, it's likely the same utterance)
        new_words = set(new_lower.split())
        last_words = set(last_lower.split())
        
        if len(new_words) > 0 and len(last_words) > 0:
            overlap = len(new_words & last_words) / max(len(new_words), len(last_words))
            if overlap > 0.7:
                return True
        
        return False
    
    def _check_for_guidance_trigger(self, latest_transcript: str, is_final: bool = True):
        """Check if we should trigger guidance generation"""
        # Don't generate if already generating
        if self._generating_guidance:
            return
        
        now = time.time()
        text_lower = latest_transcript.lower()
        
        # EXPANDED trigger keywords - life insurance specific
        # These trigger IMMEDIATELY, even mid-sentence
        hot_triggers = [
            # Price/Money - highest priority
            "afford", "expensive", "cost", "price", "money", "budget",
            "too much", "cheaper", "waste", "worth it", "tight",
            # Stalling tactics
            "think about", "talk to", "spouse", "wife", "husband",
            "not sure", "don't know", "maybe", "later", "call me back",
            "let me think", "need time", "sleep on it", "pray about", "pray on",
            # Brush-offs
            "send me information", "email me", "mail me", "send me something",
            "already have", "don't need", "not interested", "no thanks",
            "can't", "won't", "no way", "pass", "not for me",
            # Trust/Skepticism
            "scam", "pushy", "what's the catch", "fine print",
            "too good", "sounds fishy", "pyramid", "legit",
            # Health/Age objections
            "too young", "too old", "healthy", "never get sick",
            "pre-existing", "health issues", "medical",
            # Timing
            "bad time", "busy", "call back", "not now", "swamped",
            # Existing coverage
            "work insurance", "through my job", "employer", "through work",
            "social security", "government", "va ", "veteran",
            # Product objections
            "term", "whole life", "universal", "cash value",
            "investment", "stock market", "better return", "mutual fund",
            "dave ramsey", "ramsey", "suze orman",
            # Waiting/Process
            "waiting period", "how long", "blood test", "exam",
            # Decision makers
            "check with", "ask my", "run it by"
        ]
        
        has_hot_trigger = any(kw in text_lower for kw in hot_triggers)
        
        # HOT TRIGGERS: 1.5 second cooldown (react fast to objections)
        # WORD COUNT: 3 second cooldown (normal flow)
        if has_hot_trigger:
            # Reduced cooldown for hot triggers - react fast!
            if now - self.last_guidance_time < 1.5:
                return
            
            # Check for semantic duplicate (interim/final of same utterance)
            if self._is_semantic_duplicate(latest_transcript):
                print(f"[RT] Skipping duplicate trigger: '{latest_transcript[:50]}...'", flush=True)
                return
            
            print(f"[RT] 🔥 HOT TRIGGER detected: '{latest_transcript[:50]}...' - generating immediately", flush=True)
            
            # CRITICAL: Set timestamp IMMEDIATELY to prevent race condition
            # (before async call, so second trigger sees it)
            self.last_guidance_time = now
            self.last_trigger_text = latest_transcript
            
            # Add this transcript to buffer so guidance has content to work with
            if latest_transcript.strip():
                self.transcript_buffer = latest_transcript
            self._schedule_async(self._generate_guidance())
            return
        
        # For non-trigger situations, only check on FINAL transcripts
        if not is_final:
            return
            
        # Standard cooldown for word-count based triggers
        if now - self.last_guidance_time < self.GUIDANCE_COOLDOWN_SECONDS:
            return
            
        word_count = len(self.transcript_buffer.split())
        
        # Generate guidance if enough words accumulated
        if word_count > self.MIN_WORDS_FOR_GUIDANCE:
            self.last_guidance_time = now  # Set immediately for word-count triggers too
            self._schedule_async(self._generate_guidance())
                   
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
        
    async def _generate_guidance(self):
        """Generate AI guidance based on transcript buffer with timeout"""
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
            # Run guidance generation with timeout
            try:
                result = await asyncio.wait_for(
                    self._do_generate_guidance(),
                    timeout=self.GUIDANCE_TIMEOUT_SECONDS
                )
                
                if result:
                    await self.on_guidance(result)
                    
            except asyncio.TimeoutError:
                print(f"[RT] WARNING: Guidance generation timed out after {self.GUIDANCE_TIMEOUT_SECONDS}s", flush=True)
            
            # Clear buffer after processing
            self.transcript_buffer = ""
            
        except Exception as e:
            print(f"[RT] Guidance generation error: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            self._generating_guidance = False
    
    async def _do_generate_guidance(self) -> Optional[dict]:
        """Stream guidance tokens to client in real-time with batched updates for smooth display"""
        try:
            full_text = ""
            first_chunk = True
            batch_buffer = ""
            last_send_time = time.time()
            BATCH_INTERVAL = 0.08  # Send updates every 80ms for smooth visual streaming
            
            print(f"[RT] Calling RAG engine for guidance...", flush=True)
            
            # Check for price objection and notify frontend
            if self.rag_engine.detect_price_objection(self.transcript_buffer):
                print(f"[RT] 💰 Price objection detected", flush=True)
                await self.on_guidance({
                    "type": "price_objection",
                    "trigger": self.transcript_buffer[-100:]
                })
                # Record in state machine
                if self.state_machine:
                    self.state_machine.record_objection("price", self.transcript_buffer[-200:])
            
            # Use V2 with full state if state machine exists, otherwise legacy
            if self.state_machine:
                call_state = self.state_machine.get_state_for_claude()
                print(f"[RT] Using V2 guidance (phase={call_state.get('presentation_context', {}).get('current_phase', 'N/A')}, down_close={call_state.get('down_close_level', 0)})", flush=True)
                
                guidance_stream = self.rag_engine.generate_guidance_stream_v2(
                    call_state,
                    self.transcript_buffer,  # Full buffer - state machine handles limits
                    agency=self._agency,
                    session_id=self._session_id
                )
            else:
                # Legacy fallback
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
                    
                    # Send first chunk immediately for instant feedback
                    # Then batch subsequent chunks for smooth streaming effect
                    if first_chunk or (now - last_send_time) >= BATCH_INTERVAL:
                        if first_chunk:
                            print(f"[RT] First guidance chunk received, streaming to client...", flush=True)
                        
                        # Send batched content
                        await self.on_guidance({
                            "type": "guidance_start" if first_chunk else "guidance_chunk",
                            "chunk": batch_buffer,
                            "full_text": full_text,
                            "is_complete": False
                        })
                        
                        batch_buffer = ""
                        last_send_time = now
                        first_chunk = False
            
            # Send any remaining buffered content
            if batch_buffer:
                await self.on_guidance({
                    "type": "guidance_chunk",
                    "chunk": batch_buffer,
                    "full_text": full_text,
                    "is_complete": False
                })
            
            # Send completion signal
            if full_text:
                print(f"[RT] Guidance complete ({len(full_text)} chars), sending completion signal", flush=True)
                
                # Record in state machine for history
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
            print(f"[RT] Call type set to: {self.call_context.call_type}", flush=True)
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
            
            # Get full state for context
            call_state = self.state_machine.get_state_for_claude()
            
            # Build specific down-close prompt
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
                session_id=self._session_id
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
            "audio_bytes": self._total_audio_bytes,
            "audio_duration_seconds": current_duration,
            "is_running": self.is_running,
            "transcript_buffer_words": len(self.transcript_buffer.split())
        }