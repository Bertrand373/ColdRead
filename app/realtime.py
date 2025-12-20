"""
Coachd Real-Time Engine
Deepgram streaming + live AI guidance

Production-ready with:
- Thread-safe async callbacks
- Timeout on AI guidance generation
- Proper error handling
- OPTIMIZED: Reduced latency for real-time feel
"""

import asyncio
import time
import sys
from typing import Optional, Callable
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from .config import settings
from .rag_engine import get_rag_engine, CallContext


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
    
    def __init__(self, on_transcript: Callable, on_guidance: Callable):
        self.on_transcript = on_transcript
        self.on_guidance = on_guidance
        self.connection = None
        self.transcript_buffer = ""
        self.last_guidance_time = 0
        self.call_context = CallContext()
        self.rag_engine = None
        self.is_running = False
        self._loop = None
        self._generating_guidance = False  # Prevent concurrent guidance generation
        self._agency = None  # Track agency for RAG context
        
        print(f"[RT] RealtimeTranscriber initialized (cooldown={self.GUIDANCE_COOLDOWN_SECONDS}s, min_words={self.MIN_WORDS_FOR_GUIDANCE})", flush=True)
        
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
            # MAXIMUM SPEED: Fastest possible sentence detection
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                interim_results=True,
                utterance_end_ms="400",   # Was 500 - faster sentence boundaries
                endpointing=200,          # Was 300 - react faster to pauses
                vad_events=True,          # Know when speech starts/stops
                sample_rate=16000,
                encoding="linear16",
                channels=1
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
        """Stop the transcription (non-blocking)"""
        print(f"[RT] stop() called", flush=True)
        self.is_running = False
        
        if self.connection:
            try:
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.connection.finish),
                    timeout=3.0
                )
                print(f"[RT] Connection finished", flush=True)
            except asyncio.TimeoutError:
                print(f"[RT] WARNING: connection.finish() timed out", flush=True)
            except Exception as e:
                print(f"[RT] Error closing connection: {e}", flush=True)
            finally:
                self.connection = None

    def _schedule_async(self, coro):
        """
        Safely schedule an async coroutine from a sync/thread context.
        This is the key fix for Deepgram callbacks running in separate threads.
        """
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            print("[RT] WARNING: Event loop not available for callback", flush=True)
    
    def _handle_transcript(self, *args, **kwargs):
        """Handle incoming transcript from Deepgram"""
        try:
            result = kwargs.get('result') or (args[1] if len(args) > 1 else None)
            
            if not result:
                return
            
            # Extract transcript from result
            if hasattr(result, 'channel') and result.channel:
                alternatives = result.channel.alternatives
                if alternatives and len(alternatives) > 0:
                    transcript = alternatives[0].transcript
                    is_final = getattr(result, 'is_final', False)
                    
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
                                
        except Exception as e:
            print(f"[RT] Error handling transcript: {e}", flush=True)
    
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
        
        # HOT TRIGGERS: 1 second cooldown (react fast to objections)
        # WORD COUNT: 3 second cooldown (normal flow)
        if has_hot_trigger:
            # Reduced cooldown for hot triggers - react fast!
            if now - self.last_guidance_time < 1.0:
                return
            print(f"[RT] 🔥 HOT TRIGGER detected: '{latest_transcript[:50]}...' - generating immediately", flush=True)
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
        
    async def _generate_guidance(self):
        """Generate AI guidance based on transcript buffer with timeout"""
        if not self.rag_engine or not self.transcript_buffer.strip():
            return
        
        if self._generating_guidance:
            return
            
        self._generating_guidance = True
        self.last_guidance_time = time.time()
        
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
        finally:
            self._generating_guidance = False
    
    async def _do_generate_guidance(self) -> Optional[dict]:
        """Stream guidance tokens to client in real-time"""
        try:
            full_text = ""
            first_chunk = True
            
            # Stream guidance using the new streaming method
            for chunk in self.rag_engine.generate_guidance_stream(
                self.transcript_buffer,
                self.call_context,
                agency=self._agency
            ):
                if chunk:
                    full_text += chunk
                    
                    # Send each chunk immediately over WebSocket
                    await self.on_guidance({
                        "type": "guidance_chunk" if not first_chunk else "guidance_start",
                        "chunk": chunk,
                        "full_text": full_text,  # Include accumulated text for simpler client handling
                        "is_complete": False
                    })
                    first_chunk = False
            
            # Send completion signal
            if full_text:
                await self.on_guidance({
                    "type": "guidance_complete",
                    "guidance": full_text,
                    "trigger": self.transcript_buffer[-100:] if len(self.transcript_buffer) > 100 else self.transcript_buffer
                })
                return None  # Already sent via streaming
            
            return None
            
        except Exception as e:
            print(f"[RT] Error in guidance generation: {e}", flush=True)
            return None
            
    def update_context(self, context_data: dict):
        """Update call context with client information"""
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
