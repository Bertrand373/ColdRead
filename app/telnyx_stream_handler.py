"""
Coachd Telnyx Stream Handler - Claude-Powered Dual Channel
==========================================================
- Agent stream: Deepgram → context buffer (not shown to agent)
- Client stream: Deepgram → live display + Claude analysis → guidance

Claude sees BOTH sides, decides when guidance is needed.
"""

import asyncio
import base64
import time
import audioop
from typing import Dict, Optional
from dataclasses import dataclass, field

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import anthropic

from .config import settings
from .call_session import session_manager
from .usage_tracker import log_deepgram_usage, log_claude_usage
from .vector_db import get_vector_db

import logging
logger = logging.getLogger(__name__)


# Shared conversation buffer per session
_conversation_buffers: Dict[str, 'ConversationBuffer'] = {}


@dataclass
class ConversationBuffer:
    """Holds the full conversation for Claude context"""
    session_id: str
    turns: list = field(default_factory=list)
    agent_name: str = ""
    client_name: str = ""
    agency: str = ""
    
    def add_turn(self, speaker: str, text: str):
        """Add a conversation turn"""
        self.turns.append({
            "speaker": speaker,
            "text": text,
            "timestamp": time.time()
        })
        # Keep last 50 turns max
        if len(self.turns) > 50:
            self.turns = self.turns[-50:]
    
    def get_context(self, max_chars: int = 3000) -> str:
        """Get formatted conversation for Claude"""
        lines = []
        for turn in self.turns:
            speaker = "AGENT" if turn["speaker"] == "agent" else "CLIENT"
            lines.append(f"{speaker}: {turn['text']}")
        
        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[-max_chars:]
        return context
    
    def get_full_transcript(self) -> list:
        """Get full transcript for post-call analysis"""
        return self.turns.copy()


def get_conversation_buffer(session_id: str) -> ConversationBuffer:
    """Get or create conversation buffer for session"""
    if session_id not in _conversation_buffers:
        _conversation_buffers[session_id] = ConversationBuffer(session_id=session_id)
    return _conversation_buffers[session_id]


def remove_conversation_buffer(session_id: str):
    """Clean up conversation buffer"""
    if session_id in _conversation_buffers:
        del _conversation_buffers[session_id]


class ClaudeAnalyzer:
    """Analyzes conversation and decides when guidance is needed"""
    
    ANALYSIS_COOLDOWN = 2.5  # Seconds between analyses
    MIN_CLIENT_WORDS = 5  # Minimum words to trigger analysis
    
    def __init__(self, session_id: str, agency: str = None, client_context: dict = None):
        self.session_id = session_id
        self.agency = agency
        self.client_context = client_context or {}
        # Use ASYNC client for true token-by-token streaming
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
        self.last_analysis_time = 0
        self.is_analyzing = False
    
    def _format_client_context(self) -> str:
        """Format Quick Prep context for Claude prompt"""
        if not self.client_context:
            return ""
        
        ctx = self.client_context
        parts = []
        
        # Product interest
        product_map = {
            'whole_life': 'Whole Life Insurance',
            'term': 'Term Life Insurance', 
            'cancer': 'Cancer Policy',
            'accident': 'Accident Policy',
            'critical_illness': 'Critical Illness Policy'
        }
        if ctx.get('product'):
            parts.append(f"Product Interest: {product_map.get(ctx['product'], ctx['product'])}")
        
        # Demographics
        if ctx.get('age'):
            parts.append(f"Age: {ctx['age']}")
        if ctx.get('sex'):
            parts.append(f"Sex: {'Male' if ctx['sex'] == 'M' else 'Female'}")
        if ctx.get('married'):
            parts.append(f"Married: {'Yes' if ctx['married'] == 'Y' else 'No'}")
        if ctx.get('kids') == 'Y':
            kids_str = f"Has Kids: Yes"
            if ctx.get('kidsCount'):
                kids_str += f" ({ctx['kidsCount']})"
            parts.append(kids_str)
        
        # Health/Risk factors
        if ctx.get('tobacco'):
            parts.append(f"Tobacco User: {'Yes' if ctx['tobacco'] == 'Y' else 'No'}")
        if ctx.get('weight'):
            parts.append(f"Weight: {ctx['weight']} lbs")
        
        # Financial
        if ctx.get('income'):
            parts.append(f"Income: {ctx['income']}")
        if ctx.get('budget'):
            parts.append(f"Budget: ${ctx['budget']}/month")
        
        # Special notes
        if ctx.get('issue'):
            parts.append(f"Notes: {ctx['issue']}")
        
        if not parts:
            return ""
        
        return "CLIENT BACKGROUND (from agent's prep):\n" + "\n".join(f"• {p}" for p in parts)
    
    def _get_relevant_training(self, client_text: str) -> str:
        """Dynamically search training materials based on what client said"""
        try:
            db = get_vector_db()
            # Search using the actual client statement
            results = db.search(
                client_text,
                top_k=5,
                agency=self.agency
            )
            return "\n\n".join([r["content"] for r in results])
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return ""
    
    async def should_analyze(self, client_text: str) -> bool:
        """Check if we should run analysis"""
        if not self.client:
            return False
        if self.is_analyzing:
            return False
        if len(client_text.split()) < self.MIN_CLIENT_WORDS:
            return False
        if time.time() - self.last_analysis_time < self.ANALYSIS_COOLDOWN:
            return False
        return True
    
    async def analyze(self, client_text: str, conversation: ConversationBuffer) -> Optional[str]:
        """
        Analyze if guidance is needed. Returns guidance text or None.
        """
        if not await self.should_analyze(client_text):
            return None
        
        self.is_analyzing = True
        self.last_analysis_time = time.time()
        
        try:
            context = conversation.get_context()
            training_materials = self._get_relevant_training(client_text)
            client_background = self._format_client_context()
            
            prompt = f"""You are a live sales coach for a Globe Life insurance agent on an active call.

{client_background}

RELEVANT TRAINING MATERIALS:
{training_materials[:3000] if training_materials else "Use standard insurance sales techniques."}

CONVERSATION SO FAR:
{context}

CLIENT JUST SAID: "{client_text}"

Analyze this moment. Does the agent need guidance RIGHT NOW?

Consider:
- Is this an OBJECTION? (price, spouse, think about it, timing, trust)
- Is this a BUYING SIGNAL the agent should capitalize on?
- Is the agent ALREADY handling this well? (check their last response)
- Is this just NORMAL conversation flow?
- Use the CLIENT BACKGROUND to personalize your guidance (reference their age, family situation, product interest, etc.)

RESPOND WITH EXACTLY ONE OF:
1. If guidance needed: The exact words the agent should say. Be conversational, not scripted. 2-3 sentences max. Use the training materials as your guide.
2. If no guidance needed: Just the word NO_GUIDANCE_NEEDED

Do not explain your reasoning. Just give the guidance or NO_GUIDANCE_NEEDED."""

            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Log usage
            log_claude_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                agency_code=self.agency,
                model="claude-sonnet-4-20250514",
                operation="live_analysis"
            )
            
            result = response.content[0].text.strip()
            
            if "NO_GUIDANCE_NEEDED" in result.upper():
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return None
        finally:
            self.is_analyzing = False
    
    async def analyze_stream(self, client_text: str, conversation: ConversationBuffer):
        """
        Streaming version - yields guidance token-by-token for fastest display.
        Uses async client for true real-time streaming like chat.
        """
        if not await self.should_analyze(client_text):
            return
        
        self.is_analyzing = True
        self.last_analysis_time = time.time()
        
        try:
            context = conversation.get_context()
            training_materials = self._get_relevant_training(client_text)
            client_background = self._format_client_context()
            
            prompt = f"""You are a live sales coach for a Globe Life insurance agent on an active call.

{client_background}

RELEVANT TRAINING MATERIALS:
{training_materials[:3000] if training_materials else "Use standard insurance sales techniques."}

CONVERSATION SO FAR:
{context}

CLIENT JUST SAID: "{client_text}"

If the agent needs guidance right now (objection, buying signal, confusion), give them the exact words to say. 2-3 sentences, conversational. Use the training materials and client background to personalize your guidance.

If no guidance needed, respond only with: NO_GUIDANCE_NEEDED"""

            collected_text = ""
            input_tokens = 0
            output_tokens = 0
            
            # TRUE async streaming - token by token like chat
            async with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    collected_text += text
                    
                    # Check early if it's NO_GUIDANCE_NEEDED
                    if "NO_GUIDANCE" in collected_text.upper():
                        return
                    
                    yield text
                
                # Get final usage after stream completes
                final = await stream.get_final_message()
                if final and final.usage:
                    input_tokens = final.usage.input_tokens
                    output_tokens = final.usage.output_tokens
            
            # Log usage
            if input_tokens or output_tokens:
                log_claude_usage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    agency_code=self.agency,
                    model="claude-sonnet-4-20250514",
                    operation="live_analysis_stream"
                )
                
        except Exception as e:
            logger.error(f"Claude stream error: {e}")
        finally:
            self.is_analyzing = False


class AgentStreamHandler:
    """
    Handles AGENT audio stream.
    Transcribes via Deepgram, adds to conversation buffer.
    NOT shown to agent (they know what they said).
    """
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self.conversation = get_conversation_buffer(session_id)
        
        self._total_audio_bytes = 0
        self._session_start_time = None
        
        logger.info(f"[AgentStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize Deepgram for agent audio"""
        print(f"[AgentStream] Starting for {self.session_id}", flush=True)
        self._session_start_time = time.time()
        
        if not settings.deepgram_api_key:
            logger.warning("[AgentStream] No Deepgram key")
            self.is_running = True
            return True
        
        try:
            self.deepgram = DeepgramClient(settings.deepgram_api_key)
            self.connection = self.deepgram.listen.asynclive.v("1")
            
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                interim_results=False,  # Only finals for agent
                utterance_end_ms=1000,
                encoding="linear16",
                sample_rate=self.SAMPLE_RATE,
                channels=1
            )
            
            await self.connection.start(options)
            self.is_running = True
            
            print(f"[AgentStream] Deepgram connected", flush=True)
            return True
            
        except Exception as e:
            logger.error(f"[AgentStream] Deepgram error: {e}")
            self.is_running = True
            return True
    
    async def handle_telnyx_message(self, message: dict):
        """Process Telnyx message with agent audio"""
        event = message.get("event")
        
        if event == "connected":
            print(f"[AgentStream] Telnyx connected", flush=True)
        elif event == "start":
            print(f"[AgentStream] Stream started", flush=True)
        elif event == "media":
            media = message.get("media", {})
            payload = media.get("payload")
            
            if payload:
                try:
                    ulaw_audio = base64.b64decode(payload)
                    self._total_audio_bytes += len(ulaw_audio)
                    pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
                    
                    if self.connection and self.is_running:
                        await self.connection.send(pcm_audio)
                except Exception as e:
                    logger.error(f"[AgentStream] Audio error: {e}")
        elif event == "stop":
            await self.stop()
    
    async def _on_open(self, *args, **kwargs):
        print(f"[AgentStream] Deepgram open", flush=True)
    
    async def _on_transcript(self, *args, **kwargs):
        """Handle agent transcript - add to buffer, don't display"""
        try:
            result = kwargs.get('result') or (args[1] if len(args) > 1 else None)
            if not result:
                return
            
            alternatives = result.channel.alternatives
            if not alternatives:
                return
            
            transcript = alternatives[0].transcript
            if not transcript or not result.is_final:
                return
            
            print(f"[AgentStream] AGENT: {transcript[:60]}", flush=True)
            
            # Add to conversation buffer (for Claude context)
            self.conversation.add_turn("agent", transcript)
            
        except Exception as e:
            logger.error(f"[AgentStream] Transcript error: {e}")
    
    async def _on_error(self, *args, **kwargs):
        error = kwargs.get('error') or (args[1] if len(args) > 1 else "Unknown")
        logger.error(f"[AgentStream] Deepgram error: {error}")
    
    async def _on_close(self, *args, **kwargs):
        logger.info(f"[AgentStream] Deepgram closed")
    
    async def stop(self):
        """Stop and log usage"""
        print(f"[AgentStream] Stopping {self.session_id}", flush=True)
        self.is_running = False
        
        if self._total_audio_bytes > 0 and self._session_start_time:
            bytes_per_second = self.SAMPLE_RATE * 2
            duration_seconds = self._total_audio_bytes / bytes_per_second
            log_deepgram_usage(
                duration_seconds=duration_seconds,
                agency_code=self.conversation.agency,
                session_id=self.session_id,
                model='nova-2'
            )
            print(f"[AgentStream] Usage: {duration_seconds:.1f}s", flush=True)
        
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=3.0)
            except:
                pass


class ClientStreamHandler:
    """
    Handles CLIENT audio stream.
    Transcribes via Deepgram, displays live, triggers Claude analysis.
    """
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self.conversation = get_conversation_buffer(session_id)
        
        self._total_audio_bytes = 0
        self._session_start_time = None
        self._client_buffer = ""  # Current utterance being built
        
        # Claude analyzer
        self._analyzer = None
        self._generating_guidance = False
        
        logger.info(f"[ClientStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize Deepgram for client audio"""
        print(f"[ClientStream] Starting for {self.session_id}", flush=True)
        self._session_start_time = time.time()
        
        # Get agency and client context from session
        session = await session_manager.get_session(self.session_id)
        if session:
            self.conversation.agency = getattr(session, 'agency', None)
        
        # Initialize Claude analyzer with client context from Quick Prep
        client_context = getattr(session, 'client_context', None) if session else None
        if settings.anthropic_api_key:
            self._analyzer = ClaudeAnalyzer(
                self.session_id, 
                self.conversation.agency,
                client_context=client_context
            )
        
        if not settings.deepgram_api_key:
            logger.warning("[ClientStream] No Deepgram key")
            self.is_running = True
            await self._broadcast({"type": "ready"})
            return True
        
        try:
            self.deepgram = DeepgramClient(settings.deepgram_api_key)
            self.connection = self.deepgram.listen.asynclive.v("1")
            
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                punctuate=True,
                interim_results=True,  # Show live typing for client
                utterance_end_ms=1000,
                encoding="linear16",
                sample_rate=self.SAMPLE_RATE,
                channels=1
            )
            
            await self.connection.start(options)
            self.is_running = True
            
            print(f"[ClientStream] Deepgram connected", flush=True)
            
            await self._broadcast({
                "type": "ready",
                "message": "Coaching active"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"[ClientStream] Deepgram error: {e}")
            self.is_running = True
            return True
    
    async def handle_telnyx_message(self, message: dict):
        """Process Telnyx message with client audio"""
        event = message.get("event")
        
        if event == "connected":
            print(f"[ClientStream] Telnyx connected", flush=True)
        elif event == "start":
            print(f"[ClientStream] Stream started", flush=True)
        elif event == "media":
            media = message.get("media", {})
            payload = media.get("payload")
            
            if payload:
                try:
                    ulaw_audio = base64.b64decode(payload)
                    self._total_audio_bytes += len(ulaw_audio)
                    pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
                    
                    if self.connection and self.is_running:
                        await self.connection.send(pcm_audio)
                except Exception as e:
                    logger.error(f"[ClientStream] Audio error: {e}")
        elif event == "stop":
            await self.stop()
    
    async def _on_open(self, *args, **kwargs):
        print(f"[ClientStream] Deepgram open", flush=True)
    
    async def _on_transcript(self, *args, **kwargs):
        """Handle client transcript - display and analyze"""
        try:
            result = kwargs.get('result') or (args[1] if len(args) > 1 else None)
            if not result:
                return
            
            alternatives = result.channel.alternatives
            if not alternatives:
                return
            
            transcript = alternatives[0].transcript
            is_final = result.is_final
            
            if not transcript:
                return
            
            # Send to frontend for live display
            await self._broadcast({
                "type": "client_transcript",
                "text": transcript,
                "is_final": is_final
            })
            
            if is_final:
                print(f"[ClientStream] CLIENT: {transcript[:60]}", flush=True)
                
                # Add to conversation buffer
                self.conversation.add_turn("client", transcript)
                self._client_buffer = transcript
                
                # Trigger Claude analysis
                if self._analyzer and not self._generating_guidance:
                    asyncio.create_task(self._run_analysis(transcript))
                    
        except Exception as e:
            logger.error(f"[ClientStream] Transcript error: {e}")
    
    async def _run_analysis(self, client_text: str):
        """Run Claude analysis and stream guidance if needed"""
        if self._generating_guidance:
            return
        
        self._generating_guidance = True
        
        try:
            guidance_started = False
            full_guidance = ""
            
            async for chunk in self._analyzer.analyze_stream(client_text, self.conversation):
                if not guidance_started:
                    # First chunk - notify frontend guidance is coming
                    await self._broadcast({
                        "type": "guidance_start",
                        "trigger": client_text[:50]
                    })
                    guidance_started = True
                
                full_guidance += chunk
                await self._broadcast({
                    "type": "guidance_chunk",
                    "text": chunk
                })
            
            if guidance_started:
                await self._broadcast({
                    "type": "guidance_complete",
                    "full_text": full_guidance
                })
                
                # Store in session
                await session_manager.add_guidance(self.session_id, {
                    "trigger": client_text[:50],
                    "response": full_guidance,
                    "timestamp": time.time()
                })
                
        except Exception as e:
            logger.error(f"[ClientStream] Analysis error: {e}")
        finally:
            self._generating_guidance = False
    
    async def _broadcast(self, message: dict):
        """Send to frontend"""
        await session_manager._broadcast_to_session(self.session_id, message)
    
    async def _on_error(self, *args, **kwargs):
        error = kwargs.get('error') or (args[1] if len(args) > 1 else "Unknown")
        logger.error(f"[ClientStream] Deepgram error: {error}")
    
    async def _on_close(self, *args, **kwargs):
        logger.info(f"[ClientStream] Deepgram closed")
    
    async def stop(self):
        """Stop and log usage"""
        print(f"[ClientStream] Stopping {self.session_id}", flush=True)
        self.is_running = False
        
        if self._total_audio_bytes > 0 and self._session_start_time:
            bytes_per_second = self.SAMPLE_RATE * 2
            duration_seconds = self._total_audio_bytes / bytes_per_second
            log_deepgram_usage(
                duration_seconds=duration_seconds,
                agency_code=self.conversation.agency,
                session_id=self.session_id,
                model='nova-2'
            )
            print(f"[ClientStream] Usage: {duration_seconds:.1f}s", flush=True)
        
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=3.0)
            except:
                pass
        
        await self._broadcast({"type": "stream_ended"})


# Handler management
_client_handlers: Dict[str, ClientStreamHandler] = {}
_agent_handlers: Dict[str, AgentStreamHandler] = {}


async def get_or_create_client_handler(session_id: str) -> ClientStreamHandler:
    """Get or create client stream handler"""
    if session_id not in _client_handlers:
        handler = ClientStreamHandler(session_id)
        if await handler.start():
            _client_handlers[session_id] = handler
        else:
            raise Exception("Failed to start client handler")
    return _client_handlers[session_id]


async def get_or_create_agent_handler(session_id: str) -> AgentStreamHandler:
    """Get or create agent stream handler"""
    if session_id not in _agent_handlers:
        handler = AgentStreamHandler(session_id)
        if await handler.start():
            _agent_handlers[session_id] = handler
        else:
            raise Exception("Failed to start agent handler")
    return _agent_handlers[session_id]


async def remove_handler(session_id: str):
    """Remove handlers and cleanup for session"""
    if session_id in _client_handlers:
        handler = _client_handlers.pop(session_id)
        await handler.stop()
    if session_id in _agent_handlers:
        handler = _agent_handlers.pop(session_id)
        await handler.stop()
    remove_conversation_buffer(session_id)


def get_session_transcript(session_id: str) -> list:
    """Get full transcript for post-call analysis"""
    if session_id in _conversation_buffers:
        return _conversation_buffers[session_id].get_full_transcript()
    return []