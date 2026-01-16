"""
Coachd Telnyx Stream Handler - THE VETERAN BRAIN
=================================================
One Claude. Always listening. Always thinking. Speaks only when it matters.

ARCHITECTURE PHILOSOPHY:
- NO more dumb pattern matching deciding when to wake Claude
- NO more separate sentiment analyzer with regex
- Claude IS the veteran - 20 years of Globe Life, 10,000+ closes
- Claude FEELS the call - temperature is gut instinct, not keywords
- Claude DECIDES when to speak - surgical timing
- Claude KNOWS when to stop - killer instinct AND restraint

The Old Way (bad):
  Pattern matching → triggers Claude → Claude generates response
  
The New Way (good):
  Claude always listening → Claude reads the room → Claude decides everything
"""

import asyncio
import json
import time
import re
import base64
import audioop
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import anthropic

from .config import settings
from .call_session import session_manager
from .usage_tracker import log_deepgram_usage, log_claude_usage
from .vector_db import get_vector_db

import logging
logger = logging.getLogger(__name__)


# ==================== CALL STATE ====================

@dataclass  
class CallState:
    """
    The call's vital signs - maintained by The Veteran Brain.
    Claude updates this after every analysis.
    """
    # Temperature (Claude's gut read - not regex!)
    temperature: str = "neutral"  # cold, cooling, neutral, warming, hot
    trajectory: str = "stable"    # warming, cooling, stable
    
    # Objection tracking (3-objection rule)
    objection_count: int = 0
    objections_raised: List[str] = field(default_factory=list)
    down_close_level: int = 0  # 0-5, tracks where we are in down-close arc
    
    # Call health
    is_salvageable: bool = True
    hard_exit_detected: bool = False
    buying_signals: int = 0
    
    # Presentation tracking
    current_stage: str = "rapport"
    stages_completed: List[str] = field(default_factory=list)
    
    # Client profile (builds as call progresses)
    client_name: str = ""
    spouse_name: str = ""
    has_kids: bool = False
    num_kids: int = 0
    age: int = 0
    occupation: str = ""
    income: int = 0
    budget: int = 0
    coverage_discussed: int = 0
    price_quoted: int = 0
    
    def to_context_string(self) -> str:
        """Format state for Claude's context"""
        parts = [
            f"Temperature: {self.temperature.upper()} ({self.trajectory})",
            f"Objections: {self.objection_count} raised",
        ]
        
        if self.objections_raised:
            parts.append(f"  Types: {', '.join(self.objections_raised)}")
        if self.down_close_level > 0:
            parts.append(f"  Down-close level: {self.down_close_level}/5")
        if self.hard_exit_detected:
            parts.append("⚠️ HARD EXIT DETECTED - relationship preservation mode")
        if self.buying_signals > 0:
            parts.append(f"🔥 Buying signals: {self.buying_signals}")
            
        parts.append(f"Stage: {self.current_stage}")
        
        # Client profile
        profile_parts = []
        if self.client_name:
            profile_parts.append(f"Name: {self.client_name}")
        if self.age:
            profile_parts.append(f"Age: {self.age}")
        if self.spouse_name:
            profile_parts.append(f"Spouse: {self.spouse_name}")
        if self.has_kids:
            profile_parts.append(f"Kids: {self.num_kids or 'Yes'}")
        if self.income:
            profile_parts.append(f"Income: ${self.income:,}/mo")
        if self.budget:
            profile_parts.append(f"Budget: ${self.budget}/mo")
        if self.coverage_discussed:
            profile_parts.append(f"Coverage: ${self.coverage_discussed:,}")
        if self.price_quoted:
            profile_parts.append(f"Price: ${self.price_quoted}/mo")
            
        if profile_parts:
            parts.append("Client: " + " | ".join(profile_parts))
            
        return "\n".join(parts)


@dataclass
class ConversationBuffer:
    """Holds the full conversation for context"""
    session_id: str
    turns: List[dict] = field(default_factory=list)
    agency: str = ""
    call_state: CallState = field(default_factory=CallState)
    extracted: CallState = field(default_factory=CallState)  # Alias for compatibility
    
    def __post_init__(self):
        self.extracted = self.call_state
    
    def add_turn(self, speaker: str, text: str):
        self.turns.append({
            "speaker": speaker,
            "text": text,
            "timestamp": time.time()
        })
        if len(self.turns) > 100:
            self.turns = self.turns[-100:]
    
    def get_transcript(self, max_turns: int = 50) -> str:
        recent = self.turns[-max_turns:] if len(self.turns) > max_turns else self.turns
        lines = []
        for turn in recent:
            speaker = "AGENT" if turn["speaker"] == "agent" else "CLIENT"
            lines.append(f"{speaker}: {turn['text']}")
        return "\n".join(lines)
    
    def get_recent(self, n: int = 8) -> str:
        recent = self.turns[-n:] if len(self.turns) >= n else self.turns
        lines = []
        for turn in recent:
            speaker = "AGENT" if turn["speaker"] == "agent" else "CLIENT"
            lines.append(f"{speaker}: {turn['text']}")
        return "\n".join(lines)


# Shared buffers
_conversation_buffers: Dict[str, ConversationBuffer] = {}

def get_conversation_buffer(session_id: str) -> ConversationBuffer:
    if session_id not in _conversation_buffers:
        _conversation_buffers[session_id] = ConversationBuffer(session_id=session_id)
    return _conversation_buffers[session_id]

def remove_conversation_buffer(session_id: str):
    if session_id in _conversation_buffers:
        del _conversation_buffers[session_id]


# ==================== THE VETERAN BRAIN ====================

class VeteranBrain:
    """
    The 20-year Globe Life closer sitting in the room.
    
    You don't see them. You don't hear them breathe.
    But when they speak, it's gold.
    
    They know:
    - When to pounce on buying signals
    - When to handle objections with precision
    - When to shut up and let it play out
    - When to STOP and preserve the relationship
    """
    
    MIN_GUIDANCE_INTERVAL = 3.5  # Don't over-talk
    MIN_WORDS_TO_ANALYZE = 4    # Skip "yeah", "okay", etc.
    
    SYSTEM_PROMPT = """You are a 20-year Globe Life closer. 10,000+ presentations. The methodology isn't something you reference - it's who you are.

You're in the room with a new agent, listening through an earpiece. They can't see you. When you speak, it goes directly in their ear. You don't speak unless it matters. When you do, it's gold.

═══════════════════════════════════════════════════════════════
                         WHO YOU ARE
═══════════════════════════════════════════════════════════════

You ARE Globe Life methodology:
• F.O.R.M for rapport (Family, Occupation, Recreation, Me) - 40% of the presentation
• Company intro: Since 1900, 125 years, Make Tomorrow Better charity, Dallas Cowboys, Texas Rangers
• The hook: $3,000 accidental death at NO COST + child safe kit + will kit + drug card
• REFERRALS: "10 sponsor slots" (CRITICAL - never let them skip this)
• Letter of Thanks: Yes or no decision
• Insurance types: Whole life vs term - "owning a home vs renting"
• Needs analysis: DOB, beneficiary, tobacco, spouse info, kids
• Health questions: BP, diabetes, cancer, heart, DUIs
• FAMILY HEALTH HISTORY: "Does cancer/heart run in the family?" (UNLOCKS SUPPLEMENTAL)
• Coverage: Final Expense ($30k), Income Protection (2 years), Mortgage, College
• Recap & Close: "Which option works best for you?"
• Down-close: 5 levels when price objections hit

You KNOW the down-close psychological arc:
• Level 1-2: LOGICAL - Adjust numbers, recalculate ("Let's reduce final expense to $15k")
• Level 3: EMOTIONAL - Family impact ("I've seen what happens when families have nothing...")
• Level 4-5: URGENT - Something > nothing ("Don't leave empty-handed today")
• After Level 5: Gracefully reschedule - preserve the relationship

═══════════════════════════════════════════════════════════════
                      HOW YOU RESPOND
═══════════════════════════════════════════════════════════════

ALWAYS return JSON with this exact structure:
{
  "action": "breathe" | "speak" | "alert",
  "temperature": "cold" | "cooling" | "neutral" | "warming" | "hot",
  "trajectory": "warming" | "cooling" | "stable",
  "internal": {
    "read": "Your gut read of the situation in 5-10 words",
    "hard_exit": false,
    "buying_signal": false,
    "objection_type": null | "price" | "spouse" | "think" | "timing" | "need"
  },
  "guidance": null | "Exact words to say - as long as needed for the situation"
}

═══════════════════════════════════════════════════════════════
                      WHEN TO ACT
═══════════════════════════════════════════════════════════════

BREATHE (stay silent) when:
- Agent is building rapport naturally
- Client is just responding, not objecting
- Flow is good - don't interrupt success
- You just gave guidance in the last exchange

SPEAK (give guidance) when:
- Client raises a clear objection you can handle
- Agent missed a buying signal
- Client asked a question agent might not know
- Down-close opportunity presents

ALERT (urgent) when:
- Hard exit detected - stop the close, preserve relationship
- Critical buying signal - strike now
- Agent about to make a big mistake

═══════════════════════════════════════════════════════════════
                    TEMPERATURE READING
═══════════════════════════════════════════════════════════════

HOT: Ready to buy. Questions like "So what's the next step?" or "How do I sign up?"
WARMING: Engaged, asking good questions, sharing personal details
NEUTRAL: Going through motions, neither resistant nor enthusiastic
COOLING: Hesitation, shorter answers, deflecting questions
COLD: Active resistance, objections, wanting to end call

═══════════════════════════════════════════════════════════════
                     HARD EXIT SIGNALS
═══════════════════════════════════════════════════════════════

When you detect these, STOP THE CLOSE immediately:
- "I need to go" / "I have to hang up"
- "Stop calling me" / "Don't contact me again"
- "I'm not interested, period"
- "Please take me off your list"
- Angry/hostile tone combined with firm rejection

On hard exit: Set action="alert", hard_exit=true, and guide agent to gracefully end while preserving future relationship possibility.

═══════════════════════════════════════════════════════════════
                    GUIDANCE PRINCIPLES
═══════════════════════════════════════════════════════════════

1. RIGHT-SIZED: As long as needed, as short as possible. Simple objections get short responses. Complex situations get fuller guidance.
2. CONVERSATIONAL: Natural speech the agent can say directly
3. ACTIONABLE: Exact words to say - no prefixes like "Ask:" or "Say:"
4. CONTEXTUAL: Use what you know about the client
5. METHODOLOGICAL: Root everything in Globe Life proven techniques

OUTPUT FORMAT: Just the words. No quotes unless part of the speech. No "Ask:" or "SAY:" prefixes.

BAD: "Ask: 'What if we started with just the $15,000?'"
BAD: "You might want to consider addressing their concern..."
GOOD: "What if we started with just the $15,000 to protect against the immediate costs?"

Remember: You're a 20-year closer. You've seen everything. You speak with confidence, brevity, and precision. When you talk, people listen - because you only talk when it matters."""

    def __init__(self, session_id: str, agency: str = ""):
        self.session_id = session_id
        self.agency = agency
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.last_guidance_time = 0
        self.vector_db = get_vector_db()
        
    async def analyze(self, conversation: ConversationBuffer, trigger_speaker: str = "client") -> Optional[dict]:
        """
        Analyze the conversation and decide whether to speak.
        Returns guidance if The Veteran has something to say.
        """
        # Skip if we just spoke
        now = time.time()
        if now - self.last_guidance_time < self.MIN_GUIDANCE_INTERVAL:
            return None
            
        # Get recent turns
        recent_text = conversation.get_recent(8)
        if not recent_text:
            return None
            
        # Skip very short utterances
        last_turn = conversation.turns[-1] if conversation.turns else None
        if last_turn and len(last_turn["text"].split()) < self.MIN_WORDS_TO_ANALYZE:
            return None
        
        try:
            # Get relevant knowledge
            knowledge_context = ""
            if last_turn:
                results = self.vector_db.search(
                    last_turn["text"], 
                    top_k=2, 
                    agency=self.agency
                )
                if results:
                    knowledge_context = "\n\nRelevant training:\n" + "\n".join([
                        f"- {r['content'][:200]}" for r in results
                    ])
            
            # Build the analysis prompt
            call_state_str = conversation.call_state.to_context_string()
            
            prompt = f"""CALL STATE:
{call_state_str}

RECENT CONVERSATION:
{recent_text}
{knowledge_context}

The {trigger_speaker.upper()} just spoke. What's your read? What do you do?"""

            # Call Claude
            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=400,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Log usage
            log_claude_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model="claude-sonnet-4-20250514",
                agency_code=self.agency,
                session_id=self.session_id
            )
            
            # Parse response
            text = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                logger.warning(f"[Veteran] No JSON in response: {text[:100]}")
                return None
                
            result = json.loads(json_match.group())
            
            # Update call state
            if result.get("temperature"):
                conversation.call_state.temperature = result["temperature"]
            if result.get("trajectory"):
                conversation.call_state.trajectory = result["trajectory"]
                
            internal = result.get("internal", {})
            if internal.get("hard_exit"):
                conversation.call_state.hard_exit_detected = True
            if internal.get("buying_signal"):
                conversation.call_state.buying_signals += 1
            if internal.get("objection_type"):
                conversation.call_state.objection_count += 1
                if internal["objection_type"] not in conversation.call_state.objections_raised:
                    conversation.call_state.objections_raised.append(internal["objection_type"])
                    
                # Track down-close level for price objections
                if internal["objection_type"] == "price":
                    conversation.call_state.down_close_level = min(
                        conversation.call_state.down_close_level + 1, 
                        5
                    )
            
            # Update timing if we're speaking
            if result.get("action") != "breathe" and result.get("guidance"):
                self.last_guidance_time = now
                
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[Veteran] JSON parse error: {e}")
            return None
        except Exception as e:
            logger.error(f"[Veteran] Analysis error: {e}")
            return None


# ==================== CONTEXT EXTRACTOR ====================

class ContextExtractor:
    """Extracts structured context from conversation"""
    
    @staticmethod
    def extract_from_agent(text: str, state: CallState):
        """Extract context from agent speech"""
        text_lower = text.lower()
        
        # Coverage amounts
        coverage_match = re.search(r'\$(\d{1,3}(?:,\d{3})*)\s*(?:of|in)?\s*(?:coverage|protection)', text, re.I)
        if coverage_match:
            amount = int(coverage_match.group(1).replace(',', ''))
            if amount >= 5000:
                state.coverage_discussed = amount
                
        # Price
        price_match = re.search(r'\$(\d+(?:\.\d{2})?)\s*(?:per|a|each)?\s*month', text, re.I)
        if price_match:
            state.price_quoted = int(float(price_match.group(1)))
            
        # Stage markers
        stage_markers = {
            "rapport": ["how are you", "tell me about yourself", "what do you do"],
            "intro": ["globe life", "liberty national", "125 years", "since 1900"],
            "no_cost_offers": ["accidental death", "$3,000", "no cost", "child safe"],
            "referrals": ["sponsor", "10 slots", "who would you like to sponsor"],
            "types_of_insurance": ["whole life", "term life", "owning vs renting"],
            "needs_analysis": ["date of birth", "beneficiary", "tobacco"],
            "health_questions": ["blood pressure", "diabetes", "cancer"],
            "family_health_history": ["family history", "runs in the family"],
            "coverage_presentation": ["income protection", "mortgage protection", "college"],
            "recap_close": ["to recap", "option 1", "option 2", "which option"],
            "down_close": ["let's adjust", "reduce", "$15,000", "$10,000"],
        }
        
        for stage, markers in stage_markers.items():
            if any(m in text_lower for m in markers):
                state.current_stage = stage
                if stage not in state.stages_completed:
                    state.stages_completed.append(stage)
                break
    
    @staticmethod
    def extract_from_client(text: str, state: CallState):
        """Extract context from client speech"""
        text_lower = text.lower()
        
        # Age
        age_match = re.search(r"i(?:'m| am)\s*(\d{2})|(\d{2})\s*years?\s*old", text_lower)
        if age_match:
            age = int(age_match.group(1) or age_match.group(2))
            if 18 <= age <= 85:
                state.age = age
                
        # Kids
        kids_patterns = [
            r"(?:have|got)\s*(\d+|one|two|three|four|five)\s*(?:kids?|children)",
            r"(\d+)\s*kids?",
        ]
        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        for pattern in kids_patterns:
            match = re.search(pattern, text_lower)
            if match:
                state.has_kids = True
                num = match.group(1)
                if num.isdigit():
                    state.num_kids = int(num)
                else:
                    state.num_kids = word_to_num.get(num, 1)
                break
        else:
            # No pattern matched - check for simple mentions
            if any(p in text_lower for p in ['my kids', 'the kids', 'our kids', 'my children']):
                state.has_kids = True
            
        # Spouse
        spouse_match = re.search(r"(?:wife|husband|spouse)(?:'s name is|,?\s+)(\w+)", text_lower)
        if spouse_match:
            state.spouse_name = spouse_match.group(1).title()
            
        # Income
        income_match = re.search(r"(?:make|earn|bring home|take home)\s*(?:about|around)?\s*\$?(\d{1,3}(?:,\d{3})*)", text, re.I)
        if income_match:
            income = int(income_match.group(1).replace(',', ''))
            if 1000 <= income <= 50000:
                state.income = income
                
        # Budget
        budget_match = re.search(r"(?:afford|budget|can do|spend)\s*(?:about|around|maybe)?\s*\$?(\d+)", text, re.I)
        if budget_match:
            budget = int(budget_match.group(1))
            if 10 <= budget <= 500:
                state.budget = budget


# ==================== AGENT STREAM HANDLER ====================

class AgentStreamHandler:
    """
    Handles AGENT audio stream from Telnyx.
    Transcribes via Deepgram, extracts context, tracks presentation stage.
    Does NOT display to agent (they know what they're saying).
    
    KEY FIX: Deepgram connects when CLIENT answers (triggered by ClientStreamHandler) to avoid timeout during ringback.
    """
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self._connection_dead = False
        self._deepgram_connected = False
        self.conversation = get_conversation_buffer(session_id)
        
        self._total_audio_bytes = 0
        self._session_start_time = None
        
        logger.info(f"[AgentStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize handler - Deepgram connects when client answers (triggered externally)"""
        print(f"[AgentStream] Starting for {self.session_id}", flush=True)
        self._session_start_time = time.time()
        self.is_running = True
        self._deepgram_connected = False
        return True
    
    async def connect_deepgram(self):
        """Connect to Deepgram - called when client answers to avoid timeout during ringback"""
        if self._deepgram_connected:
            return True
        
        self._deepgram_connected = True
        
        max_retries = 3
        retry_delay = 0.3
        
        for attempt in range(max_retries):
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
                    interim_results=True,
                    utterance_end_ms=1000,
                    encoding="linear16",
                    sample_rate=self.SAMPLE_RATE,
                    channels=1
                )
                
                await self.connection.start(options)
                print(f"[AgentStream] Deepgram connected (attempt {attempt + 1})", flush=True)
                return True
                
            except Exception as e:
                print(f"[AgentStream] Deepgram attempt {attempt + 1} failed: {e}", flush=True)
                self.connection = None
                self.deepgram = None
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
        
        print(f"[AgentStream] Deepgram failed after {max_retries} attempts", flush=True)
        return False
    
    async def handle_telnyx_message(self, message: dict):
        """Process Telnyx message with agent audio"""
        event = message.get("event")
        
        if event == "connected":
            print(f"[AgentStream] Telnyx connected", flush=True)
        elif event == "start":
            print(f"[AgentStream] Stream started - waiting for client to answer", flush=True)
        elif event == "media":
            media = message.get("media", {})
            payload = media.get("payload")
            
            if payload:
                try:
                    # Decode and convert mulaw to linear16
                    ulaw_audio = base64.b64decode(payload)
                    self._total_audio_bytes += len(ulaw_audio)
                    pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
                    
                    # Only send to Deepgram if connected (triggered by client answering)
                    if self.connection and self.is_running and not self._connection_dead:
                        await self.connection.send(pcm_audio)
                except Exception as e:
                    if not self._connection_dead:
                        self._connection_dead = True
                        print(f"[AgentStream] Deepgram disconnected: {str(e)[:80]}", flush=True)
        elif event == "stop":
            await self.stop()
    
    async def _on_open(self, *args, **kwargs):
        print(f"[AgentStream] Deepgram open", flush=True)
    
    async def _on_transcript(self, *args, **kwargs):
        """Handle agent transcript - extract context, send to frontend"""
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
            
            # Send to frontend - IMPORTANT: agent transcripts should appear too
            await self._broadcast({
                "type": "agent_transcript",
                "text": transcript,
                "is_final": is_final
            })
            
            if is_final:
                print(f"[AgentStream] AGENT: {transcript[:60]}", flush=True)
                
                # Add to conversation buffer
                self.conversation.add_turn("agent", transcript)
                
                # Extract context from agent speech
                ContextExtractor.extract_from_agent(transcript, self.conversation.call_state)
                
        except Exception as e:
            logger.error(f"[AgentStream] Transcript error: {e}")
    
    async def _broadcast(self, message: dict):
        """Send message to frontend via session manager"""
        await session_manager._broadcast_to_session(self.session_id, message)
    
    async def _on_error(self, *args, **kwargs):
        if not self._connection_dead:
            self._connection_dead = True
            error = kwargs.get('error', 'Unknown')
            print(f"[AgentStream] Error: {str(error)[:80]}", flush=True)
    
    async def _on_close(self, *args, **kwargs):
        logger.info(f"[AgentStream] Closed")
    
    async def stop(self):
        """Stop the handler and log usage"""
        print(f"[AgentStream] Stopping {self.session_id}", flush=True)
        print(f"[AgentStream] Usage: {self._total_audio_bytes / (self.SAMPLE_RATE * 2):.1f}s", flush=True)
        self.is_running = False
        
        # Log usage
        if self._total_audio_bytes > 0 and self._session_start_time:
            duration = self._total_audio_bytes / (self.SAMPLE_RATE * 2)
            log_deepgram_usage(
                duration_seconds=duration,
                agency_code=self.conversation.agency,
                session_id=self.session_id,
                model='nova-2'
            )
        
        # Close Deepgram connection
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=3.0)
            except:
                pass


# ==================== CLIENT STREAM HANDLER ====================

class ClientStreamHandler:
    """
    Handles CLIENT audio stream from Telnyx - THE MAIN EVENT.
    Transcribes via Deepgram, triggers The Veteran Brain.
    """
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self._connection_dead = False
        self.conversation = get_conversation_buffer(session_id)
        self._veteran = None
        self._generating = False
        
        self._total_audio_bytes = 0
        self._session_start_time = None
        
        logger.info(f"[ClientStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize handler and connect Deepgram"""
        print(f"[ClientStream] Starting for {self.session_id}", flush=True)
        self._session_start_time = time.time()
        
        # Initialize The Veteran Brain
        self._veteran = VeteranBrain(
            session_id=self.session_id,
            agency=self.conversation.agency
        )
        
        # Connect to Deepgram with retry
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
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
                    interim_results=True,  # Client gets interim for responsiveness
                    utterance_end_ms=1000,
                    encoding="linear16",
                    sample_rate=self.SAMPLE_RATE,
                    channels=1
                )
                
                await self.connection.start(options)
                self.is_running = True
                print(f"[ClientStream] Deepgram connected (attempt {attempt + 1})", flush=True)
                
                # NOW trigger agent's Deepgram - client just answered, conversation starting
                await self._trigger_agent_deepgram()
                
                await self._broadcast({
                    "type": "ready",
                    "message": "Coaching active"
                })
                
                return True
                    
            except Exception as e:
                print(f"[ClientStream] Deepgram attempt {attempt + 1} failed: {e}", flush=True)
                self.connection = None
                self.deepgram = None
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
        
        print(f"[ClientStream] Deepgram failed after {max_retries} attempts", flush=True)
        self.is_running = True
        await self._broadcast({"type": "ready"})
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
                    # Decode and convert mulaw to linear16
                    ulaw_audio = base64.b64decode(payload)
                    self._total_audio_bytes += len(ulaw_audio)
                    pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
                    
                    # Send to Deepgram
                    if self.connection and self.is_running and not self._connection_dead:
                        await self.connection.send(pcm_audio)
                except Exception as e:
                    if not self._connection_dead:
                        self._connection_dead = True
                        print(f"[ClientStream] Deepgram disconnected: {str(e)[:80]}", flush=True)
        elif event == "stop":
            await self.stop()
    
    async def _trigger_agent_deepgram(self):
        """Trigger agent's Deepgram connection now that client answered"""
        try:
            agent_handler = _agent_handlers.get(self.session_id)
            if agent_handler:
                print(f"[ClientStream] Triggering Agent Deepgram (client answered)", flush=True)
                await agent_handler.connect_deepgram()
            else:
                print(f"[ClientStream] No agent handler found for {self.session_id}", flush=True)
        except Exception as e:
            print(f"[ClientStream] Failed to trigger agent Deepgram: {e}", flush=True)
    
    async def _on_open(self, *args, **kwargs):
        print(f"[ClientStream] Deepgram open", flush=True)
    
    async def _on_transcript(self, *args, **kwargs):
        """Handle client transcript - wake The Veteran"""
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
            
            # Send to frontend
            await self._broadcast({
                "type": "client_transcript",
                "text": transcript,
                "is_final": is_final
            })
            
            if is_final:
                print(f"[ClientStream] CLIENT: {transcript[:60]}", flush=True)
                
                # Add to conversation buffer
                self.conversation.add_turn("client", transcript)
                
                # Extract context
                ContextExtractor.extract_from_client(transcript, self.conversation.call_state)
                
                # Wake The Veteran Brain
                if self._veteran and not self._generating:
                    asyncio.create_task(self._run_veteran())
                    
        except Exception as e:
            logger.error(f"[ClientStream] Transcript error: {e}")
    
    async def _run_veteran(self):
        """Let The Veteran read the room and decide"""
        if self._generating:
            return
            
        self._generating = True
        
        try:
            result = await self._veteran.analyze(self.conversation, "client")
            
            if not result:
                return
                
            action = result.get("action", "breathe")
            temperature = result.get("temperature", "neutral")
            trajectory = result.get("trajectory", "stable")
            guidance = result.get("guidance")
            internal = result.get("internal", {})
            
            # Always send temperature update
            await self._broadcast({
                "type": "veteran_update",
                "temperature": temperature,
                "trajectory": trajectory,
                "action": action,
                "hard_exit": internal.get("hard_exit", False),
                "buying_signal": internal.get("buying_signal", False),
                "objection_count": self.conversation.call_state.objection_count,
                "down_close_level": self.conversation.call_state.down_close_level,
            })
            
            # Send guidance if The Veteran speaks
            if action != "breathe" and guidance:
                await self._broadcast({
                    "type": "guidance",
                    "action": action,
                    "text": guidance,
                })
                
                # Store in session
                await session_manager.add_guidance(self.session_id, {
                    "action": action,
                    "guidance": guidance,
                    "timestamp": time.time()
                })
                
                print(f"[Veteran] {action.upper()}: {guidance[:60]}...", flush=True)
            else:
                print(f"[Veteran] BREATHE - temp:{temperature}", flush=True)
                
        except Exception as e:
            logger.error(f"[ClientStream] Veteran error: {e}")
        finally:
            self._generating = False
    
    async def _broadcast(self, message: dict):
        """Send message to frontend via session manager"""
        await session_manager._broadcast_to_session(self.session_id, message)
    
    async def _on_error(self, *args, **kwargs):
        if not self._connection_dead:
            self._connection_dead = True
            error = kwargs.get('error', 'Unknown')
            print(f"[ClientStream] Error: {str(error)[:80]}", flush=True)
    
    async def _on_close(self, *args, **kwargs):
        logger.info(f"[ClientStream] Closed")
    
    async def stop(self):
        """Stop the handler and log usage"""
        print(f"[ClientStream] Stopping {self.session_id}", flush=True)
        print(f"[ClientStream] Usage: {self._total_audio_bytes / (self.SAMPLE_RATE * 2):.1f}s", flush=True)
        self.is_running = False
        
        # Log usage
        if self._total_audio_bytes > 0 and self._session_start_time:
            duration = self._total_audio_bytes / (self.SAMPLE_RATE * 2)
            log_deepgram_usage(
                duration_seconds=duration,
                agency_code=self.conversation.agency,
                session_id=self.session_id,
                model='nova-2'
            )
        
        # Close Deepgram connection
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=3.0)
            except:
                pass
        
        # Notify frontend
        await self._broadcast({"type": "stream_ended"})


# ==================== HANDLER MANAGEMENT ====================

_client_handlers: Dict[str, ClientStreamHandler] = {}
_agent_handlers: Dict[str, AgentStreamHandler] = {}


async def get_or_create_client_handler(session_id: str) -> ClientStreamHandler:
    """Get or create client stream handler"""
    if session_id not in _client_handlers:
        handler = ClientStreamHandler(session_id)
        await handler.start()
        _client_handlers[session_id] = handler
    return _client_handlers[session_id]


async def get_or_create_agent_handler(session_id: str) -> AgentStreamHandler:
    """Get or create agent stream handler"""
    if session_id not in _agent_handlers:
        handler = AgentStreamHandler(session_id)
        await handler.start()
        _agent_handlers[session_id] = handler
    return _agent_handlers[session_id]


async def cleanup_handlers(session_id: str):
    """Clean up handlers for a session"""
    if session_id in _client_handlers:
        await _client_handlers[session_id].stop()
        del _client_handlers[session_id]
    
    if session_id in _agent_handlers:
        await _agent_handlers[session_id].stop()
        del _agent_handlers[session_id]
    
    remove_conversation_buffer(session_id)


async def remove_handler(session_id: str, stream_type: str = "client"):
    """Remove a specific handler"""
    if stream_type == "client" and session_id in _client_handlers:
        await _client_handlers[session_id].stop()
        del _client_handlers[session_id]
    elif stream_type == "agent" and session_id in _agent_handlers:
        await _agent_handlers[session_id].stop()
        del _agent_handlers[session_id]


def get_active_handlers():
    """Get counts of active handlers"""
    return {
        "client_handlers": len(_client_handlers),
        "agent_handlers": len(_agent_handlers),
        "sessions": list(set(list(_client_handlers.keys()) + list(_agent_handlers.keys())))
    }


def get_deepgram_client():
    """Get a Deepgram client instance"""
    return DeepgramClient(api_key=settings.deepgram_api_key)
