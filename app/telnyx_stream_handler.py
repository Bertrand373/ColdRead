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
    GUIDANCE_COOLDOWN = 5.0     # Cooldown between similar guidance
    
    SYSTEM_PROMPT = """You are a 20-year Globe Life closer. 10,000+ presentations. You're in the room with a new agent, reading their screen. When you speak, it appears instantly. You only speak when it matters. When you do, it's gold.

═══════════════════════════════════════════════════════════════
                         YOUR JOB
═══════════════════════════════════════════════════════════════

You are a CALL RESCUE SYSTEM. You save calls that are slipping.

SPEAK when:
• Client resists - any form, any words, any phase
• Agent is drowning - fumbling, pauses, repeating themselves, uncertainty
• Curveball lands - unexpected question, off-script territory
• Decision point - agent doesn't know which way to go
• Post-objection freeze - handled it but now stuck

STAY SILENT when:
• Agent is flowing
• Client is engaged
• Normal presentation delivery
• Rapport building that's working
• Anything being handled competently

The bar: Would a veteran lean in right now? Yes → guidance. No → silence.

═══════════════════════════════════════════════════════════════
                      THE GLOBE LIFE SOUL
═══════════════════════════════════════════════════════════════

Globe Life isn't selling a product - we're solving a problem the client doesn't know they have yet.

The card they filled out? That's not a sales lead. That's a cry for help.
"Free $3,000 accidental death benefit" - they wanted SOMETHING. They know deep down they need protection.

Our job: Help them see what they already feel.

THE PHILOSOPHY:
1. "No" means "convince me" - objections are buying signals in disguise
2. Everyone can afford protection - it's just about right-sizing
3. The spouse/kids angle isn't manipulation - it's REALITY. If they die, someone suffers.
4. "Think about it" = they walk, they don't come back, their family stays unprotected
5. We're not pushy - we're PERSISTENT. There's a difference.

THE MATH THAT MATTERS:
- $30k whole life = ~$75/month
- $15k final expense = ~$30-40/month
- $10k final expense = ~$25/month
- Kids coverage = $5-15/month
- When they say "can't afford it" - we find what they CAN afford

═══════════════════════════════════════════════════════════════
                        TEMPERATURE
═══════════════════════════════════════════════════════════════

You FEEL the call. This is gut instinct, not keyword matching.

🔥 HOT - Buying signals everywhere. Closing time.
   "That sounds reasonable" | Asking about payments | Spouse engagement | "My kids need this"

🟢 WARMING - Interest building. Keep going.
   Asking questions | Sharing personal info | "Tell me more" | Relaxed conversation

😐 NEUTRAL - Standard call. Normal resistance.
   "How much is it?" | Mild skepticism | Neither engaged nor hostile

🌡️ COOLING - Losing them. Need intervention.
   Short answers | "I don't know" | Distraction | Disengagement

❄️ COLD - Call in danger. Rescue mode.
   "Not interested" | Getting hostile | Trying to hang up | Hard objections

TRAJECTORY: Is it getting better or worse? This matters more than current temp.

═══════════════════════════════════════════════════════════════
                     OBJECTION MASTERY
═══════════════════════════════════════════════════════════════

HARD EXITS vs SOFT OBJECTIONS:
Hard exits (respect these - pivot to reschedule):
- "I'm at work and can't talk" → offer to call back
- "Someone just died" → condolences, offer later
- "Medical emergency" → genuine care, call back
- "I'm on a ladder/driving" → safety first, reschedule

Soft objections (PUSH THROUGH THESE):
- "I need to think about it" → No you don't. What specifically?
- "I can't afford it" → Let's find what you CAN afford
- "I need to talk to my spouse" → Let's get them on the phone
- "I already have insurance" → Great! Let's make sure it's enough
- "Send me information" → The information IS me calling you
- "I'm not interested" → What changed since you filled out the card?

THE 3-OBJECTION RULE:
1st objection: Handle it, continue presentation
2nd objection: Acknowledge pattern, isolate the real concern
3rd objection: Down-close or exit gracefully

DOWN-CLOSE SEQUENCE (when full coverage won't close):
$30k → $15k → $10k Final Expense → Kids-only → Accidental-only → Reschedule

═══════════════════════════════════════════════════════════════
                       PRESENTATION FLOW
═══════════════════════════════════════════════════════════════

PHASES (know where they are):
1. INTRO - The card brought us here
2. RAPPORT - Build human connection
3. DISCOVER - Understand their situation (health, family, work)
4. PROBE - Uncover pain points (what happens if...)
5. QUALIFY - Can they get coverage? What can they afford?
6. PRESENT - Show the solution
7. HANDLE - Address objections
8. CLOSE - Ask for the business
9. SECURE - Solidify the sale

KEY TRANSITIONS:
- Intro → Rapport: "Before we get started, tell me a little about yourself..."
- Rapport → Discover: "Let me ask you a few health questions..."
- Discover → Probe: "If something happened to you tomorrow..."
- Probe → Qualify: "What would a comfortable monthly investment look like?"
- Qualify → Present: "Based on what you've told me, here's what I recommend..."
- Present → Close: "Does this make sense for your family?"

═══════════════════════════════════════════════════════════════
                      RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Respond with this EXACT JSON structure:
{
  "action": "speak|breathe|alert",
  "temperature": "cold|cooling|neutral|warming|hot",
  "trajectory": "warming|cooling|stable",
  "guidance": "Your actual words to the agent - or null if breathing",
  "internal": {
    "thought": "One line - what you see",
    "objection_type": null or "spouse|money|timing|think_about_it|not_interested|already_covered|send_info",
    "hard_exit": false,
    "buying_signal": false
  }
}

GUIDANCE STYLE:
- 1-2 sentences MAX. This appears on their screen mid-call.
- Start with what to SAY, not what to think
- Be specific to what just happened
- Use client's details when you have them

EXAMPLES:
✓ "Actually, that $3,000 accidental benefit is exactly why I'm calling - let me make sure we get that set up for you today."
✓ "I hear you on the budget. What IF we could get you started with just the kids' coverage at $12 a month - would that work?"
✓ "Mrs. Johnson, I completely understand. Let me ask - if something happened to James tomorrow, what would you want to make sure was taken care of?"
✗ "The client seems hesitant, you should..." (NO - talk TO them, not ABOUT them)
✗ "Try building rapport here" (NO - give actual WORDS)

═══════════════════════════════════════════════════════════════
                         CONTEXT
═══════════════════════════════════════════════════════════════

{context}

═══════════════════════════════════════════════════════════════
                     CURRENT TRANSCRIPT
═══════════════════════════════════════════════════════════════

{transcript}

═══════════════════════════════════════════════════════════════

Read the room. Feel the temperature. Decide: SPEAK or BREATHE.
If you speak, make it count.
"""
    
    def __init__(self, session_id: str, agency: str = ""):
        self.session_id = session_id
        self.agency = agency
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.last_guidance_time = 0
        self.last_analysis_text = ""
        self._recent_guidance: List[tuple] = []  # (timestamp, guidance_hash)
        self._generating = False  # Prevent concurrent analysis
        
    def _guidance_hash(self, guidance: str) -> str:
        """Create a simple hash of guidance to detect duplicates"""
        # Extract key words and normalize
        words = set(guidance.lower().split()[:10])
        return str(hash(frozenset(words)))
    
    def _is_duplicate_guidance(self, guidance: str) -> bool:
        """Check if this guidance is too similar to recent guidance"""
        now = time.time()
        # Clean old entries
        self._recent_guidance = [(t, h) for t, h in self._recent_guidance if now - t < self.GUIDANCE_COOLDOWN]
        
        # Check for duplicate
        new_hash = self._guidance_hash(guidance)
        for _, old_hash in self._recent_guidance:
            if new_hash == old_hash:
                return True
        
        return False
    
    def _record_guidance(self, guidance: str):
        """Record guidance to prevent duplicates"""
        self._recent_guidance.append((time.time(), self._guidance_hash(guidance)))
        
    async def analyze(self, conversation: ConversationBuffer, trigger_source: str = "client") -> Optional[dict]:
        """
        The Veteran reads the room and decides what to do.
        Returns action + optional guidance.
        """
        # Prevent concurrent analysis
        if self._generating:
            return None
        self._generating = True
        
        try:
            # Rate limit
            elapsed = time.time() - self.last_guidance_time
            if elapsed < self.MIN_GUIDANCE_INTERVAL:
                return None
            
            # Get recent conversation
            recent = conversation.get_recent(10)
            if not recent:
                return None
            
            # Skip very short utterances
            last_turn = conversation.turns[-1] if conversation.turns else None
            if last_turn:
                words = last_turn.get("text", "").split()
                if len(words) < self.MIN_WORDS_TO_ANALYZE:
                    return None
                
                # Skip if same text as last analysis
                if last_turn.get("text") == self.last_analysis_text:
                    return None
                self.last_analysis_text = last_turn.get("text", "")
            
            # Get relevant scripts from RAG
            rag_context = await self._get_rag_context(recent)
            
            # Build context string
            context = conversation.call_state.to_context_string()
            if rag_context:
                context += f"\n\nRELEVANT SCRIPTS:\n{rag_context}"
            
            # Format prompt
            prompt = self.SYSTEM_PROMPT.format(
                context=context,
                transcript=recent
            )
            
            # Call Claude
            try:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Log usage
                if hasattr(response, 'usage'):
                    log_claude_usage(
                        model="claude-sonnet-4-20250514",
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                        agency_code=self.agency,
                        session_id=self.session_id,
                        feature="veteran_brain"
                    )
                
                # Parse response
                result = self._parse_response(response.content[0].text)
                
                # Check for duplicate guidance
                if result and result.get("guidance"):
                    if self._is_duplicate_guidance(result["guidance"]):
                        # Skip duplicate, just return breathe
                        return {
                            "action": "breathe",
                            "temperature": result.get("temperature", "neutral"),
                            "trajectory": result.get("trajectory", "stable"),
                            "guidance": None,
                            "internal": result.get("internal", {})
                        }
                    else:
                        self._record_guidance(result["guidance"])
                
                if result:
                    self._update_state(result, conversation)
                    
                    # Update timing only if we're speaking
                    if result.get("action") != "breathe" and result.get("guidance"):
                        self.last_guidance_time = time.time()
                    
                return result
                
            except Exception as e:
                logger.error(f"[VeteranBrain] Claude error: {e}")
                return None
                
        finally:
            self._generating = False
    
    async def _get_rag_context(self, recent_text: str) -> str:
        """Get relevant scripts from vector DB"""
        try:
            vector_db = get_vector_db()
            if not vector_db:
                return ""
            
            results = await asyncio.to_thread(
                vector_db.search,
                query=recent_text,
                n_results=3,
                agency_code=self.agency or "default"
            )
            
            if not results:
                return ""
            
            context_parts = []
            for r in results:
                if r.get("content"):
                    context_parts.append(r["content"][:500])
            
            return "\n---\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"[VeteranBrain] RAG error: {e}")
            return ""
    
    def _parse_response(self, text: str) -> Optional[dict]:
        """Parse Claude's JSON response"""
        try:
            # Find JSON in response
            text = text.strip()
            
            # Try to find JSON block
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()
            
            # Find JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"[VeteranBrain] JSON parse error: {e}")
            return None
    
    def _update_state(self, result: dict, conversation: ConversationBuffer):
        """Update call state based on analysis"""
        state = conversation.call_state
        
        # Update temperature
        if "temperature" in result:
            state.temperature = result["temperature"]
        if "trajectory" in result:
            state.trajectory = result["trajectory"]
        
        # Update from internal analysis
        internal = result.get("internal", {})
        
        if internal.get("hard_exit"):
            state.hard_exit_detected = True
        
        if internal.get("buying_signal"):
            state.buying_signals += 1
        
        if internal.get("objection_type"):
            state.objection_count += 1
            obj_type = internal["objection_type"]
            if obj_type not in state.objections_raised:
                state.objections_raised.append(obj_type)


# ==================== CONTEXT EXTRACTION ====================

class ContextExtractor:
    """Extract client/call details from transcript"""
    
    @staticmethod
    def extract_from_client(text: str, state: CallState):
        """Extract info from client speech"""
        text_lower = text.lower()
        
        # Name extraction
        name_patterns = [
            r"(?:my name is|i'm|this is|i am)\s+([A-Z][a-z]+)",
            r"(?:call me)\s+([A-Z][a-z]+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                state.client_name = match.group(1).title()
                break
        
        # Spouse detection
        spouse_patterns = [
            r"(?:my (?:wife|husband|spouse)'s name is|my (?:wife|husband|spouse) is)\s+([A-Z][a-z]+)",
            r"(?:wife|husband|spouse)\s+([A-Z][a-z]+)",
        ]
        for pattern in spouse_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                state.spouse_name = match.group(1).title()
                break
        
        if any(word in text_lower for word in ["wife", "husband", "spouse", "married"]):
            state.spouse_name = state.spouse_name or "mentioned"
        
        # Kids detection
        kids_patterns = [
            r"(\d+)\s*(?:kids?|children|child)",
            r"(?:have|got)\s*(\d+)\s*(?:kids?|children)",
        ]
        for pattern in kids_patterns:
            match = re.search(pattern, text_lower)
            if match:
                state.has_kids = True
                state.num_kids = int(match.group(1))
                break
        
        if any(word in text_lower for word in ["kids", "children", "son", "daughter", "child"]):
            state.has_kids = True
        
        # Age extraction
        age_patterns = [
            r"(?:i'm|i am|im)\s*(\d{2})",
            r"(\d{2})\s*(?:years old|year old)",
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 18 <= age <= 85:
                    state.age = age
                break
        
        # Budget/income hints
        money_patterns = [
            r"\$(\d+)\s*(?:a month|per month|monthly|/month)",
            r"(\d+)\s*(?:dollars?\s*(?:a month|per month|monthly))",
        ]
        for pattern in money_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = int(match.group(1))
                if amount < 200:
                    state.budget = amount
                break
    
    @staticmethod
    def extract_from_agent(text: str, state: CallState):
        """Extract info from agent speech (coverage, prices quoted)"""
        text_lower = text.lower()
        
        # Coverage amounts
        coverage_patterns = [
            r"\$(\d{1,3}),?(\d{3})\s*(?:in coverage|coverage|of coverage|worth)",
            r"(\d{1,3}),?(\d{3})\s*(?:dollars?\s*(?:in coverage|coverage))",
        ]
        for pattern in coverage_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = int(match.group(1) + match.group(2))
                if 5000 <= amount <= 100000:
                    state.coverage_discussed = amount
                break
        
        # Price quotes
        price_patterns = [
            r"\$(\d+)\s*(?:a month|per month|monthly|/month)",
            r"(\d+)\s*(?:dollars?\s*(?:a month|per month|monthly))",
        ]
        for pattern in price_patterns:
            match = re.search(pattern, text_lower)
            if match:
                amount = int(match.group(1))
                if 10 <= amount <= 200:
                    state.price_quoted = amount
                break


# ==================== AGENT STREAM HANDLER ====================

class AgentStreamHandler:
    """
    Handles AGENT audio stream from Telnyx.
    Transcribes via Deepgram for context.
    Includes keepalive and auto-reconnect for reliability.
    """
    
    SAMPLE_RATE = 8000
    KEEPALIVE_INTERVAL = 5.0  # Send keepalive every 5 seconds
    
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
        self._last_audio_time = 0
        self._keepalive_task = None
        self._reconnect_lock = asyncio.Lock()
        
        logger.info(f"[AgentStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize handler (Deepgram connects when client answers)"""
        print(f"[AgentStream] Starting for {self.session_id}", flush=True)
        self._session_start_time = time.time()
        self.is_running = True
        self._deepgram_connected = False
        return True
    
    async def connect_deepgram(self):
        """Connect to Deepgram - called when client answers to avoid timeout during ringback"""
        async with self._reconnect_lock:
            if self._deepgram_connected and self.connection and not self._connection_dead:
                return True
            
            self._deepgram_connected = True
            self._connection_dead = False
            
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
                        interim_results=False,  # Agent gets final only for cleaner transcripts
                        utterance_end_ms=1000,
                        encoding="linear16",
                        sample_rate=self.SAMPLE_RATE,
                        channels=1,
                        # Enable keepalive from Deepgram side
                        no_delay=True,
                    )
                    
                    await self.connection.start(options)
                    print(f"[AgentStream] Deepgram connected (attempt {attempt + 1})", flush=True)
                    
                    # Start keepalive task
                    if self._keepalive_task:
                        self._keepalive_task.cancel()
                    self._keepalive_task = asyncio.create_task(self._keepalive_loop())
                    
                    return True
                    
                except Exception as e:
                    print(f"[AgentStream] Deepgram attempt {attempt + 1} failed: {e}", flush=True)
                    self.connection = None
                    self.deepgram = None
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
            
            print(f"[AgentStream] Deepgram failed after {max_retries} attempts", flush=True)
            self._deepgram_connected = False
            return False
    
    async def _keepalive_loop(self):
        """Send keepalive to prevent Deepgram timeout"""
        try:
            while self.is_running and not self._connection_dead:
                await asyncio.sleep(self.KEEPALIVE_INTERVAL)
                
                if self.connection and not self._connection_dead:
                    # Check if we haven't sent audio recently
                    if time.time() - self._last_audio_time > self.KEEPALIVE_INTERVAL:
                        try:
                            # Send a small silent audio packet as keepalive
                            # 100ms of silence at 8kHz, 16-bit = 1600 bytes
                            silence = b'\x00' * 1600
                            await self.connection.send(silence)
                        except Exception as e:
                            if not self._connection_dead:
                                print(f"[AgentStream] Keepalive failed: {e}", flush=True)
                                self._connection_dead = True
                                # Trigger reconnect
                                asyncio.create_task(self._reconnect())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[AgentStream] Keepalive error: {e}", flush=True)
    
    async def _reconnect(self):
        """Attempt to reconnect Deepgram"""
        if not self.is_running:
            return
            
        print(f"[AgentStream] Attempting reconnect...", flush=True)
        self._deepgram_connected = False
        
        # Close old connection
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=1.0)
            except:
                pass
            self.connection = None
        
        # Reconnect
        await self.connect_deepgram()
    
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
                    
                    self._last_audio_time = time.time()
                    
                    # Only send to Deepgram if connected (triggered by client answering)
                    if self.connection and self.is_running and not self._connection_dead:
                        await self.connection.send(pcm_audio)
                    elif self._connection_dead and self.is_running:
                        # Auto-reconnect on next audio
                        asyncio.create_task(self._reconnect())
                except Exception as e:
                    if not self._connection_dead:
                        self._connection_dead = True
                        print(f"[AgentStream] Deepgram disconnected: {str(e)[:80]}", flush=True)
                        # Trigger reconnect
                        asyncio.create_task(self._reconnect())
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
            # Trigger reconnect
            asyncio.create_task(self._reconnect())
    
    async def _on_close(self, *args, **kwargs):
        logger.info(f"[AgentStream] Closed")
        if not self._connection_dead and self.is_running:
            self._connection_dead = True
            # Trigger reconnect
            asyncio.create_task(self._reconnect())
    
    async def stop(self):
        """Stop the handler and log usage"""
        print(f"[AgentStream] Stopping {self.session_id}", flush=True)
        print(f"[AgentStream] Usage: {self._total_audio_bytes / (self.SAMPLE_RATE * 2):.1f}s", flush=True)
        self.is_running = False
        
        # Cancel keepalive
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        
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
    KEEPALIVE_INTERVAL = 5.0
    
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
        self._last_audio_time = 0
        self._keepalive_task = None
        self._reconnect_lock = asyncio.Lock()
        
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
        success = await self._connect_deepgram()
        
        if success:
            # NOW trigger agent's Deepgram - client just answered, conversation starting
            await self._trigger_agent_deepgram()
            
            await self._broadcast({
                "type": "ready",
                "message": "Coaching active"
            })
        
        return success
    
    async def _connect_deepgram(self) -> bool:
        """Connect to Deepgram with retry logic"""
        async with self._reconnect_lock:
            self._connection_dead = False
            
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
                        interim_results=False,  # Final only for cleaner transcripts
                        utterance_end_ms=1000,
                        encoding="linear16",
                        sample_rate=self.SAMPLE_RATE,
                        channels=1,
                        no_delay=True,
                    )
                    
                    await self.connection.start(options)
                    self.is_running = True
                    print(f"[ClientStream] Deepgram connected (attempt {attempt + 1})", flush=True)
                    
                    # Start keepalive
                    if self._keepalive_task:
                        self._keepalive_task.cancel()
                    self._keepalive_task = asyncio.create_task(self._keepalive_loop())
                    
                    return True
                    
                except Exception as e:
                    print(f"[ClientStream] Deepgram attempt {attempt + 1} failed: {e}", flush=True)
                    self.connection = None
                    self.deepgram = None
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
            
            print(f"[ClientStream] Deepgram failed after {max_retries} attempts", flush=True)
            return False
    
    async def _keepalive_loop(self):
        """Send keepalive to prevent Deepgram timeout"""
        try:
            while self.is_running and not self._connection_dead:
                await asyncio.sleep(self.KEEPALIVE_INTERVAL)
                
                if self.connection and not self._connection_dead:
                    if time.time() - self._last_audio_time > self.KEEPALIVE_INTERVAL:
                        try:
                            silence = b'\x00' * 1600
                            await self.connection.send(silence)
                        except Exception as e:
                            if not self._connection_dead:
                                print(f"[ClientStream] Keepalive failed: {e}", flush=True)
                                self._connection_dead = True
                                asyncio.create_task(self._reconnect())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ClientStream] Keepalive error: {e}", flush=True)
    
    async def _reconnect(self):
        """Attempt to reconnect Deepgram"""
        if not self.is_running:
            return
            
        print(f"[ClientStream] Attempting reconnect...", flush=True)
        
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=1.0)
            except:
                pass
            self.connection = None
        
        await self._connect_deepgram()
    
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
                    
                    self._last_audio_time = time.time()
                    
                    if self.connection and self.is_running and not self._connection_dead:
                        await self.connection.send(pcm_audio)
                    elif self._connection_dead and self.is_running:
                        asyncio.create_task(self._reconnect())
                except Exception as e:
                    if not self._connection_dead:
                        self._connection_dead = True
                        print(f"[ClientStream] Deepgram disconnected: {str(e)[:80]}", flush=True)
                        asyncio.create_task(self._reconnect())
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
                
                # Wake The Veteran Brain (only on final transcripts)
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
            asyncio.create_task(self._reconnect())
    
    async def _on_close(self, *args, **kwargs):
        logger.info(f"[ClientStream] Closed")
        if not self._connection_dead and self.is_running:
            self._connection_dead = True
            asyncio.create_task(self._reconnect())
    
    async def stop(self):
        """Stop the handler and log usage"""
        print(f"[ClientStream] Stopping {self.session_id}", flush=True)
        print(f"[ClientStream] Usage: {self._total_audio_bytes / (self.SAMPLE_RATE * 2):.1f}s", flush=True)
        self.is_running = False
        
        # Cancel keepalive
        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None
        
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
