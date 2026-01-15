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

You KNOW the 3-objection rule:
• 1st objection: Handle with standard rebuttal  
• 2nd objection: Down-close or address the real concern
• 3rd objection: Final attempt, then offer reschedule
• 4+ objections: STOP. You're burning the lead. Preserve for callback.

═══════════════════════════════════════════════════════════════
                      YOUR KILLER INSTINCT
═══════════════════════════════════════════════════════════════

🔥 POUNCE on buying signals:
"That sounds reasonable" / "What's the next step" / "When does it start"
→ CLOSE NOW. "Which option works best for you, one or two?"

💰 CATCH soft resistance before it hardens:
"That's a lot..." / "Hmm..." / "I mean..."
→ "Let me break that down - that's less than $4 a day, less than your morning coffee."

😰 RESCUE hesitation after tie-downs:
Agent: "Does that make sense?" Client: "I guess..."
→ Weak commitment. "Paint the picture - what happens to [family] if something happens tomorrow?"

🚫 HANDLE price objection with down-close:
"Can't afford it" / "Too expensive"
→ Start down-close sequence. Use their specifics. "You mentioned 3 kids counting on you..."

👫 ISOLATE the spouse objection:
"Need to talk to my wife/husband"
→ "I respect that - but what do YOU think? Does this make sense for your family?"

⏰ CRUSH the stall:
"Let me think about it" / "Call me back"
→ "Let's get SOMETHING in place today - we can always add more later."

🤫 SHUT UP when:
✓ Client says "Yeah" / "Okay" / "Uh-huh" / "I see"
✓ Client confirms info: "Yes I'm 34" / "That's correct"
✓ Client asks clarifying question: "Is that per month?"
✓ Agent is handling it well
✓ You JUST gave guidance

═══════════════════════════════════════════════════════════════
                    HARD EXITS ≠ OBJECTIONS
═══════════════════════════════════════════════════════════════

⚠️ CRITICAL - THIS IS WHERE AGENTS BURN LEADS:

OBJECTIONS are opportunities - the client is engaged but has concerns:
"I can't afford it" → Down-close
"Need to talk to spouse" → Isolate their opinion
"Let me think about it" → Create urgency

HARD EXITS are leaving signals - the client is DONE talking:
"I gotta go" / "I have to go" / "I need to leave"
"I'm about to hang up" / "I'm hanging up now"
"Bye" / "Goodbye" / "I'm done"
"No bro" / "No man" / "Nope, stop"
"Leave me alone" / "Stop calling"
"My boss is here" / "I'm in a meeting" / "I'm at work"
"I can't sit here on the phone all day"

When you sense a HARD EXIT:
→ STOP pushing immediately
→ Preserve the relationship
→ "I totally understand - let me give you my direct number. Call me when you have 2 minutes."

Pushing after a hard exit = agent looks desperate = lead burned forever.

═══════════════════════════════════════════════════════════════
                       YOUR OUTPUT FORMAT
═══════════════════════════════════════════════════════════════

Output ONLY a JSON object. No preamble. No explanation. Just JSON.

{
  "temperature": "hot|warming|neutral|cooling|cold",
  "trajectory": "warming|cooling|stable",
  "action": "strike|guide|warn|breathe|exit",
  "guidance": "The exact words for the agent to say (null if action is 'breathe')",
  "reason": "Brief internal note - why this action (agent doesn't see this)",
  "internal": {
    "objection_type": "price|spouse|stall|covered|trust|need|null",
    "buying_signal": false,
    "hard_exit": false,
    "down_close_level": 0,
    "stage_detected": "current stage name"
  }
}

ACTIONS:
• "strike" - Buying signal! Close NOW.
• "guide" - Objection or hesitation. Help them.
• "warn" - Agent missed something critical (referrals, family health).
• "breathe" - Let it play out. Don't interrupt. guidance = null
• "exit" - Hard exit detected. Preserve relationship.

GUIDANCE RULES:
• 2-4 sentences MAX - they need to say this NOW
• Conversational tone - this is spoken, not read
• Reference their specific situation when possible
• End with a soft close or question that moves toward yes
• For "exit": Give a graceful out that preserves callback opportunity

═══════════════════════════════════════════════════════════════
                      WHAT MAKES YOU ELITE
═══════════════════════════════════════════════════════════════

Elite coaches know when to STOP.

You read the room. You feel the energy shift. You protect the relationship.
You don't turn a "not today" into a "never call me again."

Your job: Help agents close deals they would have lost AND gracefully exit 
deals that can't be saved today, setting up success for tomorrow.

Be confident, not desperate.
Be helpful, not pushy. 
Trust your gut.

You've seen it all. You've closed thousands."""

    def __init__(self, session_id: str, agency: str = None):
        self.session_id = session_id
        self.agency = agency
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
        self.last_guidance_time = 0
        self.is_analyzing = False
        self.guidance_count = 0
        
    async def analyze(self, conversation: ConversationBuffer, trigger: str = "client") -> Optional[dict]:
        """
        The Veteran reads the room and decides what to do.
        Returns the full analysis including temperature, action, and guidance.
        """
        if not self.client:
            return None
            
        if self.is_analyzing:
            return None
            
        # Get the last thing said
        if not conversation.turns:
            return None
            
        last_turn = conversation.turns[-1]
        last_text = last_turn.get("text", "")
        
        # Skip very short utterances
        if len(last_text.split()) < self.MIN_WORDS_TO_ANALYZE:
            return None
            
        # Check cooldown
        now = time.time()
        if now - self.last_guidance_time < self.MIN_GUIDANCE_INTERVAL:
            return None
            
        self.is_analyzing = True
        
        try:
            # Build context
            transcript = conversation.get_transcript(max_turns=40)
            recent = conversation.get_recent(8)
            state_context = conversation.call_state.to_context_string()
            
            # Get relevant training
            training = self._get_training(last_text)
            
            user_prompt = f"""═══ CALL STATE ═══
{state_context}

═══ CONVERSATION ═══
{transcript}

═══ LAST FEW EXCHANGES ═══
{recent}

═══ RELEVANT TRAINING ═══
{training if training else "None"}

═══ WHAT JUST HAPPENED ═══
{trigger.upper()} just spoke: "{last_text}"

Read the room. Output your JSON."""

            # Call Claude
            response = await self.client.messages.create(
                model=settings.claude_model,
                max_tokens=600,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Log usage
            if response.usage:
                log_claude_usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    agency_code=self.agency,
                    model=settings.claude_model,
                    operation="veteran_brain"
                )
            
            # Parse response
            result = self._parse_response(response.content[0].text)
            
            if result:
                # Update call state
                self._update_state(conversation.call_state, result)
                
                # Record guidance time if we're actually speaking
                if result.get("action") != "breathe":
                    self.last_guidance_time = now
                    self.guidance_count += 1
                    
            return result
            
        except Exception as e:
            logger.error(f"[VeteranBrain] Error: {e}")
            return None
        finally:
            self.is_analyzing = False
    
    def _get_training(self, text: str) -> str:
        """Get relevant training materials"""
        try:
            db = get_vector_db()
            results = db.search(text, top_k=2, agency=self.agency)
            return "\n\n".join([r["content"] for r in results])
        except:
            return ""
    
    def _parse_response(self, text: str) -> Optional[dict]:
        """Parse Claude's JSON response"""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
            return None
        except Exception as e:
            logger.error(f"[VeteranBrain] Parse error: {e}, text: {text[:100]}")
            return None
    
    def _update_state(self, state: CallState, result: dict):
        """Update call state from Claude's analysis"""
        state.temperature = result.get("temperature", state.temperature)
        state.trajectory = result.get("trajectory", state.trajectory)
        
        internal = result.get("internal", {})
        
        # Track objections
        obj_type = internal.get("objection_type")
        if obj_type and obj_type != "null":
            state.objection_count += 1
            if obj_type not in state.objections_raised:
                state.objections_raised.append(obj_type)
                
        # Track down-close
        dcl = internal.get("down_close_level", 0)
        if dcl > state.down_close_level:
            state.down_close_level = dcl
            
        # Track hard exit
        if internal.get("hard_exit"):
            state.hard_exit_detected = True
            state.is_salvageable = False
            
        # Track buying signals
        if internal.get("buying_signal"):
            state.buying_signals += 1
            
        # Update stage
        stage = internal.get("stage_detected")
        if stage and stage not in ["null", "unknown"]:
            state.current_stage = stage
            if stage not in state.stages_completed:
                state.stages_completed.append(stage)


# ==================== CONTEXT EXTRACTOR ====================

class ContextExtractor:
    """Extracts client/agent context from speech to enrich the call state"""
    
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
        elif any(p in text_lower for p in ['my kids', 'the kids', 'our kids', 'my children']):
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
    """Handles agent audio stream from Telnyx"""
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_running = False
        self.connection = None
        self.conversation = get_conversation_buffer(session_id)
        self._total_audio_bytes = 0
        self._session_start_time = None
        self._connection_dead = False
        self._deepgram_initialized = False
        
    async def start(self) -> bool:
        print(f"[AgentStream] Starting {self.session_id}", flush=True)
        self._session_start_time = time.time()
        self.is_running = True
        return True
    
    async def connect_deepgram(self):
        """Connect to Deepgram - lazy initialization"""
        if self._deepgram_initialized or not settings.deepgram_api_key:
            return
            
        self._deepgram_initialized = True
        
        try:
            client = DeepgramClient(api_key=settings.deepgram_api_key)
            self.connection = client.listen.asyncwebsocket.v("1")
            
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                encoding="mulaw",
                sample_rate=self.SAMPLE_RATE,
                channels=1,
                interim_results=True,
                utterance_end_ms=1000,
            )
            
            if await self.connection.start(options):
                print(f"[AgentStream] Deepgram connected", flush=True)
                
        except Exception as e:
            logger.error(f"[AgentStream] Deepgram error: {e}")
            
    async def send_audio(self, audio_data: bytes):
        if not self.is_running or self._connection_dead:
            return
            
        # Lazy init Deepgram on first audio
        if not self._deepgram_initialized:
            await self.connect_deepgram()
            
        if not self.connection:
            return
            
        try:
            self._total_audio_bytes += len(audio_data)
            await self.connection.send(audio_data)
        except Exception as e:
            if not self._connection_dead:
                self._connection_dead = True
                logger.error(f"[AgentStream] Send error: {e}")
                
    async def _on_open(self, *args, **kwargs):
        print(f"[AgentStream] Deepgram open", flush=True)
        
    async def _on_transcript(self, *args, **kwargs):
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
                "type": "agent_transcript",
                "text": transcript,
                "is_final": is_final
            })
            
            if is_final:
                print(f"[AgentStream] AGENT: {transcript[:60]}", flush=True)
                self.conversation.add_turn("agent", transcript)
                
                # Extract context
                ContextExtractor.extract_from_agent(transcript, self.conversation.call_state)
                
        except Exception as e:
            logger.error(f"[AgentStream] Transcript error: {e}")
            
    async def _broadcast(self, message: dict):
        await session_manager._broadcast_to_session(self.session_id, message)
        
    async def _on_error(self, *args, **kwargs):
        if not self._connection_dead:
            self._connection_dead = True
            error = kwargs.get('error', 'Unknown')
            print(f"[AgentStream] Error: {str(error)[:80]}", flush=True)
            
    async def _on_close(self, *args, **kwargs):
        logger.info("[AgentStream] Closed")
        
    async def stop(self):
        print(f"[AgentStream] Stopping {self.session_id}", flush=True)
        self.is_running = False
        
        if self._total_audio_bytes > 0 and self._session_start_time:
            duration = self._total_audio_bytes / (self.SAMPLE_RATE * 2)
            log_deepgram_usage(
                duration_seconds=duration,
                agency_code=self.conversation.agency,
                session_id=self.session_id,
                model='nova-2'
            )
            
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=3.0)
            except:
                pass


# ==================== CLIENT STREAM HANDLER ====================

class ClientStreamHandler:
    """Handles client audio stream from Telnyx - THE MAIN EVENT"""
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.is_running = False
        self.connection = None
        self.conversation = get_conversation_buffer(session_id)
        self._veteran = None
        self._generating = False
        self._total_audio_bytes = 0
        self._session_start_time = None
        self._connection_dead = False
        
    async def start(self):
        if self.is_running:
            return
            
        try:
            # Initialize The Veteran Brain
            self._veteran = VeteranBrain(
                session_id=self.session_id,
                agency=self.conversation.agency
            )
            
            # Initialize Deepgram
            client = DeepgramClient(api_key=settings.deepgram_api_key)
            self.connection = client.listen.asyncwebsocket.v("1")
            
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                smart_format=True,
                encoding="mulaw",
                sample_rate=self.SAMPLE_RATE,
                channels=1,
                interim_results=True,
                utterance_end_ms=1000,
            )
            
            if await self.connection.start(options):
                self.is_running = True
                self._session_start_time = time.time()
                print(f"[ClientStream] Started {self.session_id}", flush=True)
                
        except Exception as e:
            logger.error(f"[ClientStream] Start error: {e}")
            
    async def send_audio(self, audio_data: bytes):
        if not self.is_running or not self.connection or self._connection_dead:
            return
            
        try:
            self._total_audio_bytes += len(audio_data)
            await self.connection.send(audio_data)
        except Exception as e:
            if not self._connection_dead:
                self._connection_dead = True
                logger.error(f"[ClientStream] Send error: {e}")
                
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
                
                # Add to conversation
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
        await session_manager._broadcast_to_session(self.session_id, message)
        
    async def _on_error(self, *args, **kwargs):
        if not self._connection_dead:
            self._connection_dead = True
            error = kwargs.get('error', 'Unknown')
            print(f"[ClientStream] Error: {str(error)[:80]}", flush=True)
            
    async def _on_close(self, *args, **kwargs):
        logger.info("[ClientStream] Closed")
        
    async def stop(self):
        print(f"[ClientStream] Stopping {self.session_id}", flush=True)
        self.is_running = False
        
        if self._total_audio_bytes > 0 and self._session_start_time:
            duration = self._total_audio_bytes / (self.SAMPLE_RATE * 2)
            log_deepgram_usage(
                duration_seconds=duration,
                agency_code=self.conversation.agency,
                session_id=self.session_id,
                model='nova-2'
            )
            
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.finish(), timeout=3.0)
            except:
                pass
                
        await self._broadcast({"type": "stream_ended"})


# ==================== HANDLER MANAGEMENT ====================

_client_handlers: Dict[str, ClientStreamHandler] = {}
_agent_handlers: Dict[str, AgentStreamHandler] = {}


async def get_or_create_client_handler(session_id: str) -> ClientStreamHandler:
    if session_id not in _client_handlers:
        handler = ClientStreamHandler(session_id)
        await handler.start()
        _client_handlers[session_id] = handler
    return _client_handlers[session_id]


async def get_or_create_agent_handler(session_id: str) -> AgentStreamHandler:
    if session_id not in _agent_handlers:
        handler = AgentStreamHandler(session_id)
        await handler.start()
        _agent_handlers[session_id] = handler
    return _agent_handlers[session_id]


async def cleanup_handlers(session_id: str):
    if session_id in _client_handlers:
        await _client_handlers[session_id].stop()
        del _client_handlers[session_id]
        
    if session_id in _agent_handlers:
        await _agent_handlers[session_id].stop()
        del _agent_handlers[session_id]
        
    remove_conversation_buffer(session_id)


async def remove_handler(session_id: str, stream_type: str = "client"):
    if stream_type == "client" and session_id in _client_handlers:
        await _client_handlers[session_id].stop()
        del _client_handlers[session_id]
    elif stream_type == "agent" and session_id in _agent_handlers:
        await _agent_handlers[session_id].stop()
        del _agent_handlers[session_id]


def get_active_handlers():
    return {
        "client_handlers": len(_client_handlers),
        "agent_handlers": len(_agent_handlers),
        "sessions": list(set(list(_client_handlers.keys()) + list(_agent_handlers.keys())))
    }


def get_deepgram_client():
    from deepgram import DeepgramClient
    return DeepgramClient(api_key=settings.deepgram_api_key)
