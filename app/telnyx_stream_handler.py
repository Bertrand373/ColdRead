"""
Coachd Telnyx Stream Handler - Elite Guidance System + Sentiment Analysis
==========================================================================
ARCHITECTURE:
- Agent stream: Deepgram → context extraction + stage tracking (silent)
- Client stream: Deepgram → objection/hesitation detection → guidance
- Sentiment: Pattern-based real-time detection → frontend indicator

Claude sees BOTH sides, extracts context continuously, and fires guidance ONLY when:
1. Client raises objection (price, spouse, stall, trust)
2. Client hesitates after tie-down/closing question
3. Agent skips critical presentation stage (referrals, family health history)

NOT when:
- Client gives acknowledgment (yeah, okay, uh-huh)
- Client confirms info (yes I'm 34, that's correct)
- Client asks clarifying question (is that per month?)
- Normal conversation flow
"""

import asyncio
import base64
import time
import audioop
import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import anthropic

from .config import settings
from .call_session import session_manager
from .usage_tracker import log_deepgram_usage, log_claude_usage
from .vector_db import get_vector_db

import logging
logger = logging.getLogger(__name__)


# ==================== SENTIMENT ANALYSIS ====================

class SentimentLevel(Enum):
    """Client engagement levels"""
    COLD = 1       # Disengaged, exit signals
    COOLING = 2    # Hesitation, resistance building
    NEUTRAL = 3    # Information gathering
    WARMING = 4    # Interest building
    HOT = 5        # Engaged, buying signals


@dataclass
class SentimentState:
    """Tracks sentiment with history for trajectory"""
    level: SentimentLevel = SentimentLevel.NEUTRAL
    confidence: float = 0.5  # 0-1 how confident we are
    trajectory: str = "stable"  # "warming", "cooling", "stable"
    predictive_tip: str = ""  # What we predict is coming
    last_update: float = 0
    history: List[Tuple[float, SentimentLevel]] = field(default_factory=list)
    
    def update(self, new_level: SentimentLevel, confidence: float = 0.6, tip: str = ""):
        """Update sentiment with trajectory calculation"""
        now = time.time()
        
        # Add to history
        self.history.append((now, new_level))
        # Keep last 10 data points
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        # Calculate trajectory from recent history
        if len(self.history) >= 3:
            recent = self.history[-3:]
            values = [s[1].value for s in recent]
            avg_change = (values[-1] - values[0]) / len(values)
            
            if avg_change > 0.3:
                self.trajectory = "warming"
            elif avg_change < -0.3:
                self.trajectory = "cooling"
            else:
                self.trajectory = "stable"
        
        self.level = new_level
        self.confidence = confidence
        self.predictive_tip = tip
        self.last_update = now
    
    def decay_to_neutral(self, silence_seconds: float = 30):
        """Decay toward neutral after silence"""
        if time.time() - self.last_update > silence_seconds:
            if self.level != SentimentLevel.NEUTRAL:
                self.level = SentimentLevel.NEUTRAL
                self.confidence = 0.5
                self.trajectory = "stable"
                self.predictive_tip = ""


class SentimentAnalyzer:
    """
    Real-time sentiment analysis using pattern matching.
    Fast, low-latency approach for immediate feedback.
    """
    
    # Minimum interval between updates (prevents UI flickering)
    UPDATE_COOLDOWN = 1.5
    
    # Pattern categories with weights
    HOT_PATTERNS = [
        # Direct buying signals
        (r"what'?s the next step", 0.9, "Buying signal! Push to close"),
        (r"how do (i|we) (sign up|get started|enroll)", 0.95, "Ready to buy! Move to application"),
        (r"that sounds (good|great|reasonable|fair)", 0.7, "Positive reaction - build momentum"),
        (r"i('m| am) interested", 0.85, "High interest - present options"),
        (r"tell me more", 0.7, "Engaged - keep building value"),
        (r"when (can|does) (it|this|coverage) start", 0.9, "Ready to commit!"),
        (r"can (i|we) do (that|this) today", 0.95, "Immediate intent - close now"),
        # Engagement signals
        (r"that makes sense", 0.6, ""),
        (r"i (see|understand|get it)", 0.5, ""),
        (r"(oh|ah) (okay|i see)", 0.55, ""),
        # Questions showing interest
        (r"what about (my|the) (spouse|wife|husband|kids)", 0.7, "Family interest - expand coverage"),
        (r"does (it|that|this) cover", 0.7, "Coverage questions = buying interest"),
        (r"and (the|my) (spouse|family) (is|would be) covered", 0.75, ""),
    ]
    
    WARMING_PATTERNS = [
        (r"(hmm|hm+)", 0.5, ""),
        (r"okay,? (so|and)", 0.5, ""),
        (r"how much (is|would)", 0.55, "Price question - prepare value justification"),
        (r"what if", 0.5, ""),
        (r"let me (think|see)", 0.45, "Processing - give them space"),
        (r"that'?s (interesting|good to know)", 0.5, ""),
        (r"(right|okay|yes),? (and|so|but)", 0.5, ""),
    ]
    
    COOLING_PATTERNS = [
        (r"i'?m not sure", 0.6, "Doubt emerging - reinforce value"),
        (r"i don'?t know (if|about)", 0.65, "Uncertainty - address concerns"),
        (r"(maybe|perhaps)", 0.5, ""),
        (r"i('ll| will) have to (think|talk)", 0.7, "Objection incoming - prepare rebuttal"),
        (r"we('ll| will) (think|talk|discuss)", 0.7, "Spouse objection likely"),
        (r"(that'?s|it'?s) a lot", 0.65, "Price resistance building"),
        (r"i (guess|suppose)", 0.6, "Weak commitment - value reinforcement needed"),
        (r"(um|uh),? (well|i mean)", 0.55, "Hesitation detected"),
        (r"can you (send|mail|email)", 0.7, "Stall tactic coming - redirect to today"),
    ]
    
    COLD_PATTERNS = [
        (r"(not|don'?t) (need|want) (it|this|any)", 0.85, "Hard objection - find underlying concern"),
        (r"can'?t afford", 0.8, "Budget objection - prepare down-close"),
        (r"(too|very) expensive", 0.8, "Price objection"),
        (r"already have (insurance|coverage)", 0.75, "Existing coverage objection"),
        (r"not (a good|the right) time", 0.8, "Stall tactic"),
        (r"call (me )?(back|later|another)", 0.85, "Exit attempt - urgency needed"),
        (r"(not interested|no thanks)", 0.9, "Hard no - pivot to referrals"),
        (r"(send|just mail) (me )?(something|info)", 0.85, "Stall - must close today"),
        (r"(talk to|ask) (my |the )?(wife|husband|spouse)", 0.75, "Spouse objection"),
        (r"(i|we) need to (think|discuss|talk)", 0.75, "Think-about-it objection"),
        (r"what'?s the catch", 0.6, "Trust issue - build credibility"),
        (r"(sounds|seems) too good", 0.55, "Skepticism - address with proof"),
    ]
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = SentimentState()
        self.last_analysis_time = 0
        self._presentation_stage = "intro"  # Context matters
        
    def analyze(self, text: str, speaker: str = "client", stage: str = None) -> Optional[dict]:
        """
        Analyze text for sentiment signals.
        Returns dict if update needed, None if no change.
        """
        now = time.time()
        
        # Cooldown check
        if now - self.last_analysis_time < self.UPDATE_COOLDOWN:
            return None
        
        text_lower = text.lower().strip()
        if len(text_lower) < 3:
            return None
        
        # Update stage context if provided
        if stage:
            self._presentation_stage = stage
        
        # Check for decay
        self.state.decay_to_neutral()
        
        # Analyze patterns (client speech only for now)
        if speaker == "client":
            result = self._analyze_client(text_lower)
            if result:
                self.last_analysis_time = now
                return result
        
        return None
    
    def _analyze_client(self, text: str) -> Optional[dict]:
        """Analyze client speech for sentiment"""
        
        best_match = None
        best_weight = 0
        best_tip = ""
        detected_level = None
        
        # Check HOT patterns
        for pattern, weight, tip in self.HOT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if weight > best_weight:
                    best_weight = weight
                    best_match = SentimentLevel.HOT
                    best_tip = tip
                    detected_level = SentimentLevel.HOT
        
        # Check WARMING patterns
        for pattern, weight, tip in self.WARMING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if weight > best_weight:
                    best_weight = weight
                    best_match = SentimentLevel.WARMING
                    best_tip = tip
                    detected_level = SentimentLevel.WARMING
        
        # Check COOLING patterns
        for pattern, weight, tip in self.COOLING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if weight > best_weight:
                    best_weight = weight
                    best_match = SentimentLevel.COOLING
                    best_tip = tip
                    detected_level = SentimentLevel.COOLING
        
        # Check COLD patterns
        for pattern, weight, tip in self.COLD_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if weight > best_weight:
                    best_weight = weight
                    best_match = SentimentLevel.COLD
                    best_tip = tip
                    detected_level = SentimentLevel.COLD
        
        # Only update if we found something significant
        if best_match and best_weight >= 0.5:
            # Context adjustment: cold at intro is normal
            if self._presentation_stage in ["intro", "rapport"] and detected_level == SentimentLevel.NEUTRAL:
                return None
            
            self.state.update(best_match, confidence=best_weight, tip=best_tip)
            
            return {
                "type": "sentiment_update",
                "level": self.state.level.name.lower(),
                "level_value": self.state.level.value,
                "confidence": self.state.confidence,
                "trajectory": self.state.trajectory,
                "tip": self.state.predictive_tip,
            }
        
        return None
    
    def get_current_state(self) -> dict:
        """Get current sentiment state for frontend"""
        self.state.decay_to_neutral()
        return {
            "type": "sentiment_update",
            "level": self.state.level.name.lower(),
            "level_value": self.state.level.value,
            "confidence": self.state.confidence,
            "trajectory": self.state.trajectory,
            "tip": self.state.predictive_tip,
        }


# Shared conversation buffer per session
_conversation_buffers: Dict[str, 'ConversationBuffer'] = {}


@dataclass
class ExtractedContext:
    """Context extracted from conversation in real-time"""
    # Client demographics
    client_name: str = ""
    spouse_name: str = ""
    age: Optional[int] = None
    married: Optional[bool] = None
    has_kids: Optional[bool] = None
    num_kids: Optional[int] = None
    occupation: str = ""
    
    # Financial
    income: Optional[int] = None  # Monthly take-home
    budget: Optional[int] = None  # What they said they can afford
    
    # Health flags
    tobacco: Optional[bool] = None
    health_issues: list = field(default_factory=list)  # BP, diabetes, etc.
    family_health_history: list = field(default_factory=list)  # Cancer/heart in family
    
    # Coverage being discussed
    coverage_amount: Optional[int] = None  # e.g., 50000
    monthly_price: Optional[int] = None  # e.g., 127
    option_presented: str = ""  # "Option 1", "Option 2"
    products_discussed: list = field(default_factory=list)
    
    # Presentation tracking
    stages_completed: list = field(default_factory=list)
    current_stage: str = "rapport"
    referrals_collected: int = 0
    
    def to_prompt_string(self) -> str:
        """Format for Claude prompt"""
        parts = []
        
        if self.client_name:
            parts.append(f"Client: {self.client_name}")
        if self.spouse_name:
            parts.append(f"Spouse: {self.spouse_name}")
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.married is not None:
            parts.append(f"Married: {'Yes' if self.married else 'No'}")
        if self.has_kids:
            kids_str = "Has Kids: Yes"
            if self.num_kids:
                kids_str += f" ({self.num_kids})"
            parts.append(kids_str)
        if self.occupation:
            parts.append(f"Occupation: {self.occupation}")
        if self.income:
            parts.append(f"Monthly Income: ~${self.income:,}")
        if self.budget:
            parts.append(f"Stated Budget: ~${self.budget}/mo")
        if self.tobacco is not None:
            parts.append(f"Tobacco: {'Yes' if self.tobacco else 'No'}")
        if self.health_issues:
            parts.append(f"Health: {', '.join(self.health_issues)}")
        if self.family_health_history:
            parts.append(f"Family History: {', '.join(self.family_health_history)}")
        if self.coverage_amount:
            parts.append(f"Coverage Discussed: ${self.coverage_amount:,}")
        if self.monthly_price:
            parts.append(f"Price Quoted: ${self.monthly_price}/mo")
        if self.option_presented:
            parts.append(f"Option: {self.option_presented}")
        if self.products_discussed:
            parts.append(f"Products: {', '.join(self.products_discussed)}")
        
        return "\n".join(f"• {p}" for p in parts) if parts else "No context extracted yet"
    
    def get_stages_status(self) -> str:
        """Get presentation stage status for Claude"""
        all_stages = [
            "rapport", "intro", "no_cost_offers", "referral_collection",
            "letter_of_thanks", "types_of_insurance", "needs_analysis",
            "health_questions", "family_health_history", "coverage_presentation",
            "recap_close"
        ]
        
        completed = set(self.stages_completed)
        status = []
        for stage in all_stages:
            mark = "✓" if stage in completed else "○"
            status.append(f"{mark} {stage.replace('_', ' ').title()}")
        
        return "\n".join(status)


@dataclass
class ConversationBuffer:
    """Holds the full conversation for Claude context"""
    session_id: str
    turns: list = field(default_factory=list)
    agent_name: str = ""
    client_name: str = ""
    agency: str = ""
    extracted: ExtractedContext = field(default_factory=ExtractedContext)
    
    def add_turn(self, speaker: str, text: str):
        """Add a conversation turn"""
        self.turns.append({
            "speaker": speaker,
            "text": text,
            "timestamp": time.time()
        })
        # Keep last 80 turns max (longer calls need more context)
        if len(self.turns) > 80:
            self.turns = self.turns[-80:]
    
    def get_context(self, max_chars: int = 6000) -> str:
        """Get formatted conversation for Claude"""
        lines = []
        for turn in self.turns:
            speaker = "AGENT" if turn["speaker"] == "agent" else "CLIENT"
            lines.append(f"{speaker}: {turn['text']}")
        
        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[-max_chars:]
        return context
    
    def get_recent_context(self, num_turns: int = 6) -> str:
        """Get just the last few turns for quick analysis"""
        recent = self.turns[-num_turns:] if len(self.turns) >= num_turns else self.turns
        lines = []
        for turn in recent:
            speaker = "AGENT" if turn["speaker"] == "agent" else "CLIENT"
            lines.append(f"{speaker}: {turn['text']}")
        return "\n".join(lines)
    
    def get_last_agent_statement(self) -> str:
        """Get what the agent last said (for hesitation detection)"""
        for turn in reversed(self.turns):
            if turn["speaker"] == "agent":
                return turn["text"]
        return ""
    
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


# Globe Life presentation stages and their detection patterns
STAGE_PATTERNS = {
    "intro": ["globe life", "liberty national", "since 1900", "125 years", "policy holders"],
    "charity": ["make tomorrow better", "foundation", "7 million", "communities"],
    "sports": ["dallas cowboys", "texas rangers", "atlanta braves", "lakers", "globe life field"],
    "no_cost_offers": ["accidental death", "child safe kit", "will kit", "drug card", "$3,000", "no cost", "certificates"],
    "referral_collection": ["sponsor", "10 slots", "who would you like to sponsor", "who else", "referral"],
    "letter_of_thanks": ["letter of thanks", "yes or no", "take advantage"],
    "types_of_insurance": ["whole life", "term life", "owning a home", "renting", "permanent coverage", "final expenses"],
    "needs_analysis": ["date of birth", "beneficiary", "contingent beneficiary", "tobacco", "spouse information"],
    "health_questions": ["blood pressure", "diabetes", "cancer", "cardiovascular", "heart attack", "chest pain", "dui"],
    "family_health_history": ["runs in the family", "family history", "cancer in your family", "heart condition"],
    "income_protection": ["income protection", "two years of your income", "take home pay", "income replaced"],
    "mortgage_protection": ["mortgage protection", "pay off the balance", "pay off your home"],
    "college_education": ["college education", "child's ability to go to college", "education cost"],
    "recap_close": ["to recap", "option 1", "option 2", "which option", "works best for you"],
    "down_close": ["let's adjust", "reduce", "$15,000", "$10,000", "$7,500", "feeling better about that"]
}

# Critical stages that should NOT be skipped
CRITICAL_STAGES = ["referral_collection", "family_health_history"]


class ClaudeAnalyzer:
    """
    Elite guidance analyzer using Claude.
    
    Extracts context continuously from both speakers.
    Only fires guidance on high-value moments.
    """
    
    ANALYSIS_COOLDOWN = 3.0  # Seconds between analyses (slightly longer to reduce noise)
    MIN_CLIENT_WORDS = 4  # Minimum words to trigger analysis
    
    # The master prompt that makes Claude intelligent about when to fire
    SYSTEM_PROMPT = """You are an elite real-time sales coach for Globe Life insurance agents. You're listening to a live call and must make split-second decisions about when guidance is needed.

THE GLOBE LIFE PRESENTATION FLOW:
1. Rapport Building (F.O.R.M - Family, Occupation, Recreation, Me)
2. Company Intro (125 years, policy holders, "Make Tomorrow Better" charity, sports partnerships)
3. No-Cost Offers ($3k accidental death, child safe kit, will kit, drug card)
4. ⭐ REFERRAL COLLECTION (10 sponsor slots - THIS IS CRITICAL FOR GROWTH)
5. Letter of Thanks (yes or no decision)
6. Types of Insurance (Whole Life vs Term - owning vs renting analogy)
7. Needs Analysis (DOB, beneficiary, tobacco, spouse, kids)
8. Lifestyle Questions (rent/own, employment, income, existing coverage)
9. Health Questions (BP, diabetes, cancer, heart, DUIs)
10. ⭐ FAMILY HEALTH HISTORY (cancer/heart runs in family - ENABLES SUPPLEMENTAL UPSELL)
11. Coverage Presentation (Income Protection, Mortgage Protection, College Education)
12. Recap & Close (Option 1 vs Option 2)
13. Down-Close if needed (5 drop levels)

YOUR DECISION FRAMEWORK:

🔴 FIRE GUIDANCE when:
- Client OBJECTS: "too expensive", "can't afford", "need to think about it", "talk to spouse", "not interested", "already have insurance"
- Client HESITATES after tie-down: Agent asks "Does that make sense?" or "You see how important this is, right?" and client gives weak response like "I guess", "maybe", "I mean sure"
- Client shows DOUBT: "I'm not sure if...", "I don't know if we need...", "What if..."
- Client STALLS: "Can you call back", "Not a good time", "Send me something"
- BUYING SIGNAL needs capitalizing: "That sounds reasonable", "I like that", "What's the next step"

🟢 DO NOT FIRE when:
- Client ACKNOWLEDGES: "Yeah", "Okay", "Uh-huh", "I see", "Right", "Mmhmm"
- Client CONFIRMS info: "Yes I'm 34", "That's correct", "Right, we have two kids"
- Client asks CLARIFYING question: "Is that per month?", "So the whole life is permanent?", "And that covers my spouse?"
- Normal conversation flow
- Agent is handling the situation well already
- You JUST gave guidance (respect cooldown)

🟡 FIRE REMINDER when (different tone - not objection handling):
- Agent reaches pricing but SKIPPED referral collection
- Agent closes without asking about family health history
- Agent missed a critical value-building stage

HESITATION DETECTION:
After agent tie-down questions ("Does that make sense?", "You see how important this is, right?", "Fair enough?"), watch for:
- Weak affirmatives: "I guess", "I suppose", "Maybe", "If you say so"
- Filler-heavy responses: "Um... yeah... I mean..."
- Deflection: "That's interesting", "I'll have to think about that"
- Trailing off: "Well...", "I don't know..."
These need gentle value reinforcement, NOT hard objection handling.

CONTEXT USAGE:
When you DO fire guidance, make it UNCANNY by referencing:
- Specific numbers mentioned (coverage amounts, prices, income)
- Client's personal situation (age, family, spouse name if mentioned)
- What stage of the presentation they're in
- What products have been discussed

RESPONSE FORMAT - CRITICAL:
Your FIRST CHARACTER determines everything. No preamble, no reasoning, no analysis.

- If guidance needed: Start with "SAY:" then the EXACT words. 2-4 sentences. Conversational.
- If reminder needed: Start with "📋" then the nudge
- If no action needed: Output ONLY the text: NO_GUIDANCE_NEEDED

FORBIDDEN - NEVER OUTPUT THESE:
- Never start with "The client...", "Based on...", "Looking at...", "Since...", "Given..."
- Never explain your reasoning
- Never output thinking or analysis
- The agent sees your raw output IN REAL TIME - keep it clean

GOOD: SAY: "I completely understand, and that's exactly why this is so important..."
BAD: The client seems hesitant because they mentioned affordability earlier. SAY: "I understand..."

Be the coach that makes agents close deals they would have lost."""
    
    def __init__(self, session_id: str, agency: str = None, client_context: dict = None):
        self.session_id = session_id
        self.agency = agency
        self.client_context = client_context or {}
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
        self.last_analysis_time = 0
        self.is_analyzing = False
        self.guidance_count = 0
    
    def _format_prep_context(self) -> str:
        """Format Quick Prep context if agent filled it out"""
        if not self.client_context:
            return ""
        
        ctx = self.client_context
        parts = []
        
        product_map = {
            'whole_life': 'Whole Life Insurance',
            'term': 'Term Life Insurance', 
            'cancer': 'Cancer Policy',
            'accident': 'Accident Policy',
            'critical_illness': 'Critical Illness Policy'
        }
        if ctx.get('product'):
            parts.append(f"Product Interest: {product_map.get(ctx['product'], ctx['product'])}")
        if ctx.get('age'):
            parts.append(f"Age: {ctx['age']}")
        if ctx.get('sex'):
            parts.append(f"Sex: {'Male' if ctx['sex'] == 'M' else 'Female'}")
        if ctx.get('married'):
            parts.append(f"Married: {'Yes' if ctx['married'] == 'Y' else 'No'}")
        if ctx.get('kids') == 'Y':
            kids_str = "Has Kids: Yes"
            if ctx.get('kidsCount'):
                kids_str += f" ({ctx['kidsCount']})"
            parts.append(kids_str)
        if ctx.get('tobacco'):
            parts.append(f"Tobacco: {'Yes' if ctx['tobacco'] == 'Y' else 'No'}")
        if ctx.get('income'):
            parts.append(f"Income: {ctx['income']}")
        if ctx.get('budget'):
            parts.append(f"Budget: ${ctx['budget']}/mo")
        if ctx.get('issue'):
            parts.append(f"Notes: {ctx['issue']}")
        
        if not parts:
            return ""
        
        return "FROM AGENT'S PREP:\n" + "\n".join(f"• {p}" for p in parts)
    
    def _get_relevant_training(self, text: str) -> str:
        """Search training materials based on conversation"""
        try:
            db = get_vector_db()
            results = db.search(text, top_k=3, agency=self.agency)
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
    
    async def analyze_stream(self, client_text: str, conversation: ConversationBuffer):
        """
        Stream guidance token-by-token if needed.
        The prompt does the heavy lifting of deciding whether to fire.
        """
        if not await self.should_analyze(client_text):
            return
        
        self.is_analyzing = True
        self.last_analysis_time = time.time()
        
        try:
            # Build the analysis prompt
            full_context = conversation.get_context()
            recent_context = conversation.get_recent_context(8)
            last_agent = conversation.get_last_agent_statement()
            extracted = conversation.extracted.to_prompt_string()
            prep_context = self._format_prep_context()
            training = self._get_relevant_training(client_text)
            
            # Detect what stage we might be in based on recent conversation
            detected_stage = self._detect_stage(recent_context)
            stages_status = conversation.extracted.get_stages_status()
            
            user_prompt = f"""CONVERSATION SO FAR:
{full_context}

EXTRACTED CLIENT INFO:
{extracted}

{prep_context}

LAST FEW EXCHANGES:
{recent_context}

AGENT JUST SAID: "{last_agent}"
CLIENT JUST SAID: "{client_text}"

DETECTED STAGE: {detected_stage}

PRESENTATION PROGRESS:
{stages_status}

RELEVANT TRAINING:
{training if training else "No specific training retrieved"}

Based on what the client just said, do they need guidance? Remember your response format."""

            # Stream the response
            input_tokens = 0
            output_tokens = 0
            
            async with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            ) as stream:
                full_response = ""
                async for text in stream.text_stream:
                    full_response += text
                    
                    # Check if this is a NO_GUIDANCE response
                    if "NO_GUIDANCE" in full_response.upper():
                        return
                    
                    yield text
                
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
                    operation="elite_guidance"
                )
                self.guidance_count += 1
                
        except Exception as e:
            logger.error(f"Claude stream error: {e}")
        finally:
            self.is_analyzing = False
    
    def _detect_stage(self, recent_text: str) -> str:
        """Detect current presentation stage from recent conversation"""
        recent_lower = recent_text.lower()
        
        for stage, patterns in STAGE_PATTERNS.items():
            for pattern in patterns:
                if pattern in recent_lower:
                    return stage
        
        return "unknown"


class ContextExtractor:
    """
    Extracts context from agent speech to enrich guidance.
    Runs on agent transcripts, updates the shared ExtractedContext.
    """
    
    @staticmethod
    def extract_from_agent(text: str, context: ExtractedContext) -> None:
        """Extract context from what the agent says"""
        text_lower = text.lower()
        
        # Coverage amounts
        coverage_match = re.search(r'\$(\d{1,3}(?:,\d{3})*)\s*(?:of|in)?\s*(?:coverage|protection|whole life|term)', text, re.I)
        if coverage_match:
            amount = int(coverage_match.group(1).replace(',', ''))
            if amount >= 5000:  # Reasonable coverage amount
                context.coverage_amount = amount
        
        # Monthly price
        price_match = re.search(r'\$(\d+(?:\.\d{2})?)\s*(?:per|a|each)?\s*month', text, re.I)
        if price_match:
            context.monthly_price = int(float(price_match.group(1)))
        
        # Option presented
        if 'option 1' in text_lower:
            context.option_presented = "Option 1"
        elif 'option 2' in text_lower:
            context.option_presented = "Option 2"
        
        # Income mentioned
        income_match = re.search(r'(?:take home|income|make|earn).*?\$(\d{1,3}(?:,\d{3})*)', text, re.I)
        if income_match:
            income = int(income_match.group(1).replace(',', ''))
            if 1000 <= income <= 50000:  # Monthly income range
                context.income = income
        
        # Stage detection
        for stage, patterns in STAGE_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower and stage not in context.stages_completed:
                    context.stages_completed.append(stage)
                    context.current_stage = stage
                    break
        
        # Products discussed
        products = {
            'whole life': 'Whole Life',
            'term life': 'Term Life', 
            'term insurance': 'Term Life',
            'income protection': 'Income Protection',
            'mortgage protection': 'Mortgage Protection',
            'college education': 'College Education',
            'final expense': 'Final Expenses',
            'accidental death': 'Accidental Death',
            'cancer policy': 'Cancer Policy',
            'accident policy': 'Accident Policy'
        }
        for key, name in products.items():
            if key in text_lower and name not in context.products_discussed:
                context.products_discussed.append(name)
    
    @staticmethod
    def extract_from_client(text: str, context: ExtractedContext) -> None:
        """Extract context from what the client says"""
        text_lower = text.lower()
        
        # Age
        age_patterns = [
            r"i(?:'m| am)\s*(\d{2})",
            r"(\d{2})\s*years?\s*old",
            r"turn(?:ing|ed)?\s*(\d{2})",
            r"age\s*(?:is\s*)?(\d{2})"
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 18 <= age <= 85:
                    context.age = age
                    break
        
        # Marital status
        married_phrases = ['my wife', 'my husband', 'my spouse', "i'm married", 'we are married', "we're married"]
        for phrase in married_phrases:
            if phrase in text_lower:
                context.married = True
                break
        
        single_phrases = ["i'm single", 'not married', "i'm divorced", 'i am single']
        for phrase in single_phrases:
            if phrase in text_lower:
                context.married = False
                break
        
        # Kids
        kids_match = re.search(r'(?:have|got)\s*(\d+|one|two|three|four|five|six)\s*(?:kids?|children)', text_lower)
        if kids_match:
            context.has_kids = True
            num_words = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
            num = kids_match.group(1)
            context.num_kids = num_words.get(num) or int(num) if num.isdigit() else None
        elif any(phrase in text_lower for phrase in ['my kids', 'my children', 'our kids', 'our children', 'the kids']):
            context.has_kids = True
        
        # Budget mentioned
        budget_match = re.search(r"(?:afford|do|spend|budget|pay).*?\$(\d+)", text, re.I)
        if budget_match:
            budget = int(budget_match.group(1))
            if 10 <= budget <= 1000:
                context.budget = budget
        
        # Tobacco
        if any(phrase in text_lower for phrase in ['i smoke', 'i do smoke', 'yes i smoke', 'smoker']):
            context.tobacco = True
        elif any(phrase in text_lower for phrase in ["don't smoke", "don't use tobacco", "no tobacco", "non-smoker", "nonsmoker"]):
            context.tobacco = False
        
        # Health issues
        health_keywords = {
            'blood pressure': 'High Blood Pressure',
            'high blood pressure': 'High Blood Pressure',
            'diabetes': 'Diabetes',
            'diabetic': 'Diabetes',
            'cancer': 'Cancer History',
            'heart': 'Heart Issues',
            'cardiovascular': 'Heart Issues'
        }
        for keyword, condition in health_keywords.items():
            if keyword in text_lower and condition not in context.health_issues:
                # Check if they're saying they DON'T have it
                denial_patterns = [f"no {keyword}", f"don't have {keyword}", f"never had {keyword}"]
                is_denial = any(p in text_lower for p in denial_patterns)
                if not is_denial:
                    context.health_issues.append(condition)


class AgentStreamHandler:
    """
    Handles agent audio stream.
    Transcribes via Deepgram, extracts context, tracks presentation stage.
    Does NOT display to agent (they know what they're saying).
    """
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self._connection_dead = False  # Prevents log spam on disconnect
        self._deepgram_initialized = False  # Lazy init on stream start
        self.conversation = get_conversation_buffer(session_id)
        
        self._total_audio_bytes = 0
        self._session_start_time = None
        
        logger.info(f"[AgentStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize handler - Deepgram connects lazily on stream start"""
        print(f"[AgentStream] Starting for {self.session_id}", flush=True)
        self._session_start_time = time.time()
        self.is_running = True
        self._deepgram_initialized = False
        return True
    
    async def connect_deepgram(self):
        """Connect to Deepgram - called by ClientStreamHandler after client connects"""
        if self._deepgram_initialized or not settings.deepgram_api_key:
            return
        
        self._deepgram_initialized = True
        
        # Retry logic - Deepgram sometimes rejects connections
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
                    interim_results=False,  # Only final for agent (context extraction)
                    utterance_end_ms=1500,
                    encoding="linear16",
                    sample_rate=self.SAMPLE_RATE,
                    channels=1
                )
                
                await self.connection.start(options)
                print(f"[AgentStream] Deepgram connected (attempt {attempt + 1})", flush=True)
                return
                
            except Exception as e:
                print(f"[AgentStream] Deepgram attempt {attempt + 1} failed: {e}", flush=True)
                self.connection = None
                self.deepgram = None
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
        
        print(f"[AgentStream] Deepgram failed after {max_retries} attempts - continuing without agent transcription", flush=True)
    
    async def handle_telnyx_message(self, message: dict):
        """Process Telnyx message with agent audio"""
        event = message.get("event")
        
        if event == "connected":
            print(f"[AgentStream] Telnyx connected", flush=True)
        elif event == "start":
            print(f"[AgentStream] Stream started - awaiting client trigger", flush=True)
            # NOTE: Deepgram connects AFTER client connects (see ClientStreamHandler._trigger_agent_deepgram)
        elif event == "media":
            media = message.get("media", {})
            payload = media.get("payload")
            
            if payload:
                try:
                    ulaw_audio = base64.b64decode(payload)
                    self._total_audio_bytes += len(ulaw_audio)
                    pcm_audio = audioop.ulaw2lin(ulaw_audio, 2)
                    
                    if self.connection and self.is_running and not self._connection_dead:
                        await self.connection.send(pcm_audio)
                except Exception as e:
                    # Only log first error, then mark connection as dead
                    if not self._connection_dead:
                        self._connection_dead = True
                        print(f"[AgentStream] Deepgram disconnected: {str(e)[:80]}", flush=True)
        elif event == "stop":
            await self.stop()
    
    async def _on_open(self, *args, **kwargs):
        print(f"[AgentStream] Deepgram open", flush=True)
    
    async def _on_transcript(self, *args, **kwargs):
        """Handle agent transcript - extract context, track stage"""
        try:
            result = kwargs.get('result') or (args[1] if len(args) > 1 else None)
            if not result:
                return
            
            alternatives = result.channel.alternatives
            if not alternatives:
                return
            
            transcript = alternatives[0].transcript
            is_final = result.is_final
            
            if not transcript or not is_final:
                return
            
            print(f"[AgentStream] AGENT: {transcript[:60]}", flush=True)
            
            # Add to conversation buffer
            self.conversation.add_turn("agent", transcript)
            
            # Extract context from agent speech
            ContextExtractor.extract_from_agent(transcript, self.conversation.extracted)
            
            # Send to frontend for transcript display
            await self._broadcast({
                "type": "agent_transcript",
                "text": transcript,
                "is_final": True
            })
            
        except Exception as e:
            logger.error(f"[AgentStream] Transcript error: {e}")
    
    async def _broadcast(self, message: dict):
        """Send to frontend"""
        await session_manager._broadcast_to_session(self.session_id, message)
    
    async def _on_error(self, *args, **kwargs):
        error = kwargs.get('error') or (args[1] if len(args) > 1 else "Unknown")
        # Only log first error
        if not self._connection_dead:
            self._connection_dead = True
            print(f"[AgentStream] Audio error: {str(error)[:80]}", flush=True)
    
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
    Handles client audio stream.
    Transcribes via Deepgram, displays live, triggers Claude analysis and sentiment updates.
    """
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self._connection_dead = False  # Prevents log spam on disconnect
        self.conversation = get_conversation_buffer(session_id)
        
        self._total_audio_bytes = 0
        self._session_start_time = None
        self._client_buffer = ""
        
        # Claude analyzer
        self._analyzer = None
        self._generating_guidance = False
        
        # Sentiment analyzer
        self._sentiment = None
        
        logger.info(f"[ClientStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize Deepgram for client audio with retry logic"""
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
        
        # Initialize sentiment analyzer
        self._sentiment = SentimentAnalyzer(self.session_id)
        
        # Send initial sentiment state (neutral)
        await self._broadcast(self._sentiment.get_current_state())
        
        if not settings.deepgram_api_key:
            logger.warning("[ClientStream] No Deepgram key")
            self.is_running = True
            await self._broadcast({"type": "ready"})
            return True
        
        # Retry logic - Deepgram sometimes rejects connections
        max_retries = 3
        retry_delay = 0.5  # seconds
        
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
                    interim_results=True,  # Show live typing for client
                    utterance_end_ms=1000,
                    encoding="linear16",
                    sample_rate=self.SAMPLE_RATE,
                    channels=1
                )
                
                await self.connection.start(options)
                self.is_running = True
                
                print(f"[ClientStream] Deepgram connected (attempt {attempt + 1})", flush=True)
                
                # NOW trigger agent's Deepgram connection (sequenced to avoid HTTP 400)
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
                    retry_delay *= 2  # Exponential backoff
        
        # All retries failed - continue without client transcription
        print(f"[ClientStream] Deepgram failed after {max_retries} attempts - continuing without transcription", flush=True)
        self.is_running = True
        await self._broadcast({"type": "ready"})
        return True
    
    async def _trigger_agent_deepgram(self):
        """Trigger agent's Deepgram connection after client is connected.
        
        CRITICAL: Must wait before connecting to avoid Deepgram HTTP 400 rate limit.
        Deepgram rejects rapid successive WebSocket connections from the same API key.
        """
        try:
            agent_handler = _agent_handlers.get(self.session_id)
            if agent_handler:
                # CRITICAL: Wait for client connection to fully establish
                # Deepgram rate-limits concurrent WebSocket connections
                await asyncio.sleep(1.0)  # 1 second delay prevents HTTP 400
                
                print(f"[ClientStream] Triggering Agent Deepgram...", flush=True)
                await agent_handler.connect_deepgram()
            else:
                print(f"[ClientStream] No agent handler found for {self.session_id}", flush=True)
        except Exception as e:
            print(f"[ClientStream] Failed to trigger agent Deepgram: {e}", flush=True)
    
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
                    
                    if self.connection and self.is_running and not self._connection_dead:
                        await self.connection.send(pcm_audio)
                except Exception as e:
                    # Only log first error, then mark connection as dead
                    if not self._connection_dead:
                        self._connection_dead = True
                        print(f"[ClientStream] Deepgram disconnected: {str(e)[:80]}", flush=True)
        elif event == "stop":
            await self.stop()
    
    async def _on_open(self, *args, **kwargs):
        print(f"[ClientStream] Deepgram open", flush=True)
    
    async def _on_transcript(self, *args, **kwargs):
        """Handle client transcript - display, analyze sentiment, and trigger guidance"""
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
                
                # Extract context from client speech
                ContextExtractor.extract_from_client(transcript, self.conversation.extracted)
                
                # Analyze sentiment (fast, pattern-based)
                if self._sentiment:
                    current_stage = self.conversation.extracted.current_stage
                    sentiment_update = self._sentiment.analyze(
                        transcript, 
                        speaker="client", 
                        stage=current_stage
                    )
                    if sentiment_update:
                        await self._broadcast(sentiment_update)
                
                # Trigger Claude analysis (may fire guidance)
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
                # Determine if this was a reminder vs objection guidance
                is_reminder = full_guidance.strip().startswith("📋")
                
                await self._broadcast({
                    "type": "guidance_complete",
                    "full_text": full_guidance,
                    "is_reminder": is_reminder
                })
                
                # Store in session
                await session_manager.add_guidance(self.session_id, {
                    "trigger": client_text[:50],
                    "response": full_guidance,
                    "is_reminder": is_reminder,
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
        # Only log first error
        if not self._connection_dead:
            self._connection_dead = True
            print(f"[ClientStream] Audio error: {str(error)[:80]}", flush=True)
    
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
    """Get or create client stream handler - automatically initializes Deepgram"""
    if session_id not in _client_handlers:
        handler = ClientStreamHandler(session_id)
        await handler.start()  # Always initialize Deepgram connection
        _client_handlers[session_id] = handler
    return _client_handlers[session_id]


async def get_or_create_agent_handler(session_id: str) -> AgentStreamHandler:
    """Get or create agent stream handler - automatically initializes Deepgram"""
    if session_id not in _agent_handlers:
        handler = AgentStreamHandler(session_id)
        await handler.start()  # Always initialize Deepgram connection
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
    """Remove a specific handler (for backward compatibility with main.py)"""
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
    from deepgram import DeepgramClient
    return DeepgramClient(api_key=settings.deepgram_api_key)