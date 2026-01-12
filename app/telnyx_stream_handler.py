"""
Coachd Telnyx Stream Handler - Elite Guidance System
=====================================================
ARCHITECTURE:
- Agent stream: Deepgram → context extraction + stage tracking (silent)
- Client stream: Deepgram → objection/hesitation detection → guidance

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


# Shared DeepgramClient instance - SDK may have issues with multiple clients
_deepgram_client: Optional[DeepgramClient] = None

def get_deepgram_client() -> Optional[DeepgramClient]:
    """Get or create shared Deepgram client"""
    global _deepgram_client
    if _deepgram_client is None and settings.deepgram_api_key:
        _deepgram_client = DeepgramClient(settings.deepgram_api_key)
        logger.info("[Deepgram] Shared client created")
    return _deepgram_client


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

=== CRITICAL OUTPUT RULES ===

YOUR OUTPUT IS DISPLAYED DIRECTLY ON THE AGENT'S SCREEN AS A TELEPROMPTER.
The agent will READ YOUR WORDS OUT LOUD to the client.

✅ CORRECT OUTPUT (exact script to read):
"I completely understand wanting to talk to your wife - that shows you're a thoughtful husband. Here's what we can do: let's lock in this rate today while you're still healthy, and you have 30 days to cancel with a full refund if she's not on board."

❌ WRONG - Never echo client speech:
"I don't know if life insurance is for me, bro."

❌ WRONG - Never give commentary/observations:
"This call has gone downhill." 
"The client seems resistant."
"Things are getting heated."

❌ WRONG - Never give meta-instructions:
"Try to calm them down."
"You should apologize."
"Redirect the conversation."

ONLY OUTPUT:
1. The EXACT words for the agent to say (in quotes, as a script) - OR -
2. "📋 REMINDER:" followed by a brief nudge - OR -
3. NO_GUIDANCE_NEEDED

If the conversation goes off the rails (cursing, hostility), give the agent WORDS TO SAY to recover:
"Sir, I apologize if I came across wrong - I'm just passionate about making sure families are protected. Can we start fresh?"

NOT commentary about the situation.

Be the coach that makes agents close deals they would have lost."""
    
    def __init__(self, session_id: str, agency: str = None, client_context: dict = None):
        self.session_id = session_id
        self.agency = agency
        self.client_context = client_context or {}
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None
        self.last_analysis_time = 0
        self.is_analyzing = False
        self.guidance_count = 0
        # Track active guidance for smart dismiss
        self.active_guidance = None  # {"text": str, "trigger": str, "time": float}
        self.is_checking_stale = False
    
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
            current_stage = self._detect_stage(recent_context)
            stages_status = conversation.extracted.get_stages_status()
            
            user_prompt = f"""CONVERSATION CONTEXT:
{full_context}

---

EXTRACTED CLIENT INFO:
{extracted}

{prep_context}

PRESENTATION PROGRESS:
{stages_status}
Current Stage: {current_stage}

AGENT JUST SAID: "{last_agent}"

CLIENT JUST SAID: "{client_text}"

RELEVANT TRAINING:
{training[:2000] if training else "Use Globe Life standard techniques."}

---

Analyze this moment. Does the agent need guidance RIGHT NOW?

Remember:
- Only fire on objections, hesitations, or missed critical stages
- NOT on acknowledgments, confirmations, or clarifying questions
- Make guidance specific using the extracted context
- Respect the presentation flow
- OUTPUT ONLY: exact script in quotes, 📋 REMINDER, or NO_GUIDANCE_NEEDED"""

            collected_text = ""
            input_tokens = 0
            output_tokens = 0
            first_check_done = False
            
            async with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    collected_text += text
                    
                    # Check early if it's NO_GUIDANCE_NEEDED
                    if "NO_GUIDANCE" in collected_text.upper():
                        return
                    
                    # Safety check after first ~30 chars
                    if not first_check_done and len(collected_text) > 30:
                        first_check_done = True
                        trimmed = collected_text.strip()
                        
                        # Check if echoing client speech (first 20 chars match)
                        if client_text and len(client_text) > 10:
                            client_start = client_text[:20].lower().strip()
                            output_start = trimmed[:20].lower()
                            if client_start and output_start.startswith(client_start[:15]):
                                print(f"[Claude] Filtering: echoing client speech", flush=True)
                                return
                        
                        # Check if it's commentary (doesn't start with quote or 📋)
                        # Valid starts: " ' " 📋 "Listen "Sir "I "Look "Here's "Let "That's "You "What "Mr "Mrs
                        valid_starts = ('"', "'", '"', '📋', '"', 'Listen', 'Sir', 'I ', 'Look', "Here's", 'Let', "That's", 'You', 'What', 'Mr', 'Mrs', 'Ma\'am', 'Hey')
                        if not any(trimmed.startswith(s) for s in valid_starts):
                            # Could be commentary - check for red flag words
                            commentary_flags = ['this call', 'the client', 'seems', 'appears', 'getting', 'gone', 'has become', 'situation']
                            if any(flag in trimmed.lower() for flag in commentary_flags):
                                print(f"[Claude] Filtering: commentary detected", flush=True)
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
    
    def set_active_guidance(self, guidance_text: str, trigger: str):
        """Mark guidance as active for stale checking"""
        self.active_guidance = {
            "text": guidance_text,
            "trigger": trigger,
            "time": time.time()
        }
    
    def clear_active_guidance(self):
        """Clear active guidance (dismissed)"""
        self.active_guidance = None
    
    async def check_guidance_stale(self, recent_transcript: str, conversation: ConversationBuffer) -> bool:
        """
        Check if the active guidance is now stale and should be dismissed.
        Returns True if guidance should be dismissed.
        
        Called after each transcript when guidance is active.
        """
        if not self.active_guidance or not self.client:
            return False
        
        if self.is_checking_stale:
            return False
        
        # Don't check too soon after guidance was given (give agent time to read/use it)
        if time.time() - self.active_guidance["time"] < 5.0:
            return False
        
        self.is_checking_stale = True
        
        try:
            recent_context = conversation.get_recent_context(4)
            
            check_prompt = f"""You are monitoring a live sales call. Guidance was shown to the agent.

GUIDANCE THAT WAS SHOWN:
"{self.active_guidance['text'][:200]}"

TRIGGERED BY CLIENT SAYING:
"{self.active_guidance['trigger']}"

WHAT HAS BEEN SAID SINCE:
{recent_context}

MOST RECENT UTTERANCE:
"{recent_transcript}"

Is this guidance now STALE and should be dismissed? Answer YES if:
- The agent delivered the rebuttal or something similar
- The conversation has moved past this objection
- The client has responded and topic changed
- 3+ turns have passed and the moment is over

Answer NO if:
- The objection is still unresolved
- Agent hasn't addressed it yet
- Still relevant to current discussion

Respond with exactly one word: YES or NO"""

            response = await self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": check_prompt}]
            )
            
            result = response.content[0].text.strip().upper()
            should_dismiss = result.startswith("YES")
            
            if should_dismiss:
                print(f"[SmartDismiss] Guidance stale - dismissing", flush=True)
                self.active_guidance = None
            
            return should_dismiss
            
        except Exception as e:
            logger.error(f"Stale check error: {e}")
            return False
        finally:
            self.is_checking_stale = False


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
        import re
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
        
        import re
        
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
        if any(phrase in text_lower for phrase in ['i smoke', 'i do smoke', 'yes tobacco', 'i use tobacco', 'i chew']):
            context.tobacco = True
        elif any(phrase in text_lower for phrase in ["don't smoke", "no tobacco", "i don't smoke", "non-smoker"]):
            context.tobacco = False
        
        # Health issues mentioned
        health_keywords = {
            'blood pressure': 'High BP',
            'high blood pressure': 'High BP',
            'diabetes': 'Diabetes',
            'diabetic': 'Diabetes',
            'heart': 'Heart Condition',
            'cancer': 'Cancer History',
            'stroke': 'Stroke History'
        }
        for keyword, label in health_keywords.items():
            if keyword in text_lower and label not in context.health_issues:
                context.health_issues.append(label)
        
        # Family health history
        if 'runs in' in text_lower or 'family history' in text_lower:
            if 'cancer' in text_lower and 'Cancer' not in context.family_health_history:
                context.family_health_history.append('Cancer')
            if any(h in text_lower for h in ['heart', 'cardiac', 'cardiovascular']):
                if 'Heart Disease' not in context.family_health_history:
                    context.family_health_history.append('Heart Disease')
        
        # Names (simple extraction)
        name_match = re.search(r"(?:i'm|i am|my name is|this is)\s+([A-Z][a-z]+)", text)
        if name_match and not context.client_name:
            context.client_name = name_match.group(1)
        
        spouse_match = re.search(r"(?:my (?:wife|husband|spouse)(?:'s name is|,?\s+)?)\s*([A-Z][a-z]+)", text)
        if spouse_match:
            context.spouse_name = spouse_match.group(1)


class AgentStreamHandler:
    """
    Handles AGENT audio stream.
    Transcribes via Deepgram, extracts context, tracks stages.
    NOT shown to agent (they know what they said).
    """
    
    SAMPLE_RATE = 8000
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.deepgram = None
        self.connection = None
        self.is_running = False
        self._deepgram_connected = False  # Track if Deepgram actually connected
        self.conversation = get_conversation_buffer(session_id)
        
        self._total_audio_bytes = 0
        self._session_start_time = None
        
        logger.info(f"[AgentStream] Created for session {session_id}")
    
    async def start(self) -> bool:
        """Initialize handler - Deepgram connects later via connect_deepgram()"""
        print(f"[AgentStream] Starting for {self.session_id}", flush=True)
        self._session_start_time = time.time()
        self.is_running = True
        # Don't connect Deepgram yet - wait until called explicitly
        # This avoids the HTTP 400 that happens when Agent connects before Client
        return True
    
    async def connect_deepgram(self) -> bool:
        """Connect to Deepgram - called AFTER client stream is established"""
        if self._deepgram_connected:
            return True
            
        self.deepgram = get_deepgram_client()
        if not self.deepgram:
            logger.warning("[AgentStream] No Deepgram client available")
            return False
        
        try:
            print(f"[AgentStream] Connecting to Deepgram now...", flush=True)
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
            self._deepgram_connected = True
            
            print(f"[AgentStream] Deepgram connected!", flush=True)
            return True
            
        except Exception as e:
            logger.error(f"[AgentStream] Deepgram error: {e}")
            print(f"[AgentStream] Deepgram FAILED: {e}", flush=True)
            return False
    
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
                    
                    if self.connection and self._deepgram_connected:
                        await self.connection.send(pcm_audio)
                except Exception as e:
                    logger.error(f"[AgentStream] Audio error: {e}")
        elif event == "stop":
            await self.stop()
    
    async def _on_open(self, *args, **kwargs):
        print(f"[AgentStream] Deepgram open", flush=True)
    
    async def _on_transcript(self, *args, **kwargs):
        """Handle agent transcript - extract context, broadcast to frontend"""
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
            
            # Add to conversation buffer
            self.conversation.add_turn("agent", transcript)
            
            # Extract context from agent speech
            ContextExtractor.extract_from_agent(transcript, self.conversation.extracted)
            
            # Broadcast to frontend for transcript sidebar
            await session_manager._broadcast_to_session(self.session_id, {
                "type": "agent_transcript",
                "text": transcript,
                "is_final": True
            })
            
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
        self._client_buffer = ""
        
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
        
        # Use shared Deepgram client
        self.deepgram = get_deepgram_client()
        if not self.deepgram:
            logger.warning("[ClientStream] No Deepgram client available")
            self.is_running = True
            await self._broadcast({"type": "ready"})
            return True
        
        try:
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
            
            # Now trigger Agent's Deepgram connection (delayed to avoid HTTP 400)
            await self._trigger_agent_deepgram()
            
            await self._broadcast({
                "type": "ready",
                "message": "Coaching active"
            })
            
            return True
            
        except Exception as e:
            logger.error(f"[ClientStream] Deepgram error: {e}")
            self.is_running = True
            return True
    
    async def _trigger_agent_deepgram(self):
        """Trigger Agent's Deepgram connection after Client is established"""
        try:
            if self.session_id in _agent_handlers:
                agent_handler = _agent_handlers[self.session_id]
                print(f"[ClientStream] Triggering Agent Deepgram connection...", flush=True)
                await agent_handler.connect_deepgram()
            else:
                print(f"[ClientStream] No agent handler found for {self.session_id}", flush=True)
        except Exception as e:
            logger.error(f"[ClientStream] Failed to trigger agent Deepgram: {e}")
    
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
                
                # Extract context from client speech
                ContextExtractor.extract_from_client(transcript, self.conversation.extracted)
                
                # Check if active guidance is now stale (smart dismiss)
                if self._analyzer and self._analyzer.active_guidance:
                    asyncio.create_task(self._check_stale_guidance(transcript))
                
                # Trigger Claude analysis for new guidance
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
                
                # Track active guidance for smart dismiss
                self._analyzer.set_active_guidance(full_guidance, client_text[:50])
                
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
    
    async def _check_stale_guidance(self, recent_transcript: str):
        """Check if active guidance is stale and should be dismissed"""
        try:
            should_dismiss = await self._analyzer.check_guidance_stale(
                recent_transcript, 
                self.conversation
            )
            if should_dismiss:
                await self._broadcast({"type": "guidance_dismiss"})
        except Exception as e:
            logger.error(f"[ClientStream] Stale check error: {e}")
    
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