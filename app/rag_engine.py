"""
Coachd RAG Engine
Retrieval-Augmented Generation for sales guidance

V2: Full call state awareness for intelligent, contextual guidance
- Knows what products have been tried/rejected
- Tracks objection history and patterns  
- Suggests down-closes intelligently
- Recognizes when to gracefully exit

With complete usage tracking for billing accuracy.
"""

from anthropic import Anthropic
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass

from .vector_db import get_vector_db
from .config import settings
from .usage_tracker import log_claude_usage


# Product hierarchy for down-closing
PRODUCT_HIERARCHY = [
    "whole_life",
    "term_life", 
    "cancer",
    "accident",
    "critical_illness"
]

PRODUCT_NAMES = {
    "whole_life": "Whole Life Insurance",
    "term_life": "Term Life Insurance",
    "cancer": "Cancer Insurance",
    "accident": "Accident Insurance",
    "critical_illness": "Critical Illness / Hospital ICU"
}


@dataclass
class CallContext:
    """Legacy context - kept for backward compatibility"""
    current_product: str = "whole_life"
    client_age: Optional[int] = None
    client_occupation: Optional[str] = None
    client_family: Optional[str] = None
    products_tried: List[str] = None
    objections_faced: List[str] = None
    client_sentiment: str = "neutral"
    call_type: str = "phone"  # "phone" or "presentation"
    
    def __post_init__(self):
        if self.products_tried is None:
            self.products_tried = []
        if self.objections_faced is None:
            self.objections_faced = []


class RAGEngine:
    """Retrieval-Augmented Generation engine for sales guidance"""
    
    def __init__(self):
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.vector_db = get_vector_db()
        
    def get_relevant_context(
        self, 
        query: str, 
        category: Optional[str] = None,
        top_k: int = 5,
        agency: Optional[str] = None
    ) -> str:
        """Retrieve relevant training materials for a query from agency's collection"""
        results = self.vector_db.search(
            query, 
            top_k=top_k, 
            category=category,
            agency=agency
        )
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result['metadata'].get('filename', 'Unknown')}]\n"
                f"{result['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def detect_price_objection(self, transcript: str) -> bool:
        """Quick check if transcript contains price-related objection."""
        price_keywords = [
            "can't afford", "cannot afford", "too expensive", "too much",
            "don't have the money", "tight budget", "not in my budget",
            "that's a lot", "costs too much", "price is too high",
            "where am i going to get", "how am i supposed to pay"
        ]
        lower = transcript.lower()
        return any(kw in lower for kw in price_keywords)
    
    # ============ V2: FULL STATE-AWARE GUIDANCE ============
    
    def build_system_prompt_v2(self, call_state: Dict[str, Any]) -> str:
        """
        Build comprehensive system prompt from full call state.
        
        Globe Life specific - handles both phone calls (setting appointments)
        and full presentations (closing sales).
        """
        
        # Extract state components
        call_type = call_state.get("call_type", "presentation")
        is_phone = call_state.get("is_phone_call", False)
        is_presentation = call_state.get("is_presentation", True)
        phone_context = call_state.get("phone_context")
        presentation_context = call_state.get("presentation_context")
        
        client_profile = call_state.get("client_profile", "Unknown")
        sentiment = call_state.get("client_sentiment", "neutral")
        
        # Coverage and down-close (presentation only)
        coverage_summary = call_state.get("coverage_summary", "")
        down_close_level = call_state.get("down_close_level", 0)
        down_close_exhausted = call_state.get("down_close_exhausted", False)
        
        current_product = call_state.get("current_product", "Whole Life Insurance")
        hierarchy_position = call_state.get("hierarchy_position", "1 of 5")
        products_rejected = call_state.get("products_rejected", [])
        products_remaining = call_state.get("products_remaining", [])
        hierarchy_exhausted = call_state.get("hierarchy_exhausted", False)
        
        active_objection = call_state.get("active_objection")
        objection_history = call_state.get("objection_history", [])
        objection_pattern = call_state.get("objection_pattern")
        objection_queue_size = call_state.get("objection_queue_size", 0)
        
        recent_transcript = call_state.get("recent_transcript", "")
        key_moments = call_state.get("key_moments", [])
        recent_guidance = call_state.get("recent_guidance", [])
        
        viability_score = call_state.get("viability_score", 50)
        viability_rec = call_state.get("viability_recommendation", "keep_pushing")
        buying_signals = call_state.get("buying_signals_detected", [])
        hard_nos = call_state.get("hard_nos_detected", [])
        
        call_duration = call_state.get("call_duration", "0:00")
        referrals_collected = call_state.get("referrals_collected", 0)
        
        # Build formatted sections
        objection_str = ", ".join(objection_history[-5:]) if objection_history else "None yet"
        moments_str = "\n".join([f"  - {m}" for m in key_moments[-5:]]) if key_moments else "None recorded"
        guidance_str = "\n".join([f"  - {g[:80]}..." for g in recent_guidance[-3:]]) if recent_guidance else "None yet"
        
        # ============ PHONE CALL PROMPT ============
        if is_phone:
            return f"""You are Coachd, an elite real-time coach for Globe Life Liberty National agents.
This is a PHONE CALL to SET AN APPOINTMENT - NOT a sales presentation.

═══════════════════════════════════════════════════════════════
                     PHONE CALL STATE
═══════════════════════════════════════════════════════════════

CALL INFO:
- Duration: {call_duration}
- Goal: Set appointment with BOTH spouses present
- Typical Duration: 2-3 minutes (be efficient!)

CLIENT: {client_profile}
SENTIMENT: {sentiment.upper()}

ACTIVE OBJECTION: {active_objection or 'None'}
OBJECTION HISTORY: {objection_str}

KEY MOMENTS:
{moments_str}

═══════════════════════════════════════════════════════════════
                  PHONE OBJECTION REBUTTALS
═══════════════════════════════════════════════════════════════

"HOW LONG IS THIS GOING TO TAKE?"
→ "I have a lot of people on my schedule to get this out to today, so it doesn't take long - for me the quicker, the better..."

"WHY DOES MY SPOUSE HAVE TO BE THERE?"
→ "They ask us to make sure we speak with both of you to answer any questions and make sure you know how this works in case something happens to either of you..."

"JUST MAIL IT TO ME"
→ "I sure wish we could. That would make my job a lot easier. However, we need to speak with you to designate your beneficiaries for the no-cost coverage and review everything you're entitled to through the program."

"NOT INTERESTED"
→ "Well that's OK, but you're still entitled to the no-cost coverage. I can still get that out to you and explain everything you're entitled to through the program. Most people appreciate the short time it takes to explain everything. I'll tell you what I'm going to do..."

"IS SOMEONE GOING TO TRY AND SELL ME SOMETHING?"
→ "It's our job to review your policy with you and renew your no-cost coverage. We have an appointment every half hour so it doesn't take very much of your time. I'll tell you what I'm going to do..."

═══════════════════════════════════════════════════════════════
                      YOUR ROLE
═══════════════════════════════════════════════════════════════

PHONE CALL SEQUENCE:
1. Introduction: "This is [name] with Globe Life Liberty National, your insurance company"
2. Purpose: "Home Office asked that we touch base... verify information... explain options"
3. Questions: "What time are you home from work? What time is your spouse home?"
4. Confirmation: "I'll call you and your spouse tomorrow between ___ and ___"
5. Solidify: "Go ahead and grab a pen... My name is ___ and the confirmation # is ___"
6. Lock it: "Could anything prevent us from speaking on [day] at [time]?"

YOUR GUIDANCE SHOULD:
1. Be BRIEF - this is a 2-3 minute call
2. Help overcome the CURRENT objection to SET THE APPOINTMENT
3. Always push toward getting BOTH SPOUSES on the next call
4. Include EXACT WORDS to say

RESPONSE FORMAT:
"[Exact words to say]"
Why: [1 sentence explanation]

DO NOT: Give long rebuttals, suggest selling anything, or forget the goal is APPOINTMENT SETTING."""
        
        # ============ PRESENTATION PROMPT ============
        
        # Presentation phase info
        phase_info = ""
        if presentation_context:
            current_phase = presentation_context.get("current_phase", "Unknown")
            phase_position = presentation_context.get("phase_position", "1 of 14")
            phase_info = f"""
PRESENTATION PHASE: {current_phase} ({phase_position})
COVERAGE OFFERED: {coverage_summary}
DOWN-CLOSE LEVEL: {down_close_level} ({"EXHAUSTED" if down_close_exhausted else "options remaining"})
REFERRALS COLLECTED: {referrals_collected}"""
        
        # Build rejection summary
        rejection_str = "None yet"
        if products_rejected:
            rejection_str = "\n".join([f"  - {r}" for r in products_rejected])
        
        # Determine strategic context
        strategic_note = ""
        
        if down_close_exhausted and hierarchy_exhausted:
            strategic_note = """
⚠️ CRITICAL: ALL OPTIONS EXHAUSTED
Both down-close levels AND product hierarchy exhausted. Focus on:
1. Graceful exit that preserves future opportunity
2. Schedule a callback for when circumstances change  
3. Ask for referrals - "Who do you know who might benefit from this?"
"""
        elif hard_nos:
            strategic_note = """
⚠️ CLIENT HAS INDICATED HARD NO
Do not continue pushing. Provide a professional exit:
- "I appreciate your time today"
- "If anything changes, please reach out"
- Exit promptly and respectfully
"""
        elif viability_rec == "graceful_exit":
            strategic_note = f"""
📉 VIABILITY LOW ({viability_score}/100) - Consider graceful exit
Options:
1. One final down-close attempt
2. "Let's schedule a callback when timing is better"
3. "Who do you know who might benefit from this coverage?"
"""
        elif objection_pattern == "serial_price_objection" or (active_objection == "price" and down_close_level < 3):
            strategic_note = f"""
💰 PRICE OBJECTION - INITIATE DOWN-CLOSE
Current Coverage: {coverage_summary}
Down-Close Level: {down_close_level}

DOWN-CLOSE SEQUENCE (Globe Life):
1. Option 1 → Option 2 (same coverage, different timing)
2. $30k Final Expense → $15k ("cover cost of funeral")
3. $15k → $10k ("right in the middle of current costs")
4. $10k → $7.5k ("basic cost of a funeral is approximately $7,500")
5. 2 years income → 1 year ("when things get better, you can increase")
6. "Which benefit could you go without for now?"

SAY: "Ok, I understand. Let me ask you, you understand how important this is right? So here's what we can do: let's adjust the Final Expense benefit to $15,000..."
"""
        elif active_objection == "stall":
            strategic_note = """
⏸️ "THINK ABOUT IT" OBJECTION
This usually comes down to: Do they NEED it? Or can they AFFORD it?

SAY: "I completely understand, after all this is a big decision for your family. When folks think about it, it usually comes down to one or two things: either do they need this, or can they afford this? So let me ask you, do you feel this is something your family needs?"
"""
        elif active_objection == "covered":
            strategic_note = """
🛡️ "ALREADY HAVE INSURANCE" OBJECTION
The Needs Analysis already accounted for existing coverage!

SAY: "Yes, and the Needs Analysis has already taken that into consideration, and it has clearly exposed the need for more coverage. Simply put, what you currently have is good, but it's just not enough to make sure your family is protected."
"""
        elif buying_signals:
            strategic_note = f"""
✅ BUYING SIGNALS DETECTED: {', '.join(buying_signals)}
Client is showing interest! Move toward the close:

SAY: "So to recap, we have your Final Expenses taken care of, your income is protected, your mortgage is covered, and your child's education is protected. The only question I have is: which option works best for you - Option 1 or Option 2?"
"""
        
        # Products remaining note
        if len(products_remaining) == 1:
            remaining_note = f"Only 1 product remaining: {products_remaining[0]}"
        elif len(products_remaining) == 0:
            remaining_note = "NO PRODUCTS REMAINING - exit gracefully"
        else:
            remaining_note = f"{len(products_remaining)} products remaining: {', '.join(products_remaining[:3])}"
        
        return f"""You are Coachd, an elite real-time sales coach for Globe Life Liberty National agents.
You provide instant, actionable guidance during live PRESENTATION calls.

═══════════════════════════════════════════════════════════════
                    PRESENTATION STATE
═══════════════════════════════════════════════════════════════

CALL INFO:
- Duration: {call_duration}
- Client Sentiment: {sentiment.upper()}
- Viability Score: {viability_score}/100 → {viability_rec.replace('_', ' ').upper()}
{phase_info}

CLIENT PROFILE:
{client_profile}

PRODUCT STATE:
- Currently Pitching: {current_product}
- {remaining_note}
- Rejected Products:
{rejection_str}

OBJECTION STATE:
- Active Objection: {active_objection or 'None'}
- History: {objection_str}
- Pattern: {objection_pattern or 'None'}
{f'- ⚠️ {objection_queue_size} more objections queued!' if objection_queue_size > 0 else ''}

KEY MOMENTS:
{moments_str}

GUIDANCE GIVEN:
{guidance_str}

═══════════════════════════════════════════════════════════════
                     STRATEGIC CONTEXT
═══════════════════════════════════════════════════════════════
{strategic_note if strategic_note else 'Normal flow - handle objection and continue presentation.'}

═══════════════════════════════════════════════════════════════
                   GLOBE LIFE FRAMEWORK
═══════════════════════════════════════════════════════════════

PRESENTATION PHASES (follow this sequence):
1. Rapport (40%!) - F.O.R.M: Family, Occupation, Recreation, Me
2. Introduction - "Here today to do 3 simple things..."
3. Company Credibility - Since 1900, 125 years, sports partnerships
4. Accidental Death - No-cost $3k policy + collect referrals
5. Referrals - "10 slots, who would you like to sponsor?"
6. Insurance Types - Whole life vs term explanation
7. Needs Analysis - Health questions, budget, existing coverage
8. Final Expense - $30k recommendation, funeral costs
9. Income Protection - 2 years income replacement
10. Mortgage - Pay off the home benefit
11. College - Education protection
12. Recap - Summary of all benefits
13. Close - "Which option works best for you?"
14. Application - Collect payment, verify health info

DOWN-CLOSE STRATEGY (for price objections):
First: Reduce COVERAGE AMOUNTS, not products
1. $30k → $15k Final Expense
2. $15k → $10k Final Expense
3. $10k → $7.5k Final Expense
4. 2 years → 1 year Income Protection
5. "Which benefit could you go without?"

Then: Switch to supplemental products (Cancer, Accident, Critical Illness)

CORE PHILOSOPHY: "Every presentation has a sale"
- We'd rather sell a $15/mo accident policy than nothing
- Only exit when ALL options exhausted or hard no received

YOUR GUIDANCE SHOULD:
1. Be IMMEDIATELY ACTIONABLE (agent can say it right now)
2. Be SPECIFIC to the client's situation
3. Address the CURRENT objection with Globe Life's proven rebuttals
4. Include EXACT WORDS the agent can use
5. Suggest down-close when price objection occurs
6. Be CONCISE (2-3 sentences max)

RESPONSE FORMAT:
"[Exact words to say in quotes]"

Why: [1 sentence]
Next: [If relevant - down-close or phase transition]

DO NOT:
- Give long explanations
- List multiple options  
- Ignore the call state above
- Repeat guidance already given
- Forget to suggest down-close on price objections"""

    def generate_guidance_stream_v2(
        self,
        call_state: Dict[str, Any],
        trigger_text: str,
        agency: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream guidance tokens using full call state.
        
        This is the V2 method that provides intelligent, context-aware guidance.
        """
        
        # Get relevant context from agency's knowledge base
        relevant_context = self.get_relevant_context(
            trigger_text,
            category=call_state.get("current_product_key"),
            agency=agency
        )
        
        # Build comprehensive system prompt
        system_prompt = self.build_system_prompt_v2(call_state)
        
        if relevant_context:
            system_prompt += f"\n\n═══════════════════════════════════════════════════════════════\n                    TRAINING MATERIALS\n═══════════════════════════════════════════════════════════════\n{relevant_context}"
        
        # Get recent transcript for context
        recent_transcript = call_state.get("recent_transcript", trigger_text)
        active_objection = call_state.get("active_objection", "")
        
        # Build user message
        if active_objection:
            user_message = f"""OBJECTION DETECTED: {active_objection.upper()}

Recent conversation:
"{recent_transcript[-500:]}"

Latest trigger: "{trigger_text}"

Provide immediate guidance to handle this {active_objection} objection."""
        else:
            user_message = f"""Recent conversation:
"{recent_transcript[-500:]}"

Latest: "{trigger_text}"

Provide guidance for the agent."""
        
        messages = [{"role": "user", "content": user_message}]
        
        # Stream the response
        with self.client.messages.stream(
            model=settings.claude_model,
            max_tokens=400,  # Keep responses concise
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
            
            # Get final message to access usage stats
            final_message = stream.get_final_message()
            
            # Log Claude usage for billing
            log_claude_usage(
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
                agency_code=agency,
                session_id=session_id,
                model=settings.claude_model,
                operation='guidance_v2'
            )
    
    # ============ LEGACY METHODS (kept for backward compatibility) ============
    
    def build_system_prompt(self, call_context: CallContext) -> str:
        """Build the system prompt for Claude based on call context (legacy)"""
        
        # PHONE CALL - appointment setting (2-3 min)
        if call_context.call_type == "phone":
            return """You are a real-time coach for Globe Life Liberty National agents making PHONE CALLS to set appointments.

GOAL: Set an appointment in 2-3 minutes. Don't sell - just get them to agree to meet.

PHONE SCRIPT FLOW:
1. "Hey, this is [Name] with Liberty National, your insurance company."
2. "Home Office asked that we touch base with past, current, and prospective clients..."
3. "What time are you home from work? What time is your spouse home?"
4. "I'll have your agent call between ___ and ___."
5. "Go ahead and grab a pen - your confirmation number is ___."

COMMON PHONE OBJECTIONS:

"How long will this take?" → "It doesn't take long - for me the quicker the better."

"Why does my spouse need to be there?" → "They ask us to speak with both of you to answer any questions."

"Just mail it to me" → "I wish we could, but we need to designate your beneficiaries for the no-cost coverage."

"Not interested" → "That's OK, but you're still entitled to the no-cost coverage. It doesn't take long."

"Is someone going to sell me something?" → "It's our job to review your policy and renew your no-cost coverage."

RESPONSE FORMAT: Give EXACTLY what to say. 1-2 sentences max. No headers or emojis."""

        # PRESENTATION - full sales call (30-45 min)
        current_product = PRODUCT_NAMES.get(call_context.current_product, call_context.current_product)
        products_remaining = [
            PRODUCT_NAMES[p] for p in PRODUCT_HIERARCHY 
            if p not in call_context.products_tried and p != call_context.current_product
        ]
        
        client_profile = []
        if call_context.client_age:
            client_profile.append(f"Age: ~{call_context.client_age}")
        if call_context.client_occupation:
            client_profile.append(f"Occupation: {call_context.client_occupation}")
        if call_context.client_family:
            client_profile.append(f"Family: {call_context.client_family}")
        
        client_info = ", ".join(client_profile) if client_profile else "Unknown"
        
        return f"""You are a real-time coach for Globe Life Liberty National agents during IN-HOME PRESENTATIONS.

CURRENT STATE:
- Product: {current_product}
- Client: {client_info}
- Sentiment: {call_context.client_sentiment}
- Objections: {', '.join(call_context.objections_faced) or 'None yet'}
- Remaining: {', '.join(products_remaining) or 'None'}

COMMON OBJECTIONS:

"I need to think about it" → "I understand. What specifically concerns you - the coverage or the investment?"

"I need to talk to my spouse" → "Of course. What do you think they'd be concerned about?"

"I can't afford it" → Reduce coverage. "Let me adjust this to fit your budget."

"I already have insurance" → "Great. When did you last review it? Most people are underinsured."

DOWN-CLOSE LEVELS (when they can't afford):
1. Reduce Final Expense to $15,000
2. Reduce to $10,000
3. Reduce to $7,500
4. Reduce income protection to 1 year
5. Ask which coverage they could go without

RESPONSE FORMAT: Give EXACTLY what to say. 1-3 sentences max. No headers or emojis."""

    def generate_guidance(
        self, 
        transcript_chunk: str, 
        call_context: CallContext,
        agency: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Generate guidance based on a transcript chunk and call context (legacy)"""
        
        # Get relevant context from agency's knowledge base
        relevant_context = self.get_relevant_context(
            transcript_chunk,
            category=call_context.current_product,
            agency=agency
        )
        
        system_prompt = self.build_system_prompt(call_context)
        
        if relevant_context:
            system_prompt += f"\n\nRELEVANT TRAINING MATERIALS:\n{relevant_context}"
        
        messages = [
            {
                "role": "user",
                "content": f"Conversation snippet:\n\n\"{transcript_chunk}\"\n\nProvide guidance for the agent."
            }
        ]
        
        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=500,
            system=system_prompt,
            messages=messages
        )
        
        # Log Claude usage for billing
        log_claude_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            agency_code=agency,
            session_id=session_id,
            model=settings.claude_model,
            operation='guidance'
        )
        
        return response.content[0].text

    def generate_guidance_stream(
        self, 
        transcript_chunk: str, 
        call_context: CallContext,
        agency: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Stream guidance tokens as they're generated (legacy).
        """
        
        # Get relevant context from agency's knowledge base
        relevant_context = self.get_relevant_context(
            transcript_chunk,
            category=call_context.current_product,
            agency=agency
        )
        
        system_prompt = self.build_system_prompt(call_context)
        
        if relevant_context:
            system_prompt += f"\n\nRELEVANT TRAINING MATERIALS:\n{relevant_context}"
        
        messages = [
            {
                "role": "user",
                "content": f"Conversation snippet:\n\n\"{transcript_chunk}\"\n\nProvide guidance for the agent."
            }
        ]
        
        # Stream the response and track usage
        with self.client.messages.stream(
            model=settings.claude_model,
            max_tokens=500,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
            
            # Get final message to access usage stats
            final_message = stream.get_final_message()
            
            # Log Claude usage for billing
            log_claude_usage(
                input_tokens=final_message.usage.input_tokens,
                output_tokens=final_message.usage.output_tokens,
                agency_code=agency,
                session_id=session_id,
                model=settings.claude_model,
                operation='guidance_stream'
            )
    
    def analyze_transcript(
        self, 
        transcript: str,
        agency: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze a transcript chunk for objections and buying signals"""
        
        messages = [{
            "role": "user",
            "content": f"""Analyze this sales conversation snippet and identify:
1. Any objections from the client
2. Any buying signals
3. The client's current sentiment
4. Key phrases to note

Transcript:
\"{transcript}\"

Respond in JSON format:
{{
    "objection_detected": true/false,
    "objection_type": "price|timing|spouse|already_covered|not_interested|other|null",
    "buying_signal": true/false,
    "signal_type": "interest|agreement|question_about_coverage|null",
    "client_sentiment": "positive|neutral|negative|hostile",
    "key_phrases": ["phrase1", "phrase2"]
}}"""
        }]
        
        response = self.client.messages.create(
            model=settings.claude_model,
            max_tokens=200,
            messages=messages
        )
        
        # Log Claude usage for billing
        log_claude_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            agency_code=agency,
            session_id=session_id,
            model=settings.claude_model,
            operation='analyze_transcript'
        )
        
        try:
            import json
            return json.loads(response.content[0].text)
        except:
            return {
                "objection_detected": False,
                "objection_type": None,
                "buying_signal": False,
                "signal_type": None,
                "client_sentiment": "neutral",
                "key_phrases": []
            }


# Singleton instance
_rag_instance: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get the singleton RAG engine instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGEngine()
    return _rag_instance