"""
Coachd Objection Detection System
Phrase-based detection with buying signal blocking

Architecture:
1. Check for buying signals first (BLOCKS objection detection)
2. Use phrase matching (not keywords) for accurate detection
3. Return objection type + routing info (phone rebuttal vs AI)
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of objection detection"""
    detected: bool = False
    objection_type: Optional[str] = None
    is_buying_signal: bool = False
    
    # For phone calls - Globe Life's exact rebuttal (no AI needed)
    phone_rebuttal: Optional[str] = None
    
    # For presentations - bridge phrase while AI generates
    bridge_phrase: Optional[str] = None
    
    # Routing: True = use Haiku (fast), False = use Sonnet (thorough)
    use_fast_model: bool = True
    
    # Skip RAG search for known objections
    skip_rag: bool = False


# =============================================================================
# BUYING SIGNALS - If ANY match, DO NOT fire objection handling
# =============================================================================

BUYING_SIGNALS = [
    # Affordability (POSITIVE - they're stating what they CAN do)
    "i can afford", "i could afford", "what i can afford",
    "can afford about", "afford around", "can do about",
    "could do about", "can probably do", "i can swing",
    "my budget is", "budget of about", "i could probably",
    
    # Ready to buy
    "sounds good", "that sounds good", "sounds great",
    "sounds fair", "sounds reasonable", "let's do it",
    "let's do this", "let's get started", "i'm ready",
    "sign me up", "where do i sign", "what do i need to do",
    "what's next", "i'll take it", "we'll take it",
    "let's move forward", "i'm in", "we're in",
    
    # Agreement
    "that makes sense", "makes sense to me", "i understand now",
    "okay let's", "alright let's", "yes let's",
    "i like that", "that works", "that'll work", "perfect",
    
    # Asking about next steps (positive intent)
    "when can we start", "how soon can", "what's the first payment",
    "how do we get started",
]


# =============================================================================
# PHONE CALL OBJECTIONS - Globe Life's exact rebuttals (NO AI NEEDED)
# =============================================================================

PHONE_REBUTTALS = {
    "timing": {
        "phrases": [
            "how long is this going to take",
            "how long will this take",
            "how much time",
            "don't have much time",
            "only have a few minutes",
            "in a hurry",
            "kind of busy",
            "at work right now",
            "i'm at work",
        ],
        "rebuttal": "\"I have a lot of people on my schedule to get this out today, so it doesn't take long - for me the quicker, the better.\" Then continue with the script.",
    },
    
    "spouse_presence": {
        "phrases": [
            "why does my spouse have to be there",
            "why does my wife have to be there",
            "why does my husband have to be there",
            "does my spouse need to be",
            "does my wife need to be",
            "does my husband need to be",
            "can't we do this without",
            "do we both need to be",
        ],
        "rebuttal": "\"They ask us to make sure we speak with both of you to answer any questions and to make sure you know how this works in case something happens to either of you.\" Then continue with the script.",
    },
    
    "send_info": {
        "phrases": [
            "just mail it",
            "just send it",
            "mail it to me",
            "can you mail me",
            "can you send me",
            "can you email me",
            "email me something",
            "send me information",
            "send me some info",
            "send me something",
            "send me the details",
        ],
        "rebuttal": "\"I sure wish we could. That would make my job a lot easier. However, we need to speak with you to be able to designate your beneficiaries for the no-cost coverage and review everything that you're entitled to through the program.\" Then continue with the script.",
    },
    
    "not_interested": {
        "phrases": [
            "not interested",
            "i'm not interested",
            "we're not interested",
            "don't need any",
            "don't want any",
            "no thank you",
            "no thanks",
            "we're good",
            "i'm good",
            "all set",
            "we're all set",
        ],
        "rebuttal": "\"Well that's OK, but you're still entitled to the no-cost coverage. I can still get that out to you and explain everything that you're entitled to through the program. Most people appreciate the short time it takes to explain everything to them. Like I said, it doesn't take long. I'll tell you what I'm going to do...\" Then continue with the script.",
    },
    
    "sales_skepticism": {
        "phrases": [
            "going to try and sell me",
            "going to sell me something",
            "are you selling something",
            "is this a sales call",
            "is this a sales pitch",
            "are you trying to sell",
            "what's the catch",
            "sounds too good",
        ],
        "rebuttal": "\"It's our job to review your policy with you and renew your no-cost coverage. We have an appointment every half hour so it doesn't take very much of your time. I'll tell you what I'm going to do...\" Then continue with the script.",
    },
}


# =============================================================================
# PRESENTATION OBJECTIONS - Need AI guidance with context
# =============================================================================

PRESENTATION_OBJECTIONS = {
    "price": {
        "phrases": [
            "can't afford", "cannot afford", "couldn't afford",
            "too expensive", "too much money", "costs too much",
            "that's a lot", "that's too much", "way too much",
            "more than i expected", "more than i thought",
            "more than i was thinking", "wasn't expecting that",
            "out of my budget", "over my budget",
            "don't have the money", "don't have that kind",
            "tight right now", "money is tight", "things are tight",
            "can't do that much", "can't swing that",
            "that's steep", "pretty steep",
            "stretching it", "higher than i thought",
        ],
        "bridge": "I totally understand...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "spouse": {
        "phrases": [
            "need to talk to my wife",
            "need to talk to my husband",
            "have to talk to my wife",
            "have to talk to my husband",
            "talk to my wife about it",
            "talk to my husband about it",
            "talk to my wife about this",
            "talk to my husband about this",
            "discuss it with my wife",
            "discuss it with my husband",
            "discuss with my spouse",
            "have to check with",
            "need to check with",
            "let me ask my",
            "run it by my",
            "we decide together",
            "can't decide without",
            "my spouse needs to",
            "my wife needs to",
            "my husband needs to",
            "see what my wife thinks",
            "see what my husband thinks",
            "wife would kill me",
            "husband would kill me",
            "talk to my wife first",
            "talk to my husband first",
            "ask my wife",
            "ask my husband",
        ],
        "bridge": "Of course, that makes sense...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "stall": {
        "phrases": [
            "need to think about it",
            "have to think about",
            "let me think about",
            "sleep on it",
            "pray about it",
            "pray on it",
            "big decision",
            "need some time",
            "not ready to decide",
        ],
        "bridge": "I hear you...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "covered": {
        "phrases": [
            "already have insurance",
            "already have coverage",
            "covered through work",
            "have it through my job",
            "employer provides",
            "work insurance",
            "got insurance through",
        ],
        "bridge": "That's great that you're thinking about protection...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "health": {
        "phrases": [
            "health problems",
            "pre-existing condition",
            "won't qualify",
            "can't get approved",
            "too old for",
            "health issues",
            "medical condition",
        ],
        "bridge": "Good question...",
        "use_fast_model": False,  # Sonnet for complex health discussions
        "skip_rag": False,
    },
    
    "trust": {
        "phrases": [
            "sounds like a scam",
            "is this legit",
            "too good to be true",
            "what's the catch",
            "fine print",
            "hidden fees",
        ],
        "bridge": "I appreciate you being direct...",
        "use_fast_model": False,  # Sonnet for trust issues
        "skip_rag": False,
    },
    
    "need": {
        "phrases": [
            "don't know if i need",
            "don't think i need",
            "do i really need",
            "do i even need",
            "don't really need",
            "don't need life insurance",
            "don't need insurance",
            "not sure i need",
            "don't see the point",
            "what's the point",
            "why do i need",
            "why would i need",
            "is this necessary",
            "is it necessary",
            "don't see why",
            "i'm young",
            "i'm healthy",
            "nothing's going to happen",
            "won't need it",
        ],
        "bridge": "That's a fair question...",
        "use_fast_model": True,
        "skip_rag": True,
    },
}


def detect_objection(text: str, call_type: str = "presentation") -> DetectionResult:
    """
    Detect objections using phrase matching.
    
    Args:
        text: The transcript text to analyze
        call_type: "phone" or "presentation"
    
    Returns:
        DetectionResult with routing info
    """
    text_lower = text.lower().strip()
    
    # STEP 1: Check for buying signals FIRST (blocks objection detection)
    # BUT: Don't match "i can afford" inside "i can't afford"
    # Check for NEGATED versions of buying signals first
    negated_signals = [
        "can't afford", "cannot afford", "couldn't afford",
        "can't do", "cannot do", "couldn't do",
        "can't swing", "cannot swing",
        "not afford", "never afford",
    ]
    
    # If text contains a negated buying signal, skip buying signal check entirely
    has_negated_signal = any(neg in text_lower for neg in negated_signals)
    
    if not has_negated_signal:
        for signal in BUYING_SIGNALS:
            if signal in text_lower:
                return DetectionResult(
                    detected=False,
                    is_buying_signal=True
                )
    
    # STEP 2: For phone calls, check Globe Life rebuttals
    if call_type == "phone":
        for obj_type, data in PHONE_REBUTTALS.items():
            for phrase in data["phrases"]:
                if phrase in text_lower:
                    return DetectionResult(
                        detected=True,
                        objection_type=obj_type,
                        phone_rebuttal=data["rebuttal"],
                        use_fast_model=True,
                        skip_rag=True,
                    )
    
    # STEP 3: For presentations (or phone edge cases), check presentation objections
    for obj_type, data in PRESENTATION_OBJECTIONS.items():
        for phrase in data["phrases"]:
            if phrase in text_lower:
                return DetectionResult(
                    detected=True,
                    objection_type=obj_type,
                    bridge_phrase=data.get("bridge"),
                    use_fast_model=data.get("use_fast_model", True),
                    skip_rag=data.get("skip_rag", False),
                )
    
    # STEP 4: No objection detected
    return DetectionResult(detected=False)


def has_any_trigger(text: str) -> bool:
    """
    Quick check if text contains ANY potential trigger.
    Used to decide if we should run full detection.
    """
    text_lower = text.lower()
    
    # Quick keyword check (faster than phrase matching)
    quick_triggers = [
        "afford", "expensive", "cost", "price", "budget", "tight",
        "think about", "talk to", "spouse", "wife", "husband",
        "not sure", "maybe", "later", "sleep on", "pray",
        "mail", "send", "email",
        "already have", "through work", "employer",
        "not interested", "no thanks", "i'm good",
        "scam", "catch", "legit",
        "how long", "busy", "at work",
        "health", "pre-existing", "qualify",
        "need", "point", "necessary", "why do i", "why would i",  # Added for invincible objection
    ]
    
    return any(kw in text_lower for kw in quick_triggers)