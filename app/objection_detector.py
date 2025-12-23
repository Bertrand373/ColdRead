"""
Coachd Objection Detection System v2.0
=====================================
Bulletproof detection with flexible pattern matching

Architecture:
1. Check for buying signals first (BLOCKS objection detection)
2. Tier 1: Exact phrase matching (fastest, highest confidence)
3. Tier 2: Flexible keyword patterns (catches filler words like "really", "just")
4. Return objection type + routing info (phone rebuttal vs AI)

Key improvements:
- Catches "can't really afford" (filler words don't break detection)
- Catches "don't really see the point" (the original bug!)
- 200+ phrase variations from Globe Life training materials
"""

import re
from typing import Optional, List, Set
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
    
    # Detection method (for debugging)
    match_type: Optional[str] = None  # "exact" or "pattern"


# =============================================================================
# BUYING SIGNALS - If ANY match, DO NOT fire objection handling
# =============================================================================

BUYING_SIGNALS = [
    # Affordability (POSITIVE - they're stating what they CAN do)
    "i can afford", "i could afford", "what i can afford",
    "can afford about", "afford around", "can do about",
    "could do about", "can probably do", "i can swing",
    "my budget is", "budget of about", "i could probably",
    "that's doable", "i can do that", "that works for me",
    
    # Ready to buy
    "sounds good", "that sounds good", "sounds great",
    "sounds fair", "sounds reasonable", "let's do it",
    "let's do this", "let's get started", "i'm ready",
    "sign me up", "where do i sign", "what do i need to do",
    "what's next", "i'll take it", "we'll take it",
    "let's move forward",
    "let's go ahead", "go ahead and", "let's proceed",
    
    # Agreement
    "that makes sense", "makes sense to me", "i understand now",
    "okay let's", "alright let's", "yes let's",
    "i like that", "that works", "that'll work", "perfect",
    "absolutely", "definitely", "for sure",
    "i'm in,", "we're in,",  # More specific - require comma/end
    "i'm in!", "we're in!",
    "count me in", "sign me in",
    
    # Asking about next steps (positive intent)
    "when can we start", "how soon can", "what's the first payment",
    "how do we get started", "what happens next",
    "when does coverage start", "when am i covered",
]


# =============================================================================
# FLEXIBLE PATTERN DEFINITIONS
# =============================================================================
# Each pattern defines keyword combinations that indicate an objection
# This catches variations like "can't really afford" where filler words are inserted

FLEXIBLE_PATTERNS = {
    "price": {
        # Must have ONE of these core words
        "core_words": ["afford", "expensive", "pricey", "costly", "budget", "money", "price", "cost"],
        # AND must have context suggesting negative/can't
        "context_words": ["can't", "cannot", "couldn't", "too", "don't have", "tight", "stretch", "much"],
        # Block if these appear (buying signals)
        "block_phrases": ["i can afford", "what i can afford", "can afford about", "that's doable"],
    },
    
    "stall": {
        "core_words": ["think about", "think on", "think it over", "sleep on", "pray", "decide", "decision"],
        "context_words": ["need to", "have to", "let me", "want to", "going to", "gonna", "big", "time"],
        "block_phrases": ["don't think", "i think so", "think that's", "i think i"],
    },
    
    "need": {
        "core_words": ["need", "necessary", "point", "purpose", "why"],
        "context_words": ["don't", "do i", "is it", "what's the", "see the", "really"],
        "block_phrases": ["i need to talk", "need to think", "i need this", "we need"],
    },
    
    "spouse": {
        "core_words": ["wife", "husband", "spouse", "partner", "other half"],
        "context_words": ["talk to", "ask", "check with", "run by", "discuss", "without"],
        "block_phrases": [],
    },
    
    "timing": {
        "core_words": ["time", "long", "busy", "hurry", "minutes", "work"],
        "context_words": ["how", "much", "don't have", "at", "only have", "in a"],
        "block_phrases": ["what time", "any time", "good time"],
    },
}


# =============================================================================
# PHONE CALL OBJECTIONS - Globe Life's exact rebuttals (NO AI NEEDED)
# =============================================================================

PHONE_REBUTTALS = {
    "timing": {
        "phrases": [
            # Original
            "how long is this going to take",
            "how long will this take",
            "how much time",
            "don't have much time",
            "only have a few minutes",
            "in a hurry",
            "kind of busy",
            "at work right now",
            "i'm at work",
            # NEW: Expanded variations
            "how long does this take",
            "how much time is this",
            "how much time does this take",
            "only have like five minutes",
            "only have a couple minutes",
            "running out of time",
            "about to leave",
            "about to head out",
            "gotta go soon",
            "got somewhere to be",
            "on my lunch break",
            "my break is almost over",
            "i'm working right now",
            "in the middle of something",
            "kind of in the middle of",
            "can we speed this up",
            "can you make it quick",
            "give me the short version",
            "bottom line it for me",
            "cut to the chase",
            "kinda busy",
            "really busy",
            "super busy",
        ],
        "rebuttal": "\"I have a lot of people on my schedule to get this out today, so it doesn't take long - for me the quicker, the better.\" Then continue with the script.",
    },
    
    "spouse_presence": {
        "phrases": [
            # Original
            "why does my spouse have to be there",
            "why does my wife have to be there",
            "why does my husband have to be there",
            "does my spouse need to be",
            "does my wife need to be",
            "does my husband need to be",
            "can't we do this without",
            "do we both need to be",
            # NEW: Expanded variations
            "why do they need to be there",
            "why does she have to be there",
            "why does he have to be there",
            "do they really need to be there",
            "can i just do it myself",
            "can't i just handle this",
            "she's not here right now",
            "he's at work",
            "when my spouse gets home",
            "my wife isn't available",
            "my husband isn't available",
            "can we do it without my",
        ],
        "rebuttal": "\"They ask us to make sure we speak with both of you to answer any questions and to make sure you know how this works in case something happens to either of you.\" Then continue with the script.",
    },
    
    "send_info": {
        "phrases": [
            # Original
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
            # NEW: Expanded variations
            "mail me the information",
            "send it in the mail",
            "put it in the mail",
            "drop it in the mail",
            "send me an email",
            "shoot me an email",
            "email the details",
            "send it to my email",
            "just text me",
            "text me the info",
            "send me a text",
            "let me look it over first",
            "let me read through it",
            "i need to see it in writing",
            "don't do anything over the phone",
            "can you just mail",
            "why don't you mail",
            "why don't you just send",
        ],
        "rebuttal": "\"I sure wish we could. That would make my job a lot easier. However, we need to speak with you to be able to designate your beneficiaries for the no-cost coverage and review everything that you're entitled to through the program.\" Then continue with the script.",
    },
    
    "not_interested": {
        "phrases": [
            # Original
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
            # NEW: Expanded variations
            "i'll pass",
            "i think i'm good",
            "nah i'm good",
            "thanks but no thanks",
            "appreciate it but",
            "thanks for calling but",
            "don't want it",
            "don't need it",
            "not for me",
            "not right now",
            "maybe some other time",
            "i'm all set",
            "pass on that",
            "gonna pass",
            "going to pass",
            "not today",
            "not at this time",
        ],
        "rebuttal": "\"Well that's OK, but you're still entitled to the no-cost coverage. I can still get that out to you and explain everything that you're entitled to through the program. Most people appreciate the short time it takes to explain everything to them. Like I said, it doesn't take long. I'll tell you what I'm going to do...\" Then continue with the script.",
    },
    
    "sales_skepticism": {
        "phrases": [
            # Original
            "going to try and sell me",
            "going to sell me something",
            "are you selling something",
            "is this a sales call",
            "is this a sales pitch",
            "are you trying to sell",
            "what's the catch",
            "sounds too good",
            # NEW: Expanded variations
            "you trying to sell me something",
            "are you gonna try to sell",
            "what are you selling",
            "i don't want to be sold",
            "i hate sales calls",
            "another sales pitch",
            "i know what this is",
            "this sounds fishy",
            "sounds too good to be true",
            "where's the catch",
            "there's always a catch",
            "nothing's free",
            "what do you get out of it",
            "how do you make money",
            "how do i know this is real",
            "how do i know you're legit",
            "prove you're from globe life",
            "i need to verify this",
            "let me call the company",
            "i'll call globe life directly",
            "is this legitimate",
            "is this for real",
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
            # Original core phrases
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
            # NEW: Filler word variations (THE BUG FIX!)
            "can't really afford",
            "can't quite afford",
            "just can't afford",
            "honestly can't afford",
            "really can't afford",
            # NEW: Slang/casual
            "that's too rich for my blood",
            "way out of my price range",
            "kinda expensive",
            "kind of expensive",
            "bit much",
            "little too much",
            "wasn't expecting that much",
            "that's pretty pricey",
            "little pricey",
            "kinda steep",
            "that's kinda steep",
            "more than i was expecting",
            "that's more than i was expecting",
            # NEW: Question forms
            "is there anything cheaper",
            "got anything lower",
            "what's the cheapest option",
            "do you have a cheaper plan",
            "what if i can't afford that",
            "what's the minimum",
            "anything more affordable",
            "lower option",
            "something cheaper",
            "is there something cheaper",
            # NEW: Indirect/soft
            "i'm on a fixed income",
            "social security only",
            "living paycheck to paycheck",
            "barely making ends meet",
            "fixed income",
            "on disability",
            "on a budget",
            "watching my spending",
        ],
        "bridge": "I totally understand...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "spouse": {
        "phrases": [
            # Original
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
            # NEW: Partner terminology
            "talk to my partner",
            "ask my partner",
            "check with my partner",
            "my other half",
            "my significant other",
            "my better half",
            # NEW: Generic family
            "talk to the family",
            "family decision",
            "discuss it with the family",
            # NEW: Relationship dynamics
            "my wife handles the finances",
            "my husband handles the money",
            "she makes these decisions",
            "he makes these decisions",
            "need her approval",
            "need his approval",
            "she's not here right now",
            "he's at work",
            "when my spouse gets home",
            "without my wife",
            "without my husband",
            "handles all the finances",
            "my husband handles all the finances",
            "run this by my",
            "let me run this by",
        ],
        "bridge": "Of course, that makes sense...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "stall": {
        "phrases": [
            # Original
            "need to think about it",
            "have to think about",
            "let me think about",
            "sleep on it",
            "pray about it",
            "pray on it",
            "big decision",
            "need some time",
            "not ready to decide",
            # NEW: Filler variations
            "i really need to think",
            "i just need to think",
            "gotta think about it",
            "gonna have to think",
            "really need to think",
            # NEW: Alternative phrasing
            "not ready to decide today",
            "not making a decision today",
            "don't want to rush",
            "don't want to be hasty",
            "let me sit with it",
            "let me mull it over",
            "need to process this",
            "need to digest this",
            "it's a lot to take in",
            "lot to think about",
            # NEW: Time-based stalls
            "maybe next month",
            "call me back later",
            "try me again in a few weeks",
            "after the holidays",
            "when things settle down",
            "when i get my tax return",
            "first of the month",
            "end of the month",
            "next week",
            "next pay period",
            # NEW: Religious variations
            "need to pray on it",
            "want to pray about it",
            "let me talk to god about it",
            "need to seek guidance",
        ],
        "bridge": "I hear you...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "covered": {
        "phrases": [
            # Original
            "already have insurance",
            "already have coverage",
            "covered through work",
            "have it through my job",
            "employer provides",
            "work insurance",
            "got insurance through",
            # NEW: Work coverage variations
            "my job covers me",
            "got it through my employer",
            "company provides it",
            "work takes care of that",
            "benefits at work",
            "group policy at work",
            "group plan",
            "through my union",
            "employer covers all that",
            "my employer covers all",
            "employer covers",
            # NEW: Existing policies
            "i have a policy already",
            "i'm already covered",
            "we're already insured",
            "got a plan already",
            "been with another company",
            "have another policy",
            "with state farm",
            "with allstate",
            "with geico",
            # NEW: Adequacy claims
            "i have enough coverage",
            "i'm good on insurance",
            "already fully covered",
            "maxed out my coverage",
            "plenty of coverage",
            # NEW: Spouse coverage
            "my spouse has us covered",
            "covered under my wife's plan",
            "covered under my husband's plan",
            "on my spouse's policy",
            "wife's job covers us",
            "husband's job covers us",
        ],
        "bridge": "That's great that you're thinking about protection...",
        "use_fast_model": True,
        "skip_rag": True,
    },
    
    "health": {
        "phrases": [
            # Original
            "health problems",
            "pre-existing condition",
            "won't qualify",
            "can't get approved",
            "too old for",
            "health issues",
            "medical condition",
            # NEW: Specific conditions
            "i have diabetes",
            "i have heart problems",
            "i have cancer",
            "i've had a stroke",
            "i've had a heart attack",
            "on medication",
            "take pills for",
            "health isn't great",
            "not in the best health",
            "bad health",
            "poor health",
            # NEW: Qualification concerns
            "will i even qualify",
            "can i even get approved",
            "probably won't approve me",
            "doubt i'll qualify",
            "my health is too bad",
            "too many health issues",
            "pre-existing",
            # NEW: Previous rejections
            "been denied before",
            "turned down before",
            "couldn't get coverage",
            "no one will cover me",
            "rejected before",
            "declined before",
        ],
        "bridge": "Good question...",
        "use_fast_model": False,  # Sonnet for complex health discussions
        "skip_rag": False,
    },
    
    "trust": {
        "phrases": [
            # Original
            "sounds like a scam",
            "is this legit",
            "too good to be true",
            "what's the catch",
            "fine print",
            "hidden fees",
            # NEW: Sales skepticism (presentation context)
            "is this legitimate",
            "are you for real",
            "sounds fishy",
            "something's off",
            "i don't trust",
            "how do i know",
            "prove it",
            "show me proof",
            "where's the catch",
            "there's always a catch",
            "nothing is free",
            "what am i missing",
            "what aren't you telling me",
            "read the fine print",
        ],
        "bridge": "I appreciate you being direct...",
        "use_fast_model": False,  # Sonnet for trust issues
        "skip_rag": False,
    },
    
    "need": {
        "phrases": [
            # Original
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
            # NEW: THE ORIGINAL BUG FIXES!
            "don't really see the point",
            "not sure i really need",
            "do i honestly need",
            "don't think i really need",
            "why would i even need",
            "not convinced i need",
            "really don't need",
            "just don't need",
            # NEW: Youth/health invincibility
            "i'm too young for this",
            "i'm in great shape",
            "i never get sick",
            "healthy as a horse",
            "nothing wrong with me",
            "i take care of myself",
            "take care of myself",
            "great shape",
            # NEW: Invincibility mindset
            "i'll be fine",
            "nothing's gonna happen to me",
            "i've got time",
            "i'm not going anywhere",
            "i'll deal with it later",
            "that's way down the road",
            "not worried about it",
            "cross that bridge when i come to it",
            "way too young",
            "plenty of time",
            "deal with it when i'm older",
            "i'll deal with it when i'm older",
            "i take care of myself",
            "in great shape",
            "i'm in great shape",
            # NEW: Questioning necessity
            "is this even necessary",
            "seems unnecessary",
            "do people actually need this",
            "isn't that what savings are for",
            "what do i need this for",
        ],
        "bridge": "That's a fair question...",
        "use_fast_model": True,
        "skip_rag": True,
    },
}


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def _check_flexible_pattern(text: str, pattern_name: str, pattern: dict) -> bool:
    """
    Check if text matches a flexible keyword pattern.
    
    Returns True if:
    - Text contains at least one core word
    - AND text contains at least one context word
    - AND text doesn't contain any blocking phrases
    """
    text_lower = text.lower()
    
    # Check for blocking phrases first
    for block in pattern.get("block_phrases", []):
        if block in text_lower:
            return False
    
    # Check for core word
    has_core = any(core in text_lower for core in pattern["core_words"])
    if not has_core:
        return False
    
    # Check for context word
    has_context = any(ctx in text_lower for ctx in pattern["context_words"])
    
    return has_context


def detect_objection(text: str, call_type: str = "presentation") -> DetectionResult:
    """
    Detect objections using tiered matching:
    1. Buying signals (blocks detection)
    2. Exact phrase matching (fastest, highest confidence)
    3. Flexible keyword patterns (catches variations)
    
    Args:
        text: The transcript text to analyze
        call_type: "phone" or "presentation"
    
    Returns:
        DetectionResult with routing info
    """
    text_lower = text.lower().strip()
    
    # =================================================================
    # STEP 1: Check for buying signals FIRST (blocks objection detection)
    # =================================================================
    # BUT: Don't match "i can afford" inside "i can't afford"
    negated_signals = [
        "can't afford", "cannot afford", "couldn't afford",
        "can't do", "cannot do", "couldn't do",
        "can't swing", "cannot swing",
        "not afford", "never afford",
    ]
    
    has_negated_signal = any(neg in text_lower for neg in negated_signals)
    
    if not has_negated_signal:
        for signal in BUYING_SIGNALS:
            if signal in text_lower:
                return DetectionResult(
                    detected=False,
                    is_buying_signal=True
                )
    
    # =================================================================
    # STEP 2: For phone calls, check Globe Life rebuttals (EXACT MATCH)
    # =================================================================
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
                        match_type="exact",
                    )
    
    # =================================================================
    # STEP 3: Check presentation objections (EXACT MATCH - Tier 1)
    # =================================================================
    for obj_type, data in PRESENTATION_OBJECTIONS.items():
        for phrase in data["phrases"]:
            if phrase in text_lower:
                return DetectionResult(
                    detected=True,
                    objection_type=obj_type,
                    bridge_phrase=data.get("bridge"),
                    use_fast_model=data.get("use_fast_model", True),
                    skip_rag=data.get("skip_rag", False),
                    match_type="exact",
                )
    
    # =================================================================
    # STEP 4: Flexible pattern matching (Tier 2 - catches variations)
    # =================================================================
    # This catches things like "can't really afford" where filler words
    # break exact phrase matching
    
    for pattern_name, pattern in FLEXIBLE_PATTERNS.items():
        if _check_flexible_pattern(text_lower, pattern_name, pattern):
            # Map pattern name to objection data
            obj_data = PRESENTATION_OBJECTIONS.get(pattern_name, {})
            return DetectionResult(
                detected=True,
                objection_type=pattern_name,
                bridge_phrase=obj_data.get("bridge", "I understand..."),
                use_fast_model=obj_data.get("use_fast_model", True),
                skip_rag=obj_data.get("skip_rag", False),
                match_type="pattern",
            )
    
    # =================================================================
    # STEP 5: No objection detected
    # =================================================================
    return DetectionResult(detected=False)


def has_any_trigger(text: str) -> bool:
    """
    Quick check if text contains ANY potential trigger.
    Used to decide if we should run full detection.
    """
    text_lower = text.lower()
    
    # Quick keyword check (faster than phrase matching)
    quick_triggers = [
        # Price
        "afford", "expensive", "cost", "price", "budget", "tight", "pricey", "steep",
        # Stall
        "think about", "think on", "sleep on", "pray", "decide", "decision",
        # Spouse
        "talk to", "spouse", "wife", "husband", "partner",
        # Need/Invincible
        "need", "point", "necessary", "why do i", "why would i", "young", "healthy",
        # Covered
        "already have", "through work", "employer", "covered",
        # Not interested
        "not interested", "no thanks", "i'm good", "all set", "pass",
        # Trust
        "scam", "catch", "legit", "trust", "fishy",
        # Timing
        "how long", "busy", "at work", "hurry",
        # Health
        "health", "pre-existing", "qualify", "condition", "diabetes",
        # Send info
        "mail", "send", "email", "text me",
    ]
    
    return any(kw in text_lower for kw in quick_triggers)


# =============================================================================
# TESTING / DEBUGGING
# =============================================================================

def test_detection():
    """Test cases to verify detection works correctly"""
    test_cases = [
        # Should detect - exact match
        ("I can't afford that", "price", True),
        ("Let me think about it", "stall", True),
        ("I need to talk to my wife", "spouse", True),
        
        # Should detect - filler words (THE BUG FIX!)
        ("I can't really afford that right now", "price", True),
        ("I don't really see the point", "need", True),
        ("I just can't afford it", "price", True),
        ("I'm gonna have to think about it", "stall", True),
        
        # Should detect - slang
        ("That's kinda expensive", "price", True),
        ("That's steep", "price", True),
        
        # Should NOT detect - buying signals
        ("What can I afford?", None, False),
        ("That sounds good, let's do it", None, False),
        ("I can afford about $50 a month", None, False),
        
        # Edge cases
        ("I'm good", "not_interested", True),  # Phone objection
        ("Thanks but no thanks", "not_interested", True),
    ]
    
    print("Running detection tests...\n")
    passed = 0
    failed = 0
    
    for text, expected_type, should_detect in test_cases:
        # Test both phone and presentation contexts
        result = detect_objection(text, "presentation")
        
        if should_detect:
            if result.detected and result.objection_type == expected_type:
                print(f"✓ PASS: '{text}' -> {expected_type} ({result.match_type})")
                passed += 1
            elif result.detected:
                print(f"✗ FAIL: '{text}' -> Expected {expected_type}, got {result.objection_type}")
                failed += 1
            else:
                print(f"✗ FAIL: '{text}' -> Expected detection, got none")
                failed += 1
        else:
            if not result.detected or result.is_buying_signal:
                print(f"✓ PASS: '{text}' -> No detection (correct)")
                passed += 1
            else:
                print(f"✗ FAIL: '{text}' -> Should NOT detect, got {result.objection_type}")
                failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    test_detection()