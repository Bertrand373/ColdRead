"""
Coachd.ai - Call State Machine
===============================
Tracks the complete state of a live sales call:
- Client profile (extracted from conversation)
- Product state (current, pitched, rejected)
- Objection state (active, queue, history)
- Conversation memory (rolling transcript)
- Viability signals (Phase 4)

This is the brain that makes guidance intelligent.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


# ============ CONSTANTS ============

# Call types - phone calls vs full presentations have different objection handling
CALL_TYPES = ["phone", "presentation"]

# Presentation phases - Globe Life follows a strict sequence
PRESENTATION_PHASES = [
    "rapport",           # F.O.R.M method (Family, Occupation, Recreation, Me) - 40% of presentation!
    "introduction",      # Why we're here, what we'll do
    "company_credibility", # Globe Life history, sports partnerships
    "accidental_death",  # No-cost $3k AD policy + collect referrals
    "referrals",         # Sponsor 10 people for no-cost coverage
    "insurance_types",   # Whole life vs term explanation
    "needs_analysis",    # Health questions, budget, existing coverage
    "final_expense",     # $30k recommendation, funeral costs
    "income_protection", # 2 years income replacement
    "mortgage",          # Pay off home benefit
    "college",           # Education protection
    "recap",             # Summary of all benefits
    "close",             # Option 1 vs Option 2
    "application"        # Collecting payment info, health verification
]

# Product hierarchy - only used after exhausting down-close on primary sale
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

# Down-close levels for whole life (Globe Life specific)
# The down-close reduces coverage AMOUNTS first, not products
DOWN_CLOSE_LEVELS = [
    {"final_expense": 30000, "income_years": 2, "label": "Full Coverage - Option 1"},
    {"final_expense": 30000, "income_years": 2, "label": "Full Coverage - Option 2"},  # Different pricing
    {"final_expense": 15000, "income_years": 2, "label": "Reduced Final Expense"},
    {"final_expense": 10000, "income_years": 2, "label": "Basic Final Expense"},
    {"final_expense": 7500, "income_years": 2, "label": "Minimum Final Expense"},
    {"final_expense": 7500, "income_years": 1, "label": "Reduced Income Protection"},
    {"final_expense": 7500, "income_years": 1, "label": "Pick Most Important", "allow_removal": True},
]

# Objection categories for tracking - expanded for Globe Life scripts
OBJECTION_CATEGORIES = {
    # Price objections
    "afford": "price", "expensive": "price", "cost": "price", 
    "price": "price", "money": "price", "budget": "price",
    "too much": "price", "cheaper": "price", "tight": "price",
    "can't do": "price", "out of my": "price",
    
    # Spouse/family objections
    "spouse": "spouse", "wife": "spouse", "husband": "spouse",
    "talk to": "spouse", "check with": "spouse", "ask my": "spouse",
    "run it by": "spouse",
    # Globe Life specific: Why does spouse need to be there?
    "why does my spouse": "spouse_presence", "spouse have to be": "spouse_presence",
    "without my spouse": "spouse_presence", "spouse doesn't need": "spouse_presence",
    
    # Think about it / stalling
    "think about": "stall", "not sure": "stall", "maybe": "stall",
    "later": "stall", "call me back": "stall", "let me think": "stall",
    "need time": "stall", "sleep on it": "stall", "pray about": "stall",
    "pray on": "stall", "decide later": "stall",
    
    # Send info / mail it (common phone objection)
    "send me": "send_info", "mail me": "send_info", "email me": "send_info",
    "just mail": "send_info", "send information": "send_info",
    "send something": "send_info", "brochure": "send_info",
    
    # Already covered
    "already have": "covered", "work insurance": "covered",
    "through my job": "covered", "employer": "covered",
    "through work": "covered", "have coverage": "covered",
    "i'm covered": "covered", "got insurance": "covered",
    
    # Not interested
    "not interested": "not_interested", "no thanks": "not_interested",
    "don't need": "not_interested", "pass": "not_interested",
    "not for me": "not_interested", "i'm good": "not_interested",
    
    # Trust/skepticism issues
    "scam": "trust", "pushy": "trust", "catch": "trust",
    "too good": "trust", "fishy": "trust",
    "sell me": "skepticism", "salesman": "skepticism", "sales pitch": "skepticism",
    "is this a": "skepticism", "what's the catch": "skepticism",
    
    # Timing objections (especially for phone calls)
    "bad time": "timing", "busy": "timing", "not now": "timing",
    "swamped": "timing", "how long": "timing", "take long": "timing",
    "don't have time": "timing", "in a hurry": "timing",
    "at work": "timing", "working": "timing",
    
    # Health/age
    "too young": "health", "too old": "health", "healthy": "health",
    "pre-existing": "health", "health issues": "health",
    "never get sick": "health", "don't need it yet": "health",
    
    # Lapsed policy specific (from Globe Life lapsed script)
    "canceled": "lapsed", "thought i canceled": "lapsed",
    "old job": "lapsed", "don't work there": "lapsed",
    
    # Referral resistance
    "don't know anyone": "referral_resist", "can't think of": "referral_resist",
    "rather not": "referral_resist",
}

# Signals for viability scoring (Phase 4)
POSITIVE_SIGNALS = [
    "sounds good", "that's interesting", "tell me more",
    "what's the coverage", "how much is", "my kids", "my family",
    "I like that", "that makes sense", "okay", "go on",
    "what about", "and if", "so if I"
]

BUYING_SIGNALS = [
    "where do I sign", "how do I pay", "when does it start",
    "what do I need", "let's do it", "I'm ready", "sign me up",
    "let's get started", "I'll take it", "sounds perfect"
]

HARD_NO_SIGNALS = [
    "not interested", "don't call", "stop calling",
    "take me off", "do not contact", "I said no",
    "leave me alone", "harassment", "don't call back"
]


class Sentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    RESISTANT = "resistant"
    HOSTILE = "hostile"


class ObjectionResult(Enum):
    PENDING = "pending"
    OVERCOME = "overcome"
    DOWN_CLOSE = "down_close"  # Led to product transition
    UNRESOLVED = "unresolved"


# ============ DATA CLASSES ============

@dataclass
class ClientProfile:
    """Client information extracted during call"""
    age: Optional[int] = None
    occupation: Optional[str] = None
    family: Optional[str] = None  # "married, 2 kids"
    tobacco: Optional[str] = None  # Y/N
    budget: Optional[int] = None  # Monthly budget mentioned (e.g., "I can only do $50")
    
    def to_string(self) -> str:
        """Format for Claude prompt"""
        parts = []
        if self.age:
            parts.append(f"Age: ~{self.age}")
        if self.occupation:
            parts.append(f"Occupation: {self.occupation}")
        if self.family:
            parts.append(f"Family: {self.family}")
        if self.tobacco:
            parts.append(f"Tobacco: {self.tobacco}")
        if self.budget:
            parts.append(f"Budget: ~${self.budget}/mo")
        return ", ".join(parts) if parts else "Unknown"


@dataclass
class ProductRejection:
    """Record of a rejected product"""
    product: str
    reason: str  # price, spouse, timing, etc.
    timestamp: float
    transcript_snippet: str = ""


@dataclass
class ObjectionRecord:
    """Record of an objection encountered"""
    category: str  # price, spouse, stall, etc.
    raw_text: str  # What they actually said
    timestamp: float
    result: ObjectionResult = ObjectionResult.PENDING
    guidance_given: Optional[str] = None


@dataclass
class GuidanceRecord:
    """Record of guidance provided"""
    text: str
    trigger: str  # What triggered this guidance
    timestamp: float
    objection_type: Optional[str] = None


# ============ MAIN STATE MACHINE ============

class CallStateMachine:
    """
    Tracks complete state of a live sales call.
    
    Supports two call types:
    - "phone": Setting appointments (2-3 min, different objections)
    - "presentation": Full sales presentation (30-45 min, Globe Life script)
    
    Usage:
        state = CallStateMachine(session_id="abc123", agency="ADERHOLT", call_type="presentation")
        state.update_client_profile(age=44, family="married, 2 kids")
        state.add_transcript("Client: I can't afford that")
        state.record_objection("price", "I can't afford that")
        state.record_guidance("Let me show you a more affordable option...")
        
        # Get state for Claude
        claude_context = state.get_state_for_claude()
    """
    
    # Memory limits to manage context window
    MAX_TRANSCRIPT_CHARS = 100000  # ~25k tokens - covers 2-hour presentations
    MAX_RECENT_CHARS = 2000  # Last ~30 seconds (for quick reference)
    MAX_OBJECTION_HISTORY = 10
    MAX_GUIDANCE_HISTORY = 10
    
    def __init__(
        self,
        session_id: str,
        agency: str = "default",
        agent_name: str = "Agent",
        call_type: str = "presentation"  # "phone" or "presentation"
    ):
        self.session_id = session_id
        self.agency = agency
        self.agent_name = agent_name
        self.started_at = time.time()
        
        # Call type determines objection handling strategy
        self.call_type = call_type if call_type in CALL_TYPES else "presentation"
        
        # Presentation phase tracking (only for presentation calls)
        self.presentation_phase = "rapport"
        self.phase_started_at = time.time()
        
        # Client profile
        self.client = ClientProfile()
        self.sentiment = Sentiment.NEUTRAL
        
        # Coverage amounts (Globe Life specific down-close tracking)
        self.coverage = {
            "final_expense": 30000,      # Starts at $30k, can drop to $15k, $10k, $7.5k
            "income_years": 2,           # Years of income protection (can drop to 1)
            "mortgage": True,            # Include mortgage protection
            "college": True,             # Include college education
            "option": 1                  # Option 1 or Option 2 (pricing tiers)
        }
        self.down_close_level = 0  # Index into DOWN_CLOSE_LEVELS
        
        # Product state (for supplemental products after main sale)
        self.current_product = "whole_life"
        self.products_pitched: List[str] = ["whole_life"]  # Start with whole life
        self.products_rejected: Dict[str, ProductRejection] = {}
        
        # Objection state
        self.active_objection: Optional[str] = None
        self.objection_queue: List[ObjectionRecord] = []
        self.objection_history: List[ObjectionRecord] = []
        
        # Conversation memory
        self.full_transcript: str = ""
        self.recent_transcript: str = ""  # Rolling window
        self.key_moments: List[str] = []
        self.guidance_history: List[GuidanceRecord] = []
        
        # Viability tracking (Phase 4)
        self.positive_signals: List[str] = []
        self.negative_signals: List[str] = []
        self.buying_signals: List[str] = []
        self.hard_nos: List[str] = []
        
        # Referrals collected (Globe Life specific)
        self.referrals_collected: int = 0
        
        # Guidance generation state
        self.is_generating = False
        self.last_guidance_time = 0
        
    # ============ CLIENT PROFILE ============
    
    def update_client_profile(
        self,
        age: Optional[int] = None,
        occupation: Optional[str] = None,
        family: Optional[str] = None,
        tobacco: Optional[str] = None,
        budget: Optional[int] = None
    ):
        """Update client profile with extracted/provided info"""
        if age is not None:
            self.client.age = age
        if occupation is not None:
            self.client.occupation = occupation
        if family is not None:
            self.client.family = family
        if tobacco is not None:
            self.client.tobacco = tobacco
        if budget is not None:
            self.client.budget = budget
    
    # ============ TRANSCRIPT MANAGEMENT ============
    
    def add_transcript(self, text: str, is_final: bool = True):
        """Add transcript text to conversation memory"""
        if not text.strip():
            return
            
        # Add to full transcript
        self.full_transcript += " " + text
        
        # Trim full transcript if too long (keep recent)
        if len(self.full_transcript) > self.MAX_TRANSCRIPT_CHARS:
            # Keep last 75% to maintain context
            trim_point = len(self.full_transcript) - int(self.MAX_TRANSCRIPT_CHARS * 0.75)
            self.full_transcript = "..." + self.full_transcript[trim_point:]
        
        # Update recent window (more aggressive trimming)
        self.recent_transcript += " " + text
        if len(self.recent_transcript) > self.MAX_RECENT_CHARS:
            trim_point = len(self.recent_transcript) - int(self.MAX_RECENT_CHARS * 0.75)
            self.recent_transcript = "..." + self.recent_transcript[trim_point:]
        
        # Scan for signals (viability tracking)
        if is_final:
            self._scan_for_signals(text)
    
    def _scan_for_signals(self, text: str):
        """Scan transcript for viability signals"""
        text_lower = text.lower()
        
        for signal in POSITIVE_SIGNALS:
            if signal in text_lower and signal not in self.positive_signals:
                self.positive_signals.append(signal)
                
        for signal in BUYING_SIGNALS:
            if signal in text_lower and signal not in self.buying_signals:
                self.buying_signals.append(signal)
                self._add_key_moment(f"🎯 Buying signal: '{signal}'")
                
        for signal in HARD_NO_SIGNALS:
            if signal in text_lower and signal not in self.hard_nos:
                self.hard_nos.append(signal)
                self._add_key_moment(f"🛑 Hard no: '{signal}'")
                self.sentiment = Sentiment.HOSTILE
    
    def _add_key_moment(self, moment: str):
        """Add a key moment, keep list bounded"""
        self.key_moments.append(moment)
        if len(self.key_moments) > 20:
            self.key_moments = self.key_moments[-20:]
    
    # ============ OBJECTION TRACKING ============
    
    def detect_objection_category(self, text: str) -> Optional[str]:
        """Detect objection category from text"""
        text_lower = text.lower()
        
        for trigger, category in OBJECTION_CATEGORIES.items():
            if trigger in text_lower:
                return category
        return None
    
    def record_objection(
        self,
        category: str,
        raw_text: str,
        queue_if_busy: bool = True
    ) -> bool:
        """
        Record an objection.
        
        If currently generating guidance and queue_if_busy=True,
        adds to queue instead of making active.
        
        Returns True if objection was recorded as active.
        """
        record = ObjectionRecord(
            category=category,
            raw_text=raw_text[:200],  # Truncate
            timestamp=time.time()
        )
        
        # If we're busy generating, queue it
        if self.is_generating and queue_if_busy:
            self.objection_queue.append(record)
            return False
        
        # Otherwise make it active
        self.active_objection = category
        self.objection_history.append(record)
        
        # Trim history
        if len(self.objection_history) > self.MAX_OBJECTION_HISTORY:
            self.objection_history = self.objection_history[-self.MAX_OBJECTION_HISTORY:]
        
        # Update sentiment based on objection count
        objection_count = len(self.objection_history)
        if objection_count >= 4:
            self.sentiment = Sentiment.RESISTANT
        elif objection_count >= 2:
            if self.sentiment == Sentiment.NEUTRAL:
                self.sentiment = Sentiment.RESISTANT
        
        self._add_key_moment(f"⚠️ Objection: {category} - '{raw_text[:50]}...'")
        
        return True
    
    def process_objection_queue(self) -> Optional[ObjectionRecord]:
        """
        Process next objection from queue.
        Call this after guidance generation completes.
        
        Returns the objection if one was queued, None otherwise.
        """
        if self.objection_queue:
            record = self.objection_queue.pop(0)
            self.active_objection = record.category
            self.objection_history.append(record)
            return record
        return None
    
    def resolve_objection(self, result: ObjectionResult, guidance_text: str = ""):
        """Mark current objection as resolved"""
        if self.objection_history:
            self.objection_history[-1].result = result
            self.objection_history[-1].guidance_given = guidance_text[:200]
        
        self.active_objection = None
    
    def get_objection_pattern(self) -> Optional[str]:
        """Detect patterns in objections (e.g., serial price objections)"""
        if len(self.objection_history) < 2:
            return None
            
        # Check for same objection repeated
        recent = [o.category for o in self.objection_history[-3:]]
        if len(recent) >= 3 and len(set(recent)) == 1:
            return f"serial_{recent[0]}_objection"
        
        # Check for rapid objections (different types)
        if len(self.objection_history) >= 3:
            times = [o.timestamp for o in self.objection_history[-3:]]
            if times[-1] - times[0] < 60:  # 3 objections in 1 minute
                return "rapid_fire_objections"
        
        return None
    
    # ============ PRODUCT STATE ============
    
    def get_hierarchy_position(self) -> int:
        """Get current position in product hierarchy (1-indexed)"""
        try:
            return PRODUCT_HIERARCHY.index(self.current_product) + 1
        except ValueError:
            return 1
    
    def reject_current_product(self, reason: str, transcript_snippet: str = ""):
        """
        Mark current product as rejected and suggest next.
        
        Returns the next product to pitch, or None if exhausted.
        """
        rejection = ProductRejection(
            product=self.current_product,
            reason=reason,
            timestamp=time.time(),
            transcript_snippet=transcript_snippet[:200]
        )
        self.products_rejected[self.current_product] = rejection
        
        self._add_key_moment(
            f"❌ {PRODUCT_NAMES.get(self.current_product, self.current_product)} "
            f"rejected: {reason}"
        )
        
        return self.get_next_product()
    
    def get_next_product(self) -> Optional[str]:
        """Get next product in hierarchy that hasn't been rejected"""
        for product in PRODUCT_HIERARCHY:
            if product not in self.products_rejected and product != self.current_product:
                return product
        return None
    
    def transition_to_product(self, product: str):
        """Transition to a new product"""
        if product not in PRODUCT_HIERARCHY:
            return
            
        self.current_product = product
        if product not in self.products_pitched:
            self.products_pitched.append(product)
        
        self._add_key_moment(
            f"➡️ Transitioned to {PRODUCT_NAMES.get(product, product)}"
        )
    
    def is_hierarchy_exhausted(self) -> bool:
        """Check if all products have been rejected"""
        for product in PRODUCT_HIERARCHY:
            if product not in self.products_rejected:
                return False
        return True
    
    def get_products_remaining(self) -> List[str]:
        """Get products not yet rejected"""
        return [
            PRODUCT_NAMES.get(p, p) 
            for p in PRODUCT_HIERARCHY 
            if p not in self.products_rejected
        ]
    
    # ============ PRESENTATION PHASE TRACKING ============
    
    def advance_phase(self, phase: str = None):
        """
        Advance to next presentation phase or jump to specific phase.
        
        If phase is None, advances to the next phase in sequence.
        """
        if phase:
            if phase in PRESENTATION_PHASES:
                self.presentation_phase = phase
                self.phase_started_at = time.time()
                self._add_key_moment(f"📍 Phase: {phase.replace('_', ' ').title()}")
        else:
            # Advance to next
            try:
                current_idx = PRESENTATION_PHASES.index(self.presentation_phase)
                if current_idx < len(PRESENTATION_PHASES) - 1:
                    self.presentation_phase = PRESENTATION_PHASES[current_idx + 1]
                    self.phase_started_at = time.time()
                    self._add_key_moment(f"📍 Phase: {self.presentation_phase.replace('_', ' ').title()}")
            except ValueError:
                pass
    
    def get_phase_position(self) -> str:
        """Get current phase position (e.g., '3 of 14')"""
        try:
            idx = PRESENTATION_PHASES.index(self.presentation_phase)
            return f"{idx + 1} of {len(PRESENTATION_PHASES)}"
        except ValueError:
            return "1 of 14"
    
    def detect_phase_from_transcript(self, text: str) -> Optional[str]:
        """
        Attempt to detect presentation phase from conversation content.
        Returns phase name if detected, None otherwise.
        """
        text_lower = text.lower()
        
        # Phase detection keywords
        phase_indicators = {
            "rapport": ["where are you from", "how long at your job", "what do you do for fun"],
            "introduction": ["here today to service", "three simple things", "two simple things"],
            "company_credibility": ["since 1900", "125 years", "dallas cowboys", "atlanta braves"],
            "accidental_death": ["$3,000", "accidental death", "no cost", "certificates"],
            "referrals": ["10 slots", "sponsor", "who would you like"],
            "insurance_types": ["whole life", "term life", "difference between"],
            "needs_analysis": ["monthly income", "take home pay", "health questions"],
            "final_expense": ["funeral", "$15,000", "final expense", "cost of funeral"],
            "income_protection": ["income protection", "surviving spouse", "bills"],
            "mortgage": ["mortgage protection", "pay off the home"],
            "college": ["college education", "degree", "education cost"],
            "recap": ["to recap", "summary"],
            "close": ["which option", "option 1", "option 2"],
            "application": ["driver's license", "account information", "payment"]
        }
        
        for phase, keywords in phase_indicators.items():
            if any(kw in text_lower for kw in keywords):
                return phase
        return None
    
    # ============ DOWN-CLOSE TRACKING (Globe Life Specific) ============
    
    def do_down_close(self, reason: str = "price") -> dict:
        """
        Execute a down-close - reduce coverage to lower price point.
        
        Globe Life down-close sequence:
        1. Option 1 → Option 2 (same coverage, different pricing)
        2. $30k Final Expense → $15k
        3. $15k → $10k
        4. $10k → $7.5k
        5. 2 years income → 1 year
        6. Remove benefits one at a time
        
        Returns dict with new coverage amounts and suggested script.
        """
        if self.down_close_level >= len(DOWN_CLOSE_LEVELS) - 1:
            return {
                "exhausted": True,
                "message": "All down-close options exhausted",
                "coverage": self.coverage
            }
        
        self.down_close_level += 1
        level = DOWN_CLOSE_LEVELS[self.down_close_level]
        
        # Update coverage amounts
        old_final = self.coverage["final_expense"]
        old_income = self.coverage["income_years"]
        
        self.coverage["final_expense"] = level.get("final_expense", self.coverage["final_expense"])
        self.coverage["income_years"] = level.get("income_years", self.coverage["income_years"])
        
        # Track the down-close
        changes = []
        if self.coverage["final_expense"] < old_final:
            changes.append(f"Final Expense: ${old_final:,} → ${self.coverage['final_expense']:,}")
        if self.coverage["income_years"] < old_income:
            changes.append(f"Income Protection: {old_income} years → {self.coverage['income_years']} year")
        
        self._add_key_moment(f"⬇️ Down-close #{self.down_close_level}: {level['label']}")
        
        return {
            "exhausted": False,
            "level": self.down_close_level,
            "label": level["label"],
            "coverage": self.coverage.copy(),
            "changes": changes,
            "allow_removal": level.get("allow_removal", False)
        }
    
    def get_down_close_position(self) -> str:
        """Get current down-close position"""
        return f"{self.down_close_level} of {len(DOWN_CLOSE_LEVELS) - 1}"
    
    def is_down_close_exhausted(self) -> bool:
        """Check if all down-close options have been used"""
        return self.down_close_level >= len(DOWN_CLOSE_LEVELS) - 1
    
    def get_coverage_summary(self) -> str:
        """Get human-readable coverage summary"""
        parts = [
            f"Final Expense: ${self.coverage['final_expense']:,}",
            f"Income Protection: {self.coverage['income_years']} year{'s' if self.coverage['income_years'] > 1 else ''}"
        ]
        if self.coverage.get("mortgage"):
            parts.append("Mortgage: ✓")
        if self.coverage.get("college"):
            parts.append("College: ✓")
        return " | ".join(parts)
    
    # ============ REFERRAL TRACKING ============
    
    def add_referral(self, count: int = 1):
        """Track referrals collected during presentation"""
        self.referrals_collected += count
        self._add_key_moment(f"👥 Referrals collected: {self.referrals_collected}")
    
    # ============ PHONE CALL SPECIFIC ============
    
    def is_phone_call(self) -> bool:
        """Check if this is a phone/appointment-setting call"""
        return self.call_type == "phone"
    
    def is_presentation(self) -> bool:
        """Check if this is a full presentation call"""
        return self.call_type == "presentation"
    
    # ============ GUIDANCE TRACKING ============
    
    def record_guidance(self, text: str, trigger: str, objection_type: str = None):
        """Record guidance that was provided"""
        record = GuidanceRecord(
            text=text[:500],
            trigger=trigger[:200],
            timestamp=time.time(),
            objection_type=objection_type
        )
        self.guidance_history.append(record)
        
        if len(self.guidance_history) > self.MAX_GUIDANCE_HISTORY:
            self.guidance_history = self.guidance_history[-self.MAX_GUIDANCE_HISTORY:]
        
        self.last_guidance_time = time.time()
    
    # ============ VIABILITY SCORING (PHASE 4) ============
    
    def calculate_viability_score(self) -> int:
        """
        Calculate call viability score (0-100).
        
        70+  : Keep pushing, good chance
        40-70: Try harder but prepare exit
        <40  : Suggest graceful exit
        <20  : Hard exit recommended
        """
        score = 50  # Start neutral
        
        # Positive adjustments
        score += len(self.positive_signals) * 3
        score += len(self.buying_signals) * 15
        
        # Negative adjustments
        score -= len(self.hard_nos) * 25
        score -= len(self.products_rejected) * 8
        score -= len(self.objection_history) * 3
        
        # Pattern penalties
        pattern = self.get_objection_pattern()
        if pattern and "serial" in pattern:
            score -= 15
        if pattern == "rapid_fire_objections":
            score -= 10
            
        # Exhaustion penalty
        if self.is_hierarchy_exhausted():
            score -= 20
            
        # Sentiment adjustments
        if self.sentiment == Sentiment.HOSTILE:
            score -= 20
        elif self.sentiment == Sentiment.RESISTANT:
            score -= 5
        elif self.sentiment == Sentiment.POSITIVE:
            score += 10
        
        return max(0, min(100, score))
    
    def get_viability_recommendation(self) -> str:
        """Get recommendation based on viability score"""
        score = self.calculate_viability_score()
        
        if self.hard_nos:
            return "hard_exit"
        if score >= 70:
            return "keep_pushing"
        if score >= 40:
            return "try_harder"
        if score >= 20:
            return "graceful_exit"
        return "hard_exit"
    
    # ============ STATE EXPORT FOR CLAUDE ============
    
    def get_state_for_claude(self) -> Dict[str, Any]:
        """
        Export complete state formatted for Claude's system prompt.
        
        This is the key method - gives Claude everything it needs
        to provide intelligent, contextual guidance.
        """
        # Format objection history
        objection_summary = []
        for obj in self.objection_history[-5:]:  # Last 5
            result_str = obj.result.value if obj.result else "pending"
            objection_summary.append(f"{obj.category} ({result_str})")
        
        # Format rejection history
        rejection_summary = []
        for product, rejection in self.products_rejected.items():
            name = PRODUCT_NAMES.get(product, product)
            rejection_summary.append(f"{name}: {rejection.reason}")
        
        # Format guidance history
        recent_guidance = []
        for g in self.guidance_history[-3:]:
            recent_guidance.append(g.text[:100] + "..." if len(g.text) > 100 else g.text)
        
        # Calculate call duration
        duration_seconds = int(time.time() - self.started_at)
        duration_str = f"{duration_seconds // 60}:{duration_seconds % 60:02d}"
        
        # Phone call specific context
        phone_context = None
        if self.is_phone_call():
            phone_context = {
                "goal": "Set appointment with BOTH spouses present",
                "typical_duration": "2-3 minutes",
                "key_objections": ["timing", "spouse_presence", "send_info", "not_interested"]
            }
        
        # Presentation specific context
        presentation_context = None
        if self.is_presentation():
            presentation_context = {
                "current_phase": self.presentation_phase.replace("_", " ").title(),
                "phase_position": self.get_phase_position(),
                "coverage": self.get_coverage_summary(),
                "down_close_level": self.get_down_close_position(),
                "down_close_exhausted": self.is_down_close_exhausted(),
                "referrals_collected": self.referrals_collected
            }
        
        return {
            # Session info
            "session_id": self.session_id,
            "agency": self.agency,
            "agent_name": self.agent_name,
            "call_duration": duration_str,
            
            # Call type
            "call_type": self.call_type,
            "is_phone_call": self.is_phone_call(),
            "is_presentation": self.is_presentation(),
            "phone_context": phone_context,
            "presentation_context": presentation_context,
            
            # Client profile
            "client_profile": self.client.to_string(),
            "client_sentiment": self.sentiment.value,
            
            # Coverage state (Globe Life specific)
            "current_coverage": self.coverage.copy(),
            "coverage_summary": self.get_coverage_summary(),
            "down_close_level": self.down_close_level,
            "down_close_exhausted": self.is_down_close_exhausted(),
            
            # Product state (for supplemental products)
            "current_product": PRODUCT_NAMES.get(self.current_product, self.current_product),
            "current_product_key": self.current_product,
            "hierarchy_position": f"{self.get_hierarchy_position()} of {len(PRODUCT_HIERARCHY)}",
            "products_pitched": [PRODUCT_NAMES.get(p, p) for p in self.products_pitched],
            "products_rejected": rejection_summary,
            "products_remaining": self.get_products_remaining(),
            "hierarchy_exhausted": self.is_hierarchy_exhausted(),
            
            # Objection state
            "active_objection": self.active_objection,
            "objection_history": objection_summary,
            "objection_queue_size": len(self.objection_queue),
            "objection_pattern": self.get_objection_pattern(),
            
            # Conversation context
            "recent_transcript": self.full_transcript.strip(),  # Full conversation for better context
            "key_moments": self.key_moments[-5:],
            "recent_guidance": recent_guidance,
            
            # Viability (Phase 4)
            "viability_score": self.calculate_viability_score(),
            "viability_recommendation": self.get_viability_recommendation(),
            "buying_signals_detected": self.buying_signals,
            "hard_nos_detected": self.hard_nos,
            
            # Referrals
            "referrals_collected": self.referrals_collected,
        }
    
    def get_transcript_for_guidance(self) -> str:
        """Get the transcript window to send for guidance generation"""
        return self.recent_transcript.strip()
    
    # ============ SERIALIZATION ============
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize full state for storage/debugging"""
        return {
            "session_id": self.session_id,
            "agency": self.agency,
            "agent_name": self.agent_name,
            "started_at": self.started_at,
            "client": {
                "age": self.client.age,
                "occupation": self.client.occupation,
                "family": self.client.family,
                "tobacco": self.client.tobacco,
            },
            "sentiment": self.sentiment.value,
            "current_product": self.current_product,
            "products_pitched": self.products_pitched,
            "products_rejected": {
                k: {"reason": v.reason, "timestamp": v.timestamp}
                for k, v in self.products_rejected.items()
            },
            "objection_history": [
                {"category": o.category, "result": o.result.value, "timestamp": o.timestamp}
                for o in self.objection_history
            ],
            "key_moments": self.key_moments,
            "viability_score": self.calculate_viability_score(),
        }