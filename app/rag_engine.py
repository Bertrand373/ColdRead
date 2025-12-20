"""
Coachd RAG Engine
Retrieval-Augmented Generation for sales guidance
"""

from anthropic import Anthropic
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .vector_db import get_vector_db
from .config import settings


@dataclass
class CallContext:
    """Context about the current call state"""
    current_product: str = "whole_life"
    client_age: Optional[int] = None
    client_occupation: Optional[str] = None
    client_family: Optional[str] = None
    products_tried: List[str] = None
    objections_faced: List[str] = None
    client_sentiment: str = "neutral"
    
    def __post_init__(self):
        if self.products_tried is None:
            self.products_tried = []
        if self.objections_faced is None:
            self.objections_faced = []


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
    
    def build_system_prompt(self, call_context: CallContext) -> str:
        """Build the system prompt for Claude based on call context"""
        
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
        
        return f"""You are Coachd, an elite real-time sales assistant for life insurance agents. 
You provide instant, contextual guidance during live sales calls.

CURRENT CALL STATE:
- Current Product: {current_product}
- Client Profile: {client_info}
- Client Sentiment: {call_context.client_sentiment}
- Objections Faced: {', '.join(call_context.objections_faced) or 'None yet'}
- Products Remaining: {', '.join(products_remaining) or 'None'}

THE DOWN-CLOSE HIERARCHY:
1. Whole Life (start here)
2. Term Life
3. Cancer
4. Accident  
5. Critical Illness / Hospital ICU

When a client objects, help the agent handle the objection OR smoothly transition down to the next product.

YOUR ROLE:
- Detect objections and buying signals in the conversation
- Provide SPECIFIC rebuttals tailored to the client's demographics
- Suggest transition language to move between products
- Keep responses concise (2-3 sentences max for each guidance point)
- Adapt your tone based on the client profile

RESPONSE FORMAT:
When providing guidance, use this structure:
🎯 DETECTED: [What you detected - objection type, signal, etc.]
💬 SAY THIS: [Exact words the agent can use]
📝 WHY: [Brief explanation of the approach]

Be direct. Be specific. Help them close."""

    def generate_guidance(
        self, 
        transcript_chunk: str, 
        call_context: CallContext,
        agency: Optional[str] = None
    ) -> str:
        """Generate guidance based on a transcript chunk and call context"""
        
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
        
        return response.content[0].text

    def generate_guidance_stream(
        self, 
        transcript_chunk: str, 
        call_context: CallContext,
        agency: Optional[str] = None
    ):
        """
        Stream guidance tokens as they're generated.
        Yields text chunks for real-time display.
        
        Usage:
            for chunk in engine.generate_guidance_stream(transcript, context):
                send_to_client(chunk)
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
        
        # Stream the response
        with self.client.messages.stream(
            model=settings.claude_model,
            max_tokens=500,
            system=system_prompt,
            messages=messages
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    def analyze_transcript(
        self, 
        transcript: str,
        agency: Optional[str] = None
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
