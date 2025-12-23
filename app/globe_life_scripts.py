"""
Globe Life Script Templates
Locked verbatim scripts from Globe Life training docs.
AI fills in context slots - NEVER changes the structure.
"""

# =============================================================================
# PRESENTATION OBJECTION SCRIPTS
# =============================================================================

SCRIPTS = {
    "stall": {
        "name": "Think About It",
        "template": """I completely understand, after all this is a big decision for {family_reference} and you have to do it right. I help a lot of families each day, and I help a lot of people think about this so they can make the best choice for their family. When folks think about it, no matter how many things they may be thinking of, it usually just comes down to one or two things; either do they need this? Or, can they afford this?

So let me ask you, do you feel this is something that {family_reference} needs? Do you need your funeral protected, your income protected, your mortgage paid off, and your {children_reference} sent to college? Would they be able to cover that financially on their own if something were to happen to you?""",
        "fallback": """I completely understand, after all this is a big decision for your family and you have to do it right. I help a lot of families each day, and I help a lot of people think about this so they can make the best choice for their family. When folks think about it, no matter how many things they may be thinking of, it usually just comes down to one or two things; either do they need this? Or, can they afford this?

So let me ask you, do you feel this is something that your family needs? Do you need your funeral protected, your income protected, your mortgage paid off, and your children sent to college? Would they be able to cover that financially on their own if something were to happen to you?""",
    },

    "price": {
        "name": "Can't Afford It",
        "template": """Ok, I understand. {context_acknowledgment} Let me ask you, you understand how important this is for {family_reference}, right? So here's what we can do: let's adjust the Final Expense benefit to ${down_close_amount}. That should help cover the cost of a funeral. Your income protection will remain the same, along with your mortgage coverage and the college education benefit. Let's recalculate and see where we're at. Does this make you feel better?""",
        "fallback": """Ok, I understand. Let me ask you, you understand how important this is, right? So here's what we can do: let's adjust the Final Expense benefit to $15,000. That should help cover the cost of a funeral. Your income protection will remain the same, along with your mortgage coverage and the college education benefit. Let's recalculate and see where we're at. Does this make you feel better?""",
    },

    "covered": {
        "name": "Already Has Insurance",
        "template": """Yes, and the Needs Analysis has already taken that into consideration, and it has clearly exposed the need for more coverage. Simply put, what you currently have {existing_coverage_reference} is good, but it's just not enough to make sure {family_reference} is protected. And it's a lot better to find out that your protection is too little now than when it's too late because you can do something about it now. The last thing you'd want to do is leave {family_reference} open to serious financial strain when you are not around anymore to protect them. With either option you pick, {family_reference} is going to be protected, so which one works best?""",
        "fallback": """Yes, and the Needs Analysis has already taken that into consideration, and it has clearly exposed the need for more coverage. Simply put, what you currently have is good, but it's just not enough to make sure your family is protected. And it's a lot better to find out that your protection is too little now than when it's too late because you can do something about it now. The last thing you'd want to do is leave your family open to serious financial strain when you are not around anymore to protect them. With either option you pick, your family is going to be protected, so which one works best?""",
    },

    "spouse": {
        "name": "Need to Talk to Spouse",
        "template": """Of course, I understand wanting to discuss this with {spouse_name}. But let me ask you - what do YOU think about it so far? {personalized_hook}""",
        "fallback": """Of course, I understand wanting to discuss this together. But let me ask you - what do YOU think about it so far? Does the coverage make sense for your family's needs?""",
    },

    "need": {
        "name": "Don't Need It / Invincible",
        "template": """That's a fair question, and I appreciate you being direct with me. Let me ask you this - do you know what the average cost of a funeral is today? It's around $15,000 and going up every year. {personalized_hook}

Here's the thing - it's not about whether something WILL happen, it's about making sure {family_reference} isn't left with a financial burden IF something does. The question isn't really "do I need this?" - it's "would {family_reference} be able to handle $15,000 in funeral costs, plus the mortgage, plus all the bills, if your income suddenly stopped?"

The people who need this coverage the most are often the ones who think they don't - because they're healthy, they're working, they're taking care of their family. That's exactly why NOW is the best time to lock in your rate. Does that make sense?""",
        "fallback": """That's a fair question, and I appreciate you being direct with me. Let me ask you this - do you know what the average cost of a funeral is today? It's around $15,000 and going up every year.

Here's the thing - it's not about whether something WILL happen, it's about making sure your family isn't left with a financial burden IF something does. The question isn't really "do I need this?" - it's "would your family be able to handle $15,000 in funeral costs, plus the mortgage, plus all the bills, if your income suddenly stopped?"

The people who need this coverage the most are often the ones who think they don't - because they're healthy, they're working, they're taking care of their family. That's exactly why NOW is the best time to lock in your rate. Does that make sense?""",
    },
}


# =============================================================================
# DOWN-CLOSE SCRIPTS (Sequential - use based on down_close_level)
# =============================================================================

DOWN_CLOSE_SCRIPTS = {
    1: {
        "name": "Drop 1: Option 1 to Option 2 / $30k to $15k",
        "amount": 15000,
        "template": """Ok, I understand. {context_acknowledgment} Let me ask you, you understand how important this is for {family_reference}, right? So here's what we can do: let's adjust the Final Expense benefit to $15,000. That should help cover the cost of a funeral. Your income protection will remain the same, along with your mortgage coverage and the college education benefit. Let's recalculate and see where we're at. Does this make you feel better?""",
        "fallback": """Ok, I understand. Let me ask you, you understand how important this is, right? So here's what we can do: let's adjust the Final Expense benefit to $15,000. That should help cover the cost of a funeral. Your income protection will remain the same, along with your mortgage coverage and the college education benefit. Let's recalculate and see where we're at. Does this make you feel better?""",
    },

    2: {
        "name": "Drop 2: $15k to $10k",
        "amount": 10000,
        "template": """I know you understand how important this is for {family_reference}, so here's what we can do: let's adjust this Final Expense benefit to $10,000. As you recall from the video, the cost of a final expense is between $7,500 and $15,000 currently, so this will make sure you are covered right in the middle of the current cost. That should still help cover the cost of the funeral. Again your income protection will remain the same, along with your mortgage coverage and the college education benefit. Let's recalculate and see where we're at. Does this make you feel better?""",
        "fallback": """I know you understand how important this is, so here's what we can do: let's adjust this Final Expense benefit to $10,000. As you recall from the video, the cost of a final expense is between $7,500 and $15,000 currently, so this will make sure you are covered right in the middle of the current cost. That should still help cover the cost of the funeral. Again your income protection will remain the same, along with your mortgage coverage and the college education benefit. Let's recalculate and see where we're at. Does this make you feel better?""",
    },

    3: {
        "name": "Drop 3: $10k to $7.5k",
        "amount": 7500,
        "template": """I understand and I appreciate your concern{context_acknowledgment}. As you recall from the video, the basic cost of a funeral is approximately $7,500. Let's adjust this Final Expense benefit to $7,500 and leave the rest of your coverage as is to protect {family_possessive} income, mortgage, and college education. Let's recalculate and see where we're at. Now you've got to be feeling better about that, right?""",
        "fallback": """I understand and I appreciate your concern. As you recall from the video, the basic cost of a funeral is approximately $7,500. Let's adjust this Final Expense benefit to $7,500 and leave the rest of your coverage as is to protect your income, mortgage, and college education. Let's recalculate and see where we're at. Now you've got to be feeling better about that, right?""",
    },

    4: {
        "name": "Drop 4: Reduce Income Protection to 1 Year",
        "amount": 7500,
        "template": """To make sure {family_possessive} needs are covered, we will leave the Final Expense benefit at $7,500 and adjust {income_earner_possessive} income protection from 2 years to 1 year. That way, your needs are still covered, and when things get better, you can always increase them later. Let's recalculate and see where we're at. I'm sure you are feeling much better about that now, right?""",
        "fallback": """To make sure your needs are covered, we will leave the Final Expense benefit at $7,500 and adjust your income protection from 2 years to 1 year. That way, your needs are still covered, and when things get better, you can always increase them later. Let's recalculate and see where we're at. I'm sure you are feeling much better about that now, right?""",
    },

    5: {
        "name": "Drop 5: Pick Most Important",
        "amount": None,
        "template": """{client_names}, if something were to happen unexpectedly, which one of these obligations do you feel you could go without for the immediate time being, knowing that we can add it on later?""",
        "fallback": """If something were to happen unexpectedly, which one of these obligations do you feel you could go without for the immediate time being, knowing that we can add it on later?""",
    },
}


# =============================================================================
# PHONE OBJECTION SCRIPTS (Already complete - no AI needed)
# =============================================================================

PHONE_SCRIPTS = {
    "timing": "I have a lot of people on my schedule to get this out today, so it doesn't take long - for me the quicker, the better.",
    
    "spouse_presence": "They ask us to make sure we speak with both of you to answer any questions and to make sure you know how this works in case something happens to either of you.",
    
    "send_info": "I sure wish we could. That would make my job a lot easier. However, we need to speak with you to be able to designate your beneficiaries for the no-cost coverage and review everything that you're entitled to through the program.",
    
    "not_interested": "Well that's OK, but you're still entitled to the no-cost coverage. I can still get that out to you and explain everything that you're entitled to through the program. Most people appreciate the short time it takes to explain everything to them.",
    
    "sales_skepticism": "It's our job to review your policy with you and renew your no-cost coverage. We have an appointment every half hour so it doesn't take very much of your time.",
}


# =============================================================================
# CONTEXT EXTRACTION PROMPT
# =============================================================================

CONTEXTUALIZE_PROMPT = """You are filling in a Globe Life sales script with context from the conversation.

SCRIPT TEMPLATE:
{template}

TRANSCRIPT CONTEXT:
{transcript}

YOUR TASK:
Fill in these context slots based on the transcript. If you can't find specific info, use the generic fallback shown.

SLOTS TO FILL:
- {{family_reference}} = Client's family (e.g., "Maria and the kids", "your family")
- {{family_possessive}} = Possessive form (e.g., "your family's", "Maria and the kids'")
- {{spouse_name}} = Spouse's name if mentioned (e.g., "David", "your wife")
- {{children_reference}} = Children reference (e.g., "Tyler and Sofia", "your children")
- {{client_names}} = Client name(s) (e.g., "Nate and Julie", "Maria")
- {{context_acknowledgment}} = Brief acknowledgment of something they said (e.g., ", especially with the holidays coming up" or leave empty)
- {{existing_coverage_reference}} = Reference to their existing coverage if mentioned (e.g., "through your job at Amazon")
- {{income_earner_possessive}} = Primary earner's possessive (e.g., "David's", "your")
- {{personalized_hook}} = A question based on what they've shared (e.g., "With David working nights at the warehouse, would Maria be able to handle things financially?")
- {{down_close_amount}} = Keep as shown in template

RULES:
1. NEVER change the script structure or key phrases
2. Only fill in the context slots
3. If no specific context found for a slot, use natural generic language
4. Keep acknowledgments brief (under 10 words)
5. Output ONLY the completed script, nothing else

COMPLETED SCRIPT:"""


def get_script(objection_type: str, down_close_level: int = 0) -> dict:
    """
    Get the appropriate script template for an objection.
    
    Args:
        objection_type: Type of objection (price, stall, covered, spouse)
        down_close_level: Current down-close level (0-5) for price objections
    
    Returns:
        dict with 'template' and 'fallback' keys
    """
    # For price objections, use down-close scripts based on level
    if objection_type == "price":
        # Determine which down-close script to use
        level = min(down_close_level + 1, 5)  # Next level, max 5
        if level in DOWN_CLOSE_SCRIPTS:
            return DOWN_CLOSE_SCRIPTS[level]
        return DOWN_CLOSE_SCRIPTS[1]  # Default to first down-close
    
    # For other objections, use standard scripts
    if objection_type in SCRIPTS:
        return SCRIPTS[objection_type]
    
    # Fallback - shouldn't happen
    return {
        "template": "I understand your concern. Can you tell me more about what's holding you back?",
        "fallback": "I understand your concern. Can you tell me more about what's holding you back?",
    }


def get_phone_script(objection_type: str) -> str:
    """Get phone script - no AI needed, return verbatim."""
    return PHONE_SCRIPTS.get(objection_type, "I understand. Let me explain how this works...")