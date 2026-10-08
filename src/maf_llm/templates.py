"""Prompt Templates for Lead Summarization.

Structured prompts for distilling normalized text into unified briefs.
"""

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Container for prompt templates."""

    name: str
    system_prompt: str
    user_template: str
    max_input_tokens: int = 8000
    max_output_tokens: int = 2048
    temperature: float = 0.3


# System prompts for different tasks
SUMMARIZATION_SYSTEM = """You are a lead summarization AI for an automotive service business.

Your job is to distill raw lead text into clear, actionable briefs that help mechanics understand customer needs quickly.

OUTPUT RULES:
1. Be concise - briefs are read quickly
2. Extract key facts only - no speculation
3. Highlight urgency signals (ASAP, urgent, broken, won't start)
4. Identify vehicle info (make, model, year, engine)
5. Note location/service area
6. Preserve contact info if present
7. Use structured format for easy scanning

CRITICAL: Only include information present in the source text. Do not hallucinate or infer details not explicitly stated."""

SUMMARIZATION_USER = """Summarize this lead into a structured brief.

SOURCE TEXT:
{input_text}

OUTPUT FORMAT:
## Customer Need
[1-2 sentences describing what they need]

## Vehicle
[Year/Make/Model if mentioned, or "Not specified"]

## Location
[City/Area if mentioned, or "Not specified"]

## Urgency
[High/Medium/Low based on signals in text]

## Key Details
- [Detail 1]
- [Detail 2]
- [etc]

## Contact
[Phone/Email if provided, or "Not provided"]

## Service Type
[Diagnostic/Repair/Maintenance/Unknown]

## Action Required
[What the mechanic needs to do next]

## Confidence
[High/Medium/Low - how certain you are about the above extraction]"""

EXTRACTION_SYSTEM = """You are a data extraction AI for lead processing.

Extract structured information from raw text into JSON format.

RULES:
1. Extract only what is explicitly stated
2. Use null for missing values
3. Normalize formats (phone, email, etc.)
4. Identify vehicle specifics (make, model, year, engine)
5. Extract keywords and topics

OUTPUT: Valid JSON only, no other text."""

EXTRACTION_USER = """Extract information from this lead.

SOURCE TEXT:
{input_text}

OUTPUT (JSON):
{{
  "customer_name": null or string,
  "contact": {{
    "phone": null or string (normalized: XXX-XXX-XXXX),
    "email": null or string
  }},
  "vehicle": {{
    "year": null or integer,
    "make": null or string,
    "model": null or string,
    "engine": null or string,
    "vin": null or string
  }},
  "location": {{
    "city": null or string,
    "state": null or string,
    "zip": null or string
  }},
  "service_needed": null or string,
  "urgency": "high" | "medium" | "low" | null,
  "urgency_signals": [],
  "keywords": [],
  "source_platform": null or string,
  "timestamp_mentioned": null or string
}}"""

PRIORITY_SYSTEM = """You are a lead priority classifier for an automotive service business.

Classify leads by priority based on:
1. URGENCY signals (ASAP, urgent, broken, won't start, emergency)
2. VEHICLE type (BMW, Mercedes, Ram, diesel = higher priority)
3. SERVICE type (diagnostic, mobile repair = medium, general inquiry = low)
4. CONTACT completeness (phone + location = higher)

PRIORITY LEVELS:
- P1 (Critical): Emergency, won't start, safety issue, immediate response needed
- P2 (High): Same-day service, diagnostic needed, customer is ready to book
- P3 (Medium): Scheduled service, quote request, general inquiry
- P4 (Low): Tire kicker, future interest, incomplete info
- P5 (Noise): Not a valid lead, job posting, spam

OUTPUT: JSON with priority level and reasoning."""

PRIORITY_USER = """Classify the priority of this lead.

SOURCE TEXT:
{input_text}

OUTPUT (JSON):
{{
  "priority": 1-5,
  "priority_label": "P1-Critical" | "P2-High" | "P3-Medium" | "P4-Low" | "P5-Noise",
  "reasoning": "Brief explanation",
  "urgency_signals": [],
  "vehicle_priority": true/false,
  "service_type": "diagnostic" | "repair" | "maintenance" | "inquiry" | "other"
}}"""

INTENT_SYSTEM = """You are a lead intent classifier for lead qualification.

Classify the customer's intent:
- HIGH: Ready to book, wants service now, has vehicle and contact info
- MEDIUM: Considering service, asking questions, comparison shopping
- LOW: Just browsing, future interest, incomplete information
- NOISE: Not a customer (job seeker, spam, advertisement)

OUTPUT: JSON with intent classification and signals."""

INTENT_USER = """Classify the intent of this lead.

SOURCE TEXT:
{input_text}

OUTPUT (JSON):
{{
  "intent": "high" | "medium" | "low" | "noise",
  "intent_signals": [],
  "ready_to_book": true/false,
  "has_budget": true/false/null,
  "timeline": "immediate" | "days" | "weeks" | "future" | null
}}"""

VERIFICATION_SYSTEM = """You are a fact verification AI.

Verify that summary points are grounded in source text.

RULES:
1. Each summary point MUST be directly supported by source text
2. Flag any inference or speculation
3. If information is not in source, mark as "NOT VERIFIED"
4. Preserve exact wording where possible

OUTPUT: JSON with verification status for each point."""

VERIFICATION_USER = """Verify these summary points against the source text.

SOURCE TEXT:
{source_text}

SUMMARY POINTS:
{summary_points}

OUTPUT (JSON):
{{
  "verified_points": [
    {{
      "point": "string",
      "status": "verified" | "partial" | "not_verified",
      "evidence": "exact quote from source",
      "confidence": 0.0-1.0
    }}
  ],
  "overall_confidence": 0.0-1.0,
  "missing_info": []
}}"""

BRIEF_GENERATION_SYSTEM = """You are a lead brief generator for mechanics.

Create a professional, scannable brief that helps a mechanic quickly understand:
1. What the customer needs
2. What vehicle
3. Where they are
4. How urgent
5. What to do next

Keep it under 200 words. Use bullet points. Be direct."""

BRIEF_GENERATION_USER = """Generate a brief for this processed lead.

LEAD DATA:
{lead_data}

OUTPUT: A brief, scannable summary for the mechanic."""


# Template registry
TEMPLATES = {
    "summarization": PromptTemplate(
        name="summarization",
        system_prompt=SUMMARIZATION_SYSTEM,
        user_template=SUMMARIZATION_USER,
        max_input_tokens=8000,
        max_output_tokens=2048,
        temperature=0.3,
    ),
    "extraction": PromptTemplate(
        name="extraction",
        system_prompt=EXTRACTION_SYSTEM,
        user_template=EXTRACTION_USER,
        max_input_tokens=8000,
        max_output_tokens=1024,
        temperature=0.1,
    ),
    "priority": PromptTemplate(
        name="priority",
        system_prompt=PRIORITY_SYSTEM,
        user_template=PRIORITY_USER,
        max_input_tokens=4000,
        max_output_tokens=512,
        temperature=0.2,
    ),
    "intent": PromptTemplate(
        name="intent",
        system_prompt=INTENT_SYSTEM,
        user_template=INTENT_USER,
        max_input_tokens=4000,
        max_output_tokens=512,
        temperature=0.2,
    ),
    "verification": PromptTemplate(
        name="verification",
        system_prompt=VERIFICATION_SYSTEM,
        user_template=VERIFICATION_USER,
        max_input_tokens=6000,
        max_output_tokens=1024,
        temperature=0.1,
    ),
    "brief": PromptTemplate(
        name="brief",
        system_prompt=BRIEF_GENERATION_SYSTEM,
        user_template=BRIEF_GENERATION_USER,
        max_input_tokens=4000,
        max_output_tokens=1024,
        temperature=0.3,
    ),
}


def get_template(name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]


def format_prompt(template: PromptTemplate, **kwargs) -> tuple[str, str]:
    """Format a prompt template with variables.

    Returns:
        tuple: (system_prompt, user_prompt)
    """
    user_prompt = template.user_template.format(**kwargs)
    return template.system_prompt, user_prompt
