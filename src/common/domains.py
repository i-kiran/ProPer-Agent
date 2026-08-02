"""Per-domain field accessors.

Every domain stores its raw examples differently:

    code     {"id", "query", "solution", ...}
    medical  {"id", "utterances": ["patient: ...", "doctor: ..."]}
    pwab     {"id", "instructions", "user_profile": {...}, "target_product": {...}}

Every other module talks to raw records only through the functions here, so the
domain conditionals live in exactly one file.
"""

from __future__ import annotations

DOMAINS = ("code", "medical", "pwab")

#: Task framing prepended to every RGA SFT instruction.
RGA_INSTRUCTION = {
    "code": "Given the user's request, produce the minimal correct code solution",
    "medical": (
        "Use the user's question to provide brief, general medical/diagnoses or "
        "treatment."
    ),
    "pwab": (
        "Use the user's request to deliver concise, non-personalized item "
        "suggestions."
    ),
}

#: Output-format rule appended to every RGA SFT instruction. The base pass and
#: the rewrite pass both key off these markers.
RGA_FORMAT_RULE = (
    "Be short and concise. Always produce output wrapped exactly as:\n"
    "===START===\n<output>\n===END===\n"
)


def _check(domain):
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")


def query(rec, domain):
    """The user's question, verbatim."""
    _check(domain)
    if domain == "code":
        return rec["query"]
    if domain == "medical":
        return rec["utterances"][0].split(":", 1)[-1].strip()
    return rec["instructions"]


def gold(rec, domain):
    """The reference response the RGA is trained to reproduce."""
    _check(domain)
    if domain == "code":
        return rec["solution"]
    if domain == "medical":
        utterances = rec["utterances"]
        return utterances[1].split(":", 1)[-1].strip() if len(utterances) > 1 else ""
    return ", ".join(f"{k}: {v}" for k, v in rec["target_product"].items())


def persona(rec, domain):
    """Long-lived user traits. Only pwab has them; the rest render empty."""
    _check(domain)
    if domain == "pwab":
        return ", ".join(f"{k}: {v}" for k, v in rec["user_profile"].items())
    return ""


def rga_instruction(domain):
    _check(domain)
    return RGA_INSTRUCTION[domain] + RGA_FORMAT_RULE


def teacher_fields(rec, domain):
    """Slot values for rendering prompts/{domain}/teacher_label.txt."""
    return {
        "persona": persona(rec, domain),
        "query": query(rec, domain),
        "gold_response": gold(rec, domain),
    }


def teacher_input_text(rec, domain):
    """The user turn sent to the teacher model by src/data/label_gpt.py.

    Kept byte-compatible with the format that produced the released labels.
    """
    _check(domain)
    if domain == "code":
        return f"Query: {rec['query']}\nSolution: {rec['solution']}"
    if domain == "medical":
        return "Utterances: " + "\n".join(rec["utterances"]) + "\n"
    return (
        f"- User Persona: {persona(rec, domain)}\n"
        f"- Query: {query(rec, domain)}\n"
        f"- Product: {gold(rec, domain)}\n"
    )
