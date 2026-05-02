from pathlib import Path
from pathlib import Path
from utils.llm_client import call_json
from utils.logger import stage, sub, info, warn

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "router.txt"

VALID_DOMAINS = {"scientific", "historical", "financial", "general", "fallback"}


async def classify_question(question: str) -> dict:
    stage("ROUTER", "Classifying research question")
    sub(f"Question: {question[:80]}{'...' if len(question) > 80 else ''}")

    system = PROMPT_PATH.read_text()
    prompt = f"Classify this research question:\n\n{question}"

    try:
        result = await call_json(prompt, system=system, use_critic=False)
    except Exception as e:
        warn(f"Router LLM call failed: {e} — defaulting to fallback")
        return {
            "domain": "fallback",
            "confidence": 0.0,
            "reasoning": "Router error — defaulting to fallback",
            "guardrail_triggered": False,
            "guardrail_reason": None,
        }

    # Validate domain
    domain = result.get("domain", "fallback").lower()
    if domain not in VALID_DOMAINS:
        domain = "fallback"
    result["domain"] = domain

    # Log result
    if result.get("guardrail_triggered"):
        warn(f"GUARDRAIL TRIGGERED: {result.get('guardrail_reason')}")
    else:
        info("Domain:", domain)
        info("Confidence:", f"{result.get('confidence', 0):.0%}")
        info("Reasoning:", result.get("reasoning", "")[:60])

    return result