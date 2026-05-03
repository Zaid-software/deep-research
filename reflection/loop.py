import difflib
import json
from pathlib import Path
from utils.llm_client import call_producer, call_json
from utils.logger import stage, sub, info, divider, warn

PROMPTS = Path(__file__).parent.parent / "prompts" / "reflection"

SCORE_THRESHOLD = 8.0
MAX_ITERATIONS  = 3
PLATEAU_DELTA   = 0.2


def _load(filename: str) -> str:
    return (PROMPTS / filename).read_text()


def _print_diff(before: str, after: str, label_before: str, label_after: str):
    before_lines = before.splitlines(keepends=True)
    after_lines  = after.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile=label_before,
        tofile=label_after,
        lineterm="",
    ))
    if diff:
        print("\n  [DIFF]")
        for line in diff[:60]:
            prefix = "  "
            if line.startswith("+"):
                prefix = "  +"
            elif line.startswith("-"):
                prefix = "  -"
            print(f"{prefix}{line.rstrip()}")
    else:
        print("  [DIFF] No textual changes detected.")


async def critique(draft: str, question: str, domain: str) -> dict:
    template = _load("critic.txt")
    prompt = template.format(question=question, domain=domain, draft=draft)

    try:
        result = await call_json(prompt, use_critic=True)
        scores = result.get("scores", {})
        if scores and "aggregate_score" not in result:
            result["aggregate_score"] = round(sum(scores.values()) / len(scores), 2)
        return result
    except Exception as e:
        warn(f"Critic call failed: {e}")
        return {
            "scores": {},
            "aggregate_score": 5.0,
            "weaknesses": ["Critic unavailable"],
            "revision_instructions": "Improve clarity and factual grounding.",
        }


async def revise(draft: str, critique_result: dict, question: str, domain: str) -> str:
    template = _load("revise.txt")
    prompt = template.format(
        question=question,
        domain=domain,
        draft=draft,
        critique=json.dumps(critique_result, indent=2),
    )
    return await call_producer(prompt)


async def reflection_loop(
    initial_brief: str,
    question: str,
    domain: str,
) -> tuple[str, list[dict]]:
    stage("REFLECT", f"Starting producer-critic loop (max {MAX_ITERATIONS} iterations)")
    info("Score threshold:", str(SCORE_THRESHOLD))
    info("Plateau delta:", str(PLATEAU_DELTA))

    current_draft = initial_brief
    log = []
    prev_score = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        stage("REFLECT", f"Iteration {iteration}/{MAX_ITERATIONS}")
        sub("Critic evaluating draft...")
        critique_result = await critique(current_draft, question, domain)

        scores     = critique_result.get("scores", {})
        agg_score  = critique_result.get("aggregate_score", 0)
        weaknesses = critique_result.get("weaknesses", [])
        revisions  = critique_result.get("revision_instructions", "")

        info("Aggregate score:", f"{agg_score:.1f}/10")
        for dim, val in scores.items():
            info(f"  {dim}:", f"{val}/10")
        for w in weaknesses:
            sub(f"Weakness: {w}")

        log.append({
            "iteration": iteration,
            "scores": scores,
            "aggregate_score": agg_score,
            "weaknesses": weaknesses,
            "revision_instructions": revisions,
            "draft": current_draft,
        })

        if agg_score >= SCORE_THRESHOLD:
            sub(f"Score {agg_score:.1f} ≥ threshold {SCORE_THRESHOLD} — stopping early")
            break

        if prev_score is not None:
            delta = agg_score - prev_score
            if delta < PLATEAU_DELTA:
                warn(f"Plateau detected (Δ={delta:.2f} < {PLATEAU_DELTA}) — stopping loop")
                break

        prev_score = agg_score

        sub("Producer revising draft...")
        new_draft = await revise(current_draft, critique_result, question, domain)

        divider()
        _print_diff(
            current_draft, new_draft,
            f"iteration-{iteration - 1}",
            f"iteration-{iteration}",
        )
        divider()

        current_draft = new_draft
    stage("REFLECT", "Loop complete")
    if log:
        first_score = log[0]["aggregate_score"]
        last_score  = log[-1]["aggregate_score"]
        improvement = last_score - first_score
        info("First score:", f"{first_score:.1f}/10")
        info("Final score:", f"{last_score:.1f}/10")
        info("Improvement:", f"+{improvement:.1f}" if improvement >= 0 else str(improvement))

    return current_draft, log