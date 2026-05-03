import asyncio
import json
import argparse
from pathlib import Path

from router import classify_question
from parallel import run_map_reduce
from reflection import reflection_loop
from utils.logger import header, stage, info, divider, ok


SAMPLES_PATH = Path(__file__).parent / "data" / "sample_questions.json"
RESULTS_PATH = Path(__file__).parent / "data" / "results.json"

FALLBACK_RESPONSE = (
    "I'm sorry, but I'm unable to process this request. "
    "It either falls outside the scope of this research assistant, "
    "contains disallowed content, or appears to be an adversarial input. "
    "Please ask a genuine research question."
)


async def process_question(question: str, qid: int = 0) -> dict:
    header(f"QUESTION #{qid}  |  {question[:65]}{'...' if len(question) > 65 else ''}")

    routing = await classify_question(question)
    domain  = routing.get("domain", "fallback")

    if domain == "fallback" or routing.get("guardrail_triggered"):
        stage("ROUTER", "Question routed to FALLBACK — returning graceful refusal")
        reason = routing.get("guardrail_reason") or routing.get("reasoning", "out of scope")
        info("Reason:", reason)
        return {
            "question_id": qid,
            "question": question,
            "domain": "fallback",
            "guardrail_triggered": routing.get("guardrail_triggered", False),
            "final_response": FALLBACK_RESPONSE,
            "skipped_pipeline": True,
        }

    brief, map_meta = await run_map_reduce(question, domain)

    info("Sub-questions answered:", str(len(map_meta["sub_questions"])))
    info("Parallel time:", f"{map_meta['parallel_time_s']}s")
    info("Est. sequential:", f"{map_meta['estimated_sequential_time_s']}s")
    info("Speedup:", f"~{map_meta['speedup']}x")

    final_brief, reflection_log = await reflection_loop(brief, question, domain)

    divider()
    stage("SUMMARY", "Score progression across reflection iterations")
    for entry in reflection_log:
        info(f"  Iteration {entry['iteration']}:", f"{entry['aggregate_score']:.1f}/10")

    print(f"\n  FINAL RESEARCH BRIEF:\n")
    print(f"  {final_brief[:600]}{'...' if len(final_brief) > 600 else ''}\n")

    return {
        "question_id": qid,
        "question": question,
        "domain": domain,
        "routing": routing,
        "map_reduce_metadata": map_meta,
        "reflection_log": reflection_log,
        "final_response": final_brief,
        "skipped_pipeline": False,
    }


async def main():
    parser = argparse.ArgumentParser(description="Deep Research Assistant")
    parser.add_argument("--question", type=str, help="Run a single custom question")
    parser.add_argument("--id",       type=int, help="Run question by ID from sample_questions.json")
    parser.add_argument("--all",      action="store_true", help="Run all sample questions")
    parser.add_argument("--eval",     action="store_true", help="Run routing accuracy eval")
    args = parser.parse_args()

    if args.eval:
        from eval.run_eval import run_eval
        await run_eval()
        return

    questions = json.loads(SAMPLES_PATH.read_text())
    results   = []

    if args.question:
        result = await process_question(args.question, qid=0)
        results.append(result)

    elif args.id:
        q = next((q for q in questions if q["id"] == args.id), None)
        if not q:
            print(f"No question with id={args.id}")
            return
        result = await process_question(q["question"], qid=q["id"])
        results.append(result)

    elif args.all:
        for q in questions:
            result = await process_question(q["question"], qid=q["id"])
            results.append(result)

    else:
        for q in questions[:3]:
            result = await process_question(q["question"], qid=q["id"])
            results.append(result)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ok(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())