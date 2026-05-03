import asyncio
import json
from pathlib import Path
from collections import defaultdict

from router.supervisor import classify_question
from utils.logger import header, info, divider


EVAL_PATH = Path(__file__).parent / "eval_set.json"
DOMAINS = ["scientific", "historical", "financial", "general", "fallback"]


async def run_eval():
    header("ROUTING ACCURACY EVAL")
    questions = json.loads(EVAL_PATH.read_text())

    results = []
    for q in questions:
        result = await classify_question(q["question"])
        predicted = result.get("domain", "fallback")
        expected  = q["expected_domain"]
        correct   = predicted == expected
        results.append({
            "id": q["id"],
            "question": q["question"][:60],
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
        })
        status = "✓" if correct else "✗"
        print(f"  [{status}] Q{q['id']:02d} expected={expected:<12} predicted={predicted}")

    divider()

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\n  Overall accuracy: {correct}/{total} = {correct/total:.0%}")

    print("\n  Per-class precision:")
    for domain in DOMAINS:
        predicted_as = [r for r in results if r["predicted"] == domain]
        true_pos     = [r for r in predicted_as if r["expected"] == domain]
        precision    = len(true_pos) / len(predicted_as) if predicted_as else 0.0
        print(f"    {domain:<12} precision={precision:.0%}  ({len(true_pos)}/{len(predicted_as)} correct predictions)")

    print("\n  Per-class recall:")
    for domain in DOMAINS:
        actual      = [r for r in results if r["expected"] == domain]
        true_pos    = [r for r in actual if r["predicted"] == domain]
        recall      = len(true_pos) / len(actual) if actual else 0.0
        print(f"    {domain:<12} recall={recall:.0%}  ({len(true_pos)}/{len(actual)} found)")

    return results


if __name__ == "__main__":
    asyncio.run(run_eval())