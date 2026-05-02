import asyncio
import time
import json
from pathlib import Path
from utils.llm_client import call_producer, call_json
from utils.logger import stage, sub, info, divider, warn

PROMPTS = Path(__file__).parent.parent / "prompts"
N_CANDIDATES = 3


def _load(filename: str) -> str:
    return (PROMPTS / filename).read_text()


async def decompose(question: str) -> list[str]:
    stage("MAP-REDUCE", "Decomposing question into sub-questions")
    system = _load("parallel/decompose.txt")
    result = await call_json(question, system=system)
    sub_questions = result.get("sub_questions", [question])
    for i, sq in enumerate(sub_questions, 1):
        sub(f"Sub-question {i}: {sq}")
    return sub_questions


async def generate_candidate(sub_question: str, domain_system: str, index: int) -> str:
    try:
        answer = await call_producer(sub_question, system=domain_system)
        return answer
    except Exception as e:
        warn(f"Candidate {index} failed: {e}")
        return None


async def judge_candidates(sub_question: str, candidates: list[str]) -> tuple[str, list]:
    valid = [(i, c) for i, c in enumerate(candidates) if c is not None]
    if not valid:
        return "No valid candidates generated.", []

    if len(valid) == 1:
        return valid[0][1], [{"candidate_index": valid[0][0], "note": "only valid candidate"}]

    system = _load("parallel/judge.txt")
    candidates_text = "\n\n".join(
        f"--- Candidate {i} ---\n{c}" for i, c in valid
    )
    prompt = f"Sub-question: {sub_question}\n\nCandidates:\n{candidates_text}"

    try:
        result = await call_json(prompt, system=system, use_critic=True)
        best_idx = result.get("best_index", valid[0][0])
        scores = result.get("scores", [])
        reasoning = result.get("reasoning", "")

        best_text = next((c for i, c in valid if i == best_idx), valid[0][1])
        return best_text, scores
    except Exception as e:
        warn(f"Judge failed: {e} — using first valid candidate")
        return valid[0][1], []


async def best_of_n(sub_question: str, domain_system: str, sq_index: int) -> tuple[str, list]:
    candidates = await asyncio.gather(
        *[generate_candidate(sub_question, domain_system, i) for i in range(N_CANDIDATES)],
        return_exceptions=False,
    )

    best, scores = await judge_candidates(sub_question, list(candidates))
    return best, scores


async def fan_out(sub_questions: list[str], domain: str) -> tuple[list[str], list, float, float]:
    stage("MAP-REDUCE", f"Fan-out: {len(sub_questions)} sub-questions × {N_CANDIDATES} candidates each")
    info("Total LLM calls:", str(len(sub_questions) * N_CANDIDATES))

    domain_system_path = PROMPTS / f"domains/{domain}.txt"
    if not domain_system_path.exists():
        domain_system_path = PROMPTS / "domains/general.txt"
    domain_system = domain_system_path.read_text()

    t_start = time.time()
    results = await asyncio.gather(
        *[best_of_n(sq, domain_system, i) for i, sq in enumerate(sub_questions)]
    )
    parallel_time = time.time() - t_start

    best_answers = [r[0] for r in results]
    all_scores   = [r[1] for r in results]
    estimated_sequential = parallel_time * len(sub_questions)

    info("Parallel wall-clock time:", f"{parallel_time:.1f}s")
    info("Estimated sequential time:", f"{estimated_sequential:.1f}s")
    info("Speedup factor:", f"~{estimated_sequential / parallel_time:.1f}x")

    for i, (sq, ans, scores) in enumerate(zip(sub_questions, best_answers, all_scores), 1):
        divider()
        sub(f"Sub-Q {i}: {sq[:60]}")
        sub(f"Best answer: {ans[:100]}...")
        if scores:
            for s in scores:
                sub(f"  Candidate {s.get('candidate_index')}: total={s.get('total', '?')}")

    return best_answers, all_scores, parallel_time, estimated_sequential


async def synthesize(question: str, domain: str, sub_questions: list[str], best_answers: list[str]) -> str:
    stage("MAP-REDUCE", "Synthesizing research brief")

    sub_answers_text = "\n\n".join(
        f"Sub-question {i+1}: {sq}\nAnswer: {ans}"
        for i, (sq, ans) in enumerate(zip(sub_questions, best_answers))
    )

    template = _load("parallel/synthesize.txt")
    prompt = template.format(
        question=question,
        domain=domain,
        sub_answers=sub_answers_text,
    )

    brief = await call_producer(prompt)
    sub(f"Brief length: {len(brief.split())} words")
    return brief


async def run_map_reduce(question: str, domain: str) -> tuple[str, dict]:
    sub_questions = await decompose(question)

    best_answers, all_scores, par_time, seq_time = await fan_out(sub_questions, domain)

    brief = await synthesize(question, domain, sub_questions, best_answers)

    metadata = {
        "sub_questions": sub_questions,
        "best_answers": best_answers,
        "all_candidate_scores": all_scores,
        "parallel_time_s": round(par_time, 2),
        "estimated_sequential_time_s": round(seq_time, 2),
        "speedup": round(seq_time / par_time, 2) if par_time > 0 else 0,
    }

    return brief, metadata