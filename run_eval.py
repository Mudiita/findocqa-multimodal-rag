"""Run every query in eval_queries.csv once through FinDocQA and save eval_results.csv.

Usage: venv310/bin/python run_eval.py [QUERIES.csv RESULTS.csv]
       venv310/bin/python run_eval.py --summary [RESULTS.csv]

Answers are not graded here; the last three columns are left empty for manual review.
"""

import asyncio
import csv
import re
import statistics
import sys
import time
from collections import Counter

import main

QUERIES = main.ROOT / "eval_queries.csv"
RESULTS = main.ROOT / "eval_results.csv"
FIELDS = [
    "id", "category", "question", "expected_answer", "expected_source", "answer",
    "cited_documents_pages", "number_check", "unverified_numbers", "response_time_s", "cost_usd",
    "correct (Y/N/partial)", "citation correct (Y/N)", "my notes",
]

NUMBER = re.compile(r"(?:rs\.?|₹)\s*[\d,]+|\d+(?:\.\d+)?\s*%|\b\d[\d,]*(?:\.\d+)?\b", re.I)
# Phrases that signal the answer declined, hedged or asked for clarification.
HEDGES = [
    "not contain", "not available", "do not have", "does not have", "don't have", "cannot", "can't", "not provided",
    "no information", "not mentioned", "not specified", "not found", "not include", "unable", "please specify",
    "please clarify", "could you", "which bank", "which fund", "which policy", "depends on", "do not provide",
    "does not provide", "not possible", "not in the",
]


def confident_number(answer: str) -> bool:
    text = answer.lower()
    return bool(NUMBER.search(answer)) and not any(h in text for h in HEDGES)


async def run(queries_path=QUERIES, results_path=RESULTS):
    with open(queries_path, newline="") as f:
        queries = list(csv.DictReader(f))
    rag = main.build_rag()
    rows = []
    for n, q in enumerate(queries, 1):
        cost_before = main.session_cost()
        started = time.perf_counter()
        try:
            result = await main.query(rag, q["query"])
            answer = result.answer
            check, unverified = result.verification, ", ".join(result.unverified_numbers)
            cited = "; ".join(
                f"{c.document} p.{c.page}" + ("" if c.verified else " (page unverified)") for c in result.citations
            )
        except Exception as exc:  # noqa: BLE001 - record and keep going
            answer, cited, check, unverified = f"ERROR: {type(exc).__name__}: {exc}", "", "error", ""
        elapsed = time.perf_counter() - started
        cost = main.session_cost() - cost_before
        source = q["source_document"] + (f" p.{q['source_page']}" if q.get("source_page") else "")
        rows.append({
            "id": q["query_id"], "category": q["category"], "question": q["query"],
            "expected_answer": q["expected_answer"], "expected_source": source, "answer": answer,
            "cited_documents_pages": cited, "number_check": check, "unverified_numbers": unverified,
            "response_time_s": f"{elapsed:.2f}", "cost_usd": f"{cost:.5f}",
            "correct (Y/N/partial)": "", "citation correct (Y/N)": "", "my notes": "",
        })
        # Save after every query so an interruption keeps the finished rows.
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{n:2d}/{len(queries)}] {q['query_id']:4s} {q['category']:20s} {elapsed:6.2f}s ${cost:.5f}", flush=True)
    await rag.finalize_storages()
    summarize(rows, results_path)


def summarize(rows, results_path=RESULTS):
    times = [float(r["response_time_s"]) for r in rows]
    costs = [float(r["cost_usd"]) for r in rows]
    errors = [r["id"] for r in rows if r["answer"].startswith("ERROR")]
    print("\n" + "=" * 80)
    print(f"Queries run:            {len(rows)}  (errors: {len(errors)}{' ' + str(errors) if errors else ''})")
    print(f"Response time:          average {statistics.mean(times):.2f}s, median {statistics.median(times):.2f}s, "
          f"min {min(times):.2f}s, max {max(times):.2f}s")
    print(f"Total cost:             ${sum(costs):.4f}  (average ${statistics.mean(costs):.5f} per query)")
    print("Queries per category:")
    for cat, count in Counter(r["category"] for r in rows).items():
        cat_times = [float(r["response_time_s"]) for r in rows if r["category"] == cat]
        print(f"  {cat:22s} {count:3d}   avg {statistics.mean(cat_times):.2f}s")
    print("\nout_of_scope / ambiguous answers containing a number without declining or asking for clarification")
    print("(heuristic flag for hallucination review - not a grade):")
    flagged = 0
    for r in rows:
        if r["category"] not in ("out_of_scope", "ambiguous"):
            continue
        if confident_number(r["answer"]):
            flagged += 1
            print(f"\n  {r['id']} [{r['category']}] {r['question']}\n     A: {r['answer'][:400]}"
                  f"\n     cited: {r['cited_documents_pages'] or '-'}")
    print(f"\n{flagged} of {sum(r['category'] in ('out_of_scope', 'ambiguous') for r in rows)} flagged.")
    checks = Counter(r.get("number_check", "n/a") for r in rows)
    print(f"\nNumber guard: {dict(checks)}")
    for r in rows:
        if r.get("number_check") in ("flagged", "withheld"):
            print(f"  {r['id']} {r['number_check']}: {r.get('unverified_numbers', '')}")
    print(f"\nResults saved to {results_path.name}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        path = main.ROOT / sys.argv[2] if len(sys.argv) > 2 else RESULTS
        with open(path, newline="") as f:
            summarize(list(csv.DictReader(f)), path)
    elif len(sys.argv) == 3:
        asyncio.run(run(main.ROOT / sys.argv[1], main.ROOT / sys.argv[2]))
    else:
        asyncio.run(run())
