"""Run the evaluation harness across one or more models.

Usage:
    python -m finsight.eval.run_eval                 # all default models
    python -m finsight.eval.run_eval --models mock-a # subset
    python -m finsight.eval.run_eval --gate 0.8      # fail if below threshold

Writes a JSON report and a Markdown leaderboard to ./reports/.
Exit code is non-zero if any model's mean groundedness < --gate, so this
doubles as a CI quality gate.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ..config import settings
from ..pipeline import answer, load_sample_corpus
from .dataset import GOLD
from .metrics import answer_relevance, context_precision, mean


def evaluate_model(model: str, store) -> dict:
    rows = []
    for item in GOLD:
        res = answer(item.question, store, model=model)
        rows.append({
            "qid": item.qid,
            "groundedness": res.grounded_ratio,
            "context_precision": context_precision(res.citations, item.gold_docs),
            "answer_relevance": answer_relevance(res.answer, item.reference),
            "latency_s": res.latency_s,
            "cost_usd": res.cost_usd,
        })
    agg = {
        "model": model,
        "groundedness": mean([r["groundedness"] for r in rows]),
        "context_precision": mean([r["context_precision"] for r in rows]),
        "answer_relevance": mean([r["answer_relevance"] for r in rows]),
        "avg_latency_s": mean([r["latency_s"] for r in rows]),
        "total_cost_usd": sum(r["cost_usd"] for r in rows),
        "n": len(rows),
    }
    return {"aggregate": agg, "rows": rows}


def to_markdown(results: list[dict]) -> str:
    head = ("| Model | Groundedness | Context Precision | Answer Relevance | "
            "Avg Latency (s) | Total Cost ($) |\n"
            "|---|---|---|---|---|---|\n")
    lines = []
    for r in results:
        a = r["aggregate"]
        lines.append(
            f"| {a['model']} | {a['groundedness']:.2f} | {a['context_precision']:.2f} | "
            f"{a['answer_relevance']:.2f} | {a['avg_latency_s']:.4f} | {a['total_cost_usd']:.4f} |"
        )
    return head + "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="FinSight evaluation harness")
    parser.add_argument("--models", nargs="*", default=settings.default_models())
    parser.add_argument("--gate", type=float, default=0.0,
                        help="fail if any model's groundedness < this value")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    store = load_sample_corpus()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = [evaluate_model(m, store) for m in args.models]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval_report.json"), "w") as fh:
        json.dump({"started": started, "mode": settings.mode, "results": results},
                  fh, indent=2)
    md = to_markdown(results)
    with open(os.path.join(args.out, "leaderboard.md"), "w") as fh:
        fh.write(f"# FinSight Evaluation ({settings.mode} mode)\n\n{md}")

    print(f"\nFinSight evaluation — mode={settings.mode}\n")
    print(md)

    if args.gate > 0:
        worst = min(r["aggregate"]["groundedness"] for r in results)
        if worst < args.gate:
            print(f"GATE FAILED: min groundedness {worst:.2f} < {args.gate:.2f}")
            return 1
        print(f"GATE PASSED: min groundedness {worst:.2f} >= {args.gate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
