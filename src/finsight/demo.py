"""A zero-dependency demo. Run: python -m finsight.demo

Loads the sample filings, answers a few questions through the full
router -> retriever -> analyst -> verifier pipeline, and prints a short
evaluation summary. Works with no API keys and no network.
"""
from __future__ import annotations

from .config import settings
from .pipeline import answer, load_sample_corpus
from .eval.dataset import GOLD
from .eval.metrics import answer_relevance, context_precision, mean


def main() -> None:
    store = load_sample_corpus()
    model = settings.default_models()[0]
    print(f"Loaded sample corpus: {len(store)} chunks  |  mode={settings.mode}  |  model={model}\n")

    # Answer every gold question once, then reuse for display + aggregate.
    results = [(item, answer(item.question, store, model=model)) for item in GOLD]

    for item, res in results[:3]:
        print(f"Q: {item.question}")
        print(f"  route      : {res.route['kind']} (top_k={res.route['top_k']})")
        print(f"  answer     : {res.answer}")
        print(f"  citations  : {res.citations}")
        print(f"  grounded   : {res.grounded_ratio:.2f}")
        print(f"  ctx prec.  : {context_precision(res.citations, item.gold_docs):.2f}")
        print(f"  relevance  : {answer_relevance(res.answer, item.reference):.2f}\n")

    print("Aggregate over %d questions:" % len(results))
    print("  mean groundedness     : %.2f" % mean([r.grounded_ratio for _, r in results]))
    print("  mean context precision: %.2f"
          % mean([context_precision(r.citations, i.gold_docs) for i, r in results]))
    print("  mean answer relevance : %.2f"
          % mean([answer_relevance(r.answer, i.reference) for i, r in results]))


if __name__ == "__main__":
    main()
