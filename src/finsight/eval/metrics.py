"""Lightweight, dependency-free evaluation metrics.

These are transparent proxies designed to run in CI with no external calls:

* context_precision -- fraction of retrieved chunks from a gold document.
* groundedness       -- fraction of answer sentences supported by context
                        (produced by the verifier agent).
* answer_relevance   -- token-level F1 between the answer and a reference.

In live mode you can swap in RAGAS / model-graded metrics; the harness shape
(per-question rows -> per-model aggregates) stays the same.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")


def _toks(text: str) -> list[str]:
    return _TOKEN.findall(re.sub(r"\[[^\]]+\]", "", text).lower())


def context_precision(retrieved_chunk_ids: list[str], gold_docs: tuple[str, ...]) -> float:
    if not retrieved_chunk_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_chunk_ids if cid.split("#", 1)[0] in gold_docs)
    return hits / len(retrieved_chunk_ids)


def answer_relevance(answer: str, reference: str) -> float:
    a, r = set(_toks(answer)), set(_toks(reference))
    if not a or not r:
        return 0.0
    inter = len(a & r)
    if inter == 0:
        return 0.0
    precision = inter / len(a)
    recall = inter / len(r)
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
