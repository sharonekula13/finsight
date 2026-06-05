"""Hybrid retrieval: blends dense (vector) similarity with a lexical
(keyword-overlap) score, then reranks by the combined score.

Hybrid retrieval consistently beats pure-dense on filings because many
queries hinge on exact terms (line items, fiscal years, defined terms) that
embeddings can blur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings
from .vectorstore import InMemoryStore, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")


def _keyword_score(query: str, text: str) -> float:
    q = set(_TOKEN.findall(query.lower()))
    if not q:
        return 0.0
    d = _TOKEN.findall(text.lower())
    if not d:
        return 0.0
    overlap = sum(1 for w in d if w in q)
    return overlap / len(d) ** 0.5  # length-normalized overlap


@dataclass
class RetrievedContext:
    chunks: list[ScoredChunk]

    def as_prompt_block(self) -> str:
        return "\n\n".join(
            f"{c.chunk.citation()} {c.chunk.text}" for c in self.chunks
        )

    def ids(self) -> list[str]:
        return [c.chunk.chunk_id for c in self.chunks]


def hybrid_retrieve(store: InMemoryStore, query: str, top_k: int | None = None,
                    dense_weight: float | None = None) -> RetrievedContext:
    top_k = top_k or settings.top_k
    dense_weight = settings.dense_weight if dense_weight is None else dense_weight

    # Over-fetch on the dense side, then rerank with the lexical signal.
    dense = store.search(query, top_k=max(top_k * 3, top_k))
    if not dense:
        return RetrievedContext(chunks=[])

    max_dense = max(s.score for s in dense) or 1.0
    reranked: list[ScoredChunk] = []
    for s in dense:
        kw = _keyword_score(query, s.chunk.text)
        combined = dense_weight * (s.score / max_dense) + (1 - dense_weight) * kw
        reranked.append(ScoredChunk(chunk=s.chunk, score=combined))
    reranked.sort(key=lambda s: s.score, reverse=True)
    return RetrievedContext(chunks=reranked[:top_k])
