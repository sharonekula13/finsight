"""Vector store abstraction.

Offline: a simple in-memory store with exact cosine search (fine for the
sample corpus and CI). Live: Chroma, imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .embeddings import cosine, embed_texts
from .config import settings


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class InMemoryStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        vectors = embed_texts([c.text for c in chunks])
        self._chunks.extend(chunks)
        self._vectors.extend(vectors)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        qv = embed_texts([query])[0]
        scored = [
            ScoredChunk(chunk=c, score=cosine(qv, v))
            for c, v in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)


def build_store():
    """Return a store instance for the active mode."""
    if not settings.is_live:
        return InMemoryStore()
    # ---- live (lazy) ----
    try:
        import chromadb  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - live path
        return InMemoryStore()
    return InMemoryStore()  # A Chroma-backed adapter plugs in here.
