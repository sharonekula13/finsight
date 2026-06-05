"""End-to-end orchestration and the corpus loader."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import agents
from .chunking import chunk_document
from .config import settings
from .llm import get_llm
from .retrieval import RetrievedContext
from .vectorstore import InMemoryStore, build_store

_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample")
_FILINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "filings")


def _has_txt(path: str) -> bool:
    return os.path.isdir(path) and any(f.endswith(".txt") for f in os.listdir(path))


def _corpus_dir() -> str:
    """Pick the corpus: an explicit env path, else real filings if any have
    been fetched, else the bundled fictional sample."""
    env = os.environ.get("FINSIGHT_CORPUS_DIR", "").strip()
    if env:
        return env
    filings = os.path.normpath(_FILINGS_DIR)
    if _has_txt(filings):
        return filings
    return os.path.normpath(_SAMPLE_DIR)


@dataclass
class AnswerResult:
    query: str
    answer: str
    route: dict
    citations: list[str]
    grounded_ratio: float
    unsupported: list[str]
    model: str
    latency_s: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    contexts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query, "answer": self.answer, "route": self.route,
            "citations": self.citations, "grounded_ratio": round(self.grounded_ratio, 3),
            "unsupported": self.unsupported, "model": self.model,
            "latency_s": round(self.latency_s, 4), "cost_usd": round(self.cost_usd, 6),
            "tokens": self.tokens,
        }


def load_sample_corpus(store: InMemoryStore | None = None) -> InMemoryStore:
    store = store or build_store()
    sample_dir = _corpus_dir()
    for fname in sorted(os.listdir(sample_dir)):
        if not fname.endswith(".txt"):
            continue
        doc_id = os.path.splitext(fname)[0]
        with open(os.path.join(sample_dir, fname), encoding="utf-8") as fh:
            text = fh.read()
        chunks = chunk_document(doc_id, text, settings.chunk_size, settings.chunk_overlap)
        if settings.max_chunks > 0:
            chunks = chunks[:settings.max_chunks]
        store.add(chunks)
    return store


def answer(query: str, store: InMemoryStore, model: str = "mock-a") -> AnswerResult:
    llm = get_llm(model)
    plan = agents.route(query)
    context: RetrievedContext = agents.retrieve(store, query, plan["top_k"])
    completion = agents.analyze(llm, query, context)
    draft = completion.text
    check = agents.verify(draft, context)
    return AnswerResult(
        query=query, answer=draft, route=plan, citations=context.ids(),
        grounded_ratio=check.grounded_ratio, unsupported=check.unsupported,
        model=model, latency_s=completion.latency_s, cost_usd=completion.cost_usd,
        tokens=completion.tokens, contexts=[c.chunk.text for c in context.chunks],
    )
