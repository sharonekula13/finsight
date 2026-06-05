"""The agent graph: router -> retriever -> analyst -> verifier.

It is implemented as plain, testable Python so it runs anywhere. In live mode
you can swap this for a LangGraph state machine without changing the public
``answer`` contract (see pipeline.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieval import RetrievedContext, hybrid_retrieve
from .vectorstore import InMemoryStore

_SENT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")

ANALYST_SYSTEM = (
    "You are a financial analyst. Answer ONLY from the provided context and "
    "cite the bracketed chunk id after each claim. If the context is "
    "insufficient, say so."
)


# ---------------- router ----------------
def route(query: str) -> dict:
    q = query.lower()
    if any(w in q for w in ("compare", "versus", " vs ", "difference between")):
        kind, top_k = "comparison", 6
    elif any(w in q for w in ("trend", "over the", "year over year", "growth")):
        kind, top_k = "trend", 5
    else:
        kind, top_k = "lookup", 4
    return {"kind": kind, "top_k": top_k}


# ---------------- retriever ----------------
def retrieve(store: InMemoryStore, query: str, top_k: int) -> RetrievedContext:
    return hybrid_retrieve(store, query, top_k=top_k)


# ---------------- analyst ----------------
def analyze(llm, query: str, context: RetrievedContext):
    """Returns the provider Completion (carries text, tokens, cost, latency)."""
    user = f"QUESTION: {query}\n\nCONTEXT:\n{context.as_prompt_block()}"
    return llm.complete(ANALYST_SYSTEM, user)


# ---------------- verifier ----------------
@dataclass
class Verification:
    grounded_ratio: float
    unsupported: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.grounded_ratio >= 0.5


def _supported(sentence: str, context_text: str, threshold: float = 0.5) -> bool:
    s_tokens = [w for w in _TOKEN.findall(sentence.lower()) if len(w) > 2]
    if not s_tokens:
        return True
    ctx = set(_TOKEN.findall(context_text.lower()))
    hits = sum(1 for w in s_tokens if w in ctx)
    return (hits / len(s_tokens)) >= threshold


def verify(answer: str, context: RetrievedContext) -> Verification:
    """Check each answer sentence against retrieved context (lexical
    entailment proxy). In live mode an LLM verifier can replace this."""
    ctx_text = " ".join(c.chunk.text for c in context.chunks)
    # strip citation markers before checking
    clean = re.sub(r"\[[^\]]+\]", "", answer)
    sentences = [s.strip() for s in _SENT.split(clean) if s.strip()]
    if not sentences:
        return Verification(grounded_ratio=0.0)
    supported = [s for s in sentences if _supported(s, ctx_text)]
    unsupported = [s for s in sentences if s not in supported]
    return Verification(grounded_ratio=len(supported) / len(sentences),
                        unsupported=unsupported)
