"""Embeddings.

Offline mode uses a deterministic hashing embedding (a "feature hashing"
bag-of-words vector). It is not semantically rich, but it is stable, fast,
dependency-free, and good enough to exercise and evaluate the full pipeline.

Live mode uses sentence-transformers (local) or OpenAI embeddings if a key
is set. Those imports are lazy so offline mode needs nothing installed.
"""
from __future__ import annotations

import hashlib
import math
import os
import re

from .config import settings

_DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _hash_embed(text: str, dim: int = _DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        bucket = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


_GEMINI_EMBED_MODEL: str | None = None


def _pick_gemini_embed_model(genai) -> str:
    """Find a working Gemini embedding model once and cache it. Model names
    change over time, so we try several known names (newest first)."""
    global _GEMINI_EMBED_MODEL
    if _GEMINI_EMBED_MODEL:
        return _GEMINI_EMBED_MODEL
    override = os.environ.get("FINSIGHT_GEMINI_EMBED_MODEL", "").strip()
    candidates = ([override] if override else []) + [
        "models/gemini-embedding-001",
        "models/text-embedding-004",
        "models/embedding-001",
    ]
    last_err = None
    for name in candidates:
        try:
            genai.embed_content(model=name, content="ping")
            _GEMINI_EMBED_MODEL = name
            return name
        except Exception as err:  # try the next candidate
            last_err = err
    raise RuntimeError(
        "No working Gemini embedding model found. Run "
        "`python -m finsight.list_models` to see what your key supports, then "
        f"set FINSIGHT_GEMINI_EMBED_MODEL. Last error: {last_err}"
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not settings.is_live:
        return [_hash_embed(t) for t in texts]
    # ---- live providers (lazy) ----
    if settings.gemini_api_key:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=settings.gemini_api_key)
        model_name = _pick_gemini_embed_model(genai)
        out: list[list[float]] = []
        for t in texts:
            r = genai.embed_content(model=model_name, content=t)
            out.append(r["embedding"])
        return out
    if settings.openai_api_key:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in resp.data]
    # local fallback that still counts as "live"
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return [v.tolist() for v in model.encode(texts)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
