"""LLM adapters with a single ``complete`` interface.

Offline mode ships two deterministic "mock" models that answer
*extractively* from the supplied context (so answers are inherently
grounded and the pipeline/eval are meaningful with no keys). Live mode wires
Anthropic, OpenAI, and Gemini behind lazy imports.

Every call returns a ``Completion`` carrying the text plus a rough token/cost
estimate so the evaluation harness can report latency and cost.
"""
from __future__ import annotations

import re
import os
import time
from dataclasses import dataclass

from .config import settings

_SENT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_CITED = re.compile(r"\[([^\]]+)\]\s*(.*)")

# Rough $ per 1K tokens (input+output blended) for cost estimates.
_PRICE = {"claude": 0.006, "gpt": 0.004, "gemini": 0.001, "mock-a": 0.0, "mock-b": 0.0}


@dataclass
class Completion:
    text: str
    model: str
    tokens: int
    cost_usd: float
    latency_s: float


def _approx_tokens(*texts: str) -> int:
    return sum(len(_TOKEN.findall(t)) for t in texts)


def _split_context(user: str) -> list[tuple[str, str]]:
    """Parse a CONTEXT block of '[id] text' lines into (id, sentence) pairs."""
    if "CONTEXT:" not in user:
        return []
    block = user.split("CONTEXT:", 1)[1]
    pairs: list[tuple[str, str]] = []
    for line in block.split("\n\n"):
        m = _CITED.match(line.strip())
        if not m:
            continue
        cid, text = m.group(1), m.group(2)
        for sent in _SENT.split(text):
            sent = sent.strip()
            if sent:
                pairs.append((cid, sent))
    return pairs


def _question(user: str) -> str:
    if "QUESTION:" not in user:
        return user
    return user.split("QUESTION:", 1)[1].split("CONTEXT:", 1)[0].strip()


class MockLLM:
    """Deterministic extractive answerer. ``variant`` tweaks behavior so the
    multi-model evaluation surfaces real differences between models."""

    def __init__(self, name: str = "mock-a", variant: int = 0) -> None:
        self.name = name
        self.variant = variant

    def complete(self, system: str, user: str) -> Completion:
        t0 = time.perf_counter()
        q_tokens = set(_TOKEN.findall(_question(user).lower()))
        pairs = _split_context(user)

        scored = []
        for cid, sent in pairs:
            toks = _TOKEN.findall(sent.lower())
            overlap = sum(1 for w in toks if w in q_tokens)
            if overlap:
                scored.append((overlap, cid, sent))
        scored.sort(key=lambda x: x[0], reverse=True)

        keep = 2 if self.variant == 0 else 3  # variant b is more verbose
        picked = scored[:keep]
        if picked:
            answer = " ".join(f"{sent} {('[' + cid + ']')}" for _, cid, sent in picked)
        else:
            answer = "The provided filings do not contain enough information to answer that."
        latency = time.perf_counter() - t0
        tok = _approx_tokens(system, user, answer)
        return Completion(answer, self.name, tok, _PRICE.get(self.name, 0.0) * tok / 1000, latency)


class ProviderLLM:  # pragma: no cover - exercised only in live mode
    def __init__(self, name: str) -> None:
        self.name = name

    def complete(self, system: str, user: str) -> Completion:
        t0 = time.perf_counter()
        if self.name == "claude":
            from anthropic import Anthropic  # type: ignore

            client = Anthropic(api_key=settings.anthropic_api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=600,
                system=system, messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        elif self.name == "gpt":
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=settings.openai_api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            text = resp.choices[0].message.content or ""
        elif self.name == "gemini":
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=settings.gemini_api_key)
            text = _gemini_generate(genai, system, user)
        else:
            raise ValueError(f"unknown provider {self.name}")
        latency = time.perf_counter() - t0
        tok = _approx_tokens(system, user, text)
        return Completion(text, self.name, tok, _PRICE.get(self.name, 0.0) * tok / 1000, latency)


_GEMINI_CHAT_MODEL = None

# Current chat models, ordered by free-tier generosity. Each has its OWN daily
# quota, so trying the next one when one is rate-limited extends free capacity.
_GEMINI_CHAT_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


def _is_rate_limit(err: Exception) -> bool:
    m = str(err).lower()
    return "429" in m or "quota" in m or "rate limit" in m


def _gemini_generate(genai, system: str, user: str) -> str:  # pragma: no cover - live
    """Generate with a working Gemini chat model. Falls over to another model
    on a 404 (missing) or 429 (that model's quota is used up)."""
    global _GEMINI_CHAT_MODEL
    override = os.environ.get("FINSIGHT_GEMINI_MODEL", "").strip()
    base = ([override] if override else []) + _GEMINI_CHAT_CANDIDATES
    # try the previously-working model first, then the rest
    order = ([_GEMINI_CHAT_MODEL] if _GEMINI_CHAT_MODEL else []) + \
            [m for m in base if m != _GEMINI_CHAT_MODEL]

    rate_limited, last_err = [], None
    for name in order:
        try:
            model = genai.GenerativeModel(name, system_instruction=system)
            text = model.generate_content(user).text
            _GEMINI_CHAT_MODEL = name
            return text
        except Exception as err:
            last_err = err
            if _is_rate_limit(err):
                rate_limited.append(name)
                if _GEMINI_CHAT_MODEL == name:
                    _GEMINI_CHAT_MODEL = None  # don't stick to a throttled model
            # 404 or 429 -> try the next candidate

    if rate_limited:
        raise RuntimeError(
            "Gemini free-tier daily quota reached on: " + ", ".join(rate_limited) +
            ". Free models have small daily caps. Options: wait a bit and re-run, "
            "pick a model with FINSIGHT_GEMINI_MODEL (see `python -m finsight.list_models`), "
            "or run offline with $env:FINSIGHT_MODE=\"offline\". Last error: " + str(last_err)
        )
    raise RuntimeError(
        "No working Gemini chat model found. Run `python -m finsight.list_models` "
        f"and set FINSIGHT_GEMINI_MODEL. Last error: {last_err}"
    )


def get_llm(name: str):
    if not settings.is_live or name.startswith("mock"):
        return MockLLM(name=name, variant=0 if name.endswith("a") else 1)
    return ProviderLLM(name=name)
