"""Configuration and runtime mode.

FinSight runs in two modes:

* "offline" (default) -- no API keys, no network. Uses deterministic mock
  embeddings and a deterministic extractive "LLM". Lets you run the whole
  pipeline + evaluation locally and in CI with zero cost.
* "live" -- set the relevant API keys and FINSIGHT_MODE=live to use real
  embedding models, a real vector store, and real LLM providers.

The mode is resolved once here so the rest of the code stays simple.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    mode: str = field(default_factory=lambda: _env("FINSIGHT_MODE", "offline").lower())

    # Provider keys (only needed in live mode).
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))

    # Retrieval knobs.
    chunk_size: int = field(default_factory=lambda: int(_env("FINSIGHT_CHUNK_SIZE", "120")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("FINSIGHT_CHUNK_OVERLAP", "30")))
    top_k: int = field(default_factory=lambda: int(_env("FINSIGHT_TOP_K", "4")))
    dense_weight: float = field(default_factory=lambda: float(_env("FINSIGHT_DENSE_WEIGHT", "0.6")))
    max_chunks: int = field(default_factory=lambda: int(_env("FINSIGHT_MAX_CHUNKS", "0")))

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def default_models(self) -> list[str]:
        """Models used by the evaluation harness. In live mode, only the
        providers whose API keys are present are included."""
        if self.is_live:
            models: list[str] = []
            if self.anthropic_api_key:
                models.append("claude")
            if self.openai_api_key:
                models.append("gpt")
            if self.gemini_api_key:
                models.append("gemini")
            return models or ["gemini"]
        return ["mock-a", "mock-b"]


settings = Settings()
