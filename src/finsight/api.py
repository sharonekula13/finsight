"""FastAPI service. Run: uvicorn finsight.api:app --reload

Endpoints:
    GET  /health
    POST /ask     {"question": "...", "model": "mock-a"}
    GET  /eval    -> latest aggregate metrics per model

FastAPI/uvicorn are only needed to run the server; the core pipeline has no
such dependency.
"""
from __future__ import annotations

from .config import settings
from .pipeline import answer, load_sample_corpus

try:  # keep import optional so core/tests run without web deps
    from fastapi import FastAPI
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore


if FastAPI is not None:
    app = FastAPI(title="FinSight", version="0.1.0")
    _STORE = load_sample_corpus()

    class AskRequest(BaseModel):
        question: str
        model: str = settings.default_models()[0]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "mode": settings.mode, "chunks": len(_STORE),
                "models": settings.default_models()}

    @app.post("/ask")
    def ask(req: "AskRequest") -> dict:
        return answer(req.question, _STORE, model=req.model).to_dict()

    @app.get("/eval")
    def eval_summary() -> dict:
        from .eval.run_eval import evaluate_model
        return {"results": [evaluate_model(m, _STORE)["aggregate"]
                            for m in settings.default_models()]}
