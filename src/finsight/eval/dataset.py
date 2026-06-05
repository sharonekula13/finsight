"""Gold evaluation set over the sample corpus.

Each item has the question, the set of documents that *should* be retrieved
(for context precision), and a short reference answer (for answer relevance).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalItem:
    qid: str
    question: str
    gold_docs: tuple[str, ...]
    reference: str
    kind: str = "lookup"


GOLD: list[EvalItem] = [
    EvalItem("q1", "What was NovaTech's total revenue in fiscal 2024?",
             ("novatech_10k",),
             "NovaTech reported total revenue of 4.2 billion dollars in fiscal 2024."),
    EvalItem("q2", "Which is NovaTech's largest business segment?",
             ("novatech_10k",),
             "The Cloud Platform segment was the largest, at 62 percent of revenue."),
    EvalItem("q3", "What customer concentration risk does NovaTech disclose?",
             ("novatech_10k",),
             "A single customer accounted for 22 percent of total revenue in fiscal 2024."),
    EvalItem("q4", "How did Helios Energy's revenue change in fiscal 2024?",
             ("helios_10k",),
             "Helios Energy revenue decreased 5 percent to 2.8 billion dollars in fiscal 2024."),
    EvalItem("q5", "What is the principal risk facing Helios Energy?",
             ("helios_10k",),
             "The principal risk is commodity price volatility in electricity and gas prices.",
             kind="lookup"),
    EvalItem("q6", "Compare NovaTech and Helios revenue in fiscal 2024.",
             ("novatech_10k", "helios_10k"),
             "NovaTech revenue was 4.2 billion dollars; Helios revenue was 2.8 billion dollars.",
             kind="comparison"),
]
