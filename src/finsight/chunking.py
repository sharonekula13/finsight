"""Token-ish chunking of filing text into overlapping windows.

We chunk on whitespace tokens (a dependency-free stand-in for a real
tokenizer). Each chunk keeps a reference to its source document id and its
position so retrieved context can be cited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    section: str = ""

    def citation(self) -> str:
        return f"[{self.chunk_id}]"


_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def chunk_document(doc_id: str, text: str, size: int = 120, overlap: int = 30,
                   section: str = "") -> list[Chunk]:
    """Split ``text`` into overlapping word-windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = _normalize(text).split(" ")
    if not words or words == [""]:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    idx = 0
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{idx}",
                text=" ".join(window),
                section=section,
            )
        )
        idx += 1
        if start + size >= len(words):
            break
    return chunks
