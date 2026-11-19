"""Chunking and BM25 retrieval over the policy documents.

BM25 keeps the demo offline, deterministic and dependency-free, and it is a strong baseline for
short policy text full of exact terms (STOP, E.164, 422). The retriever is behind one method,
``search``, so an embedding index or a hybrid (BM25 + vectors, then a reranker) can replace it
without touching the bot or the evals.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

_WORD = re.compile(r"[a-z0-9_.+]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has",
    "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "per", "the", "their",
    "there", "to", "what", "when", "where", "which", "who", "with", "you", "your", "get", "this",
}  # fmt: skip


def tokens(text: str) -> list[str]:
    return [w.strip(".") for w in _WORD.findall(text.casefold()) if w.strip(".") not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    id: str  # <doc id>#<paragraph number>, the id answers cite
    doc_id: str
    title: str
    text: str


def chunk_docs(docs: list[dict[str, Any]]) -> list[Chunk]:
    """One chunk per paragraph. Policy paragraphs are short and self-contained, so a paragraph
    is the unit a citation should point to."""
    chunks = []
    for doc in docs:
        paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs, start=1):
            chunks.append(
                Chunk(f"{doc['id']}#{i}", doc["id"], doc["title"], " ".join(para.split()))
            )
    return chunks


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        # The title is indexed with each chunk: "Transportation benefit" helps a ride question.
        self.docs = [Counter(tokens(f"{c.title} {c.text}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        terms = tokens(query)
        scored = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths, strict=True):
            score = 0.0
            for t in terms:
                f = doc.get(t, 0)
                if f:
                    norm = f + self.k1 * (1 - self.b + self.b * length / self.avg)
                    score += self.idf[t] * f * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, round(score, 4)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]
