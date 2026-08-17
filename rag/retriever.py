"""
Retrieval component of a basketball-analytics RAG pipeline.

Uses TF-IDF + cosine similarity for retrieval. This is a deliberate choice for
this environment: it requires no downloaded model weights (unlike a
sentence-transformer embedding model), so it runs fully offline and
reproducibly. In a production setting with model access, this retriever
could be swapped for a dense embedding model without changing the interface
below (retrieve() still returns ranked chunks).
"""

import os
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    doc_id: str
    text: str


def _chunk_document(doc_id: str, text: str, chunk_size: int = 350, overlap: int = 60):
    """Split a document into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(Chunk(doc_id=doc_id, text=" ".join(chunk_words)))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


class Retriever:
    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir
        self.chunks: list[Chunk] = []
        self._load_corpus()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def _load_corpus(self):
        for fname in sorted(os.listdir(self.corpus_dir)):
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(self.corpus_dir, fname)
            with open(path, "r") as f:
                raw = f.read()
            raw = re.sub(r"\s+", " ", raw).strip()
            doc_id = fname.replace(".txt", "")
            self.chunks.extend(_chunk_document(doc_id, raw))

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.08):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [
            {"doc_id": self.chunks[i].doc_id, "text": self.chunks[i].text, "score": float(sims[i])}
            for i in ranked_idx
            if sims[i] > min_score
        ]
