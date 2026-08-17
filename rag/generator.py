"""
Generation component of the RAG pipeline.

Two backends are provided:

1. ExtractiveGenerator: synthesizes an answer directly from the retrieved
   chunks with no external API call. This is what actually runs in this
   project, since this environment has no live LLM API access. It's a
   legitimate (if simple) generation strategy on its own, and it lets the
   whole pipeline run end-to-end, offline, deterministically.

2. ClaudeGenerator: shows exactly how this pipeline would call a real LLM
   (Claude) to generate a grounded answer from the same retrieved context.
   This is NOT executed by default (no API key wired into this sandbox) --
   it exists to show the intended production architecture. Swapping it in
   for ExtractiveGenerator requires no changes anywhere else in the
   pipeline, since both implement the same generate() interface.
"""

from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        ...


class ExtractiveGenerator(Generator):
    """Builds an answer directly from the retrieved text, with citations.
    No external model call -- runs fully offline."""

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        if not retrieved_chunks:
            return "No relevant information found in the knowledge base for this query."

        lines = [f"Based on {len(retrieved_chunks)} relevant source(s):\n"]
        for i, chunk in enumerate(retrieved_chunks, start=1):
            # Take the most relevant sentence(s) from the chunk rather than
            # dumping the whole thing, so the answer stays focused on the query.
            sentences = [s.strip() for s in chunk["text"].split(".") if s.strip()]
            best_sentences = _rank_sentences_by_query_overlap(sentences, query)[:2]
            snippet = ". ".join(best_sentences).strip()
            if snippet and not snippet.endswith("."):
                snippet += "."
            lines.append(f"[{i}] ({chunk['doc_id']}, relevance={chunk['score']:.2f}) {snippet}")
        return "\n".join(lines)


def _rank_sentences_by_query_overlap(sentences: list[str], query: str) -> list[str]:
    query_words = set(w.lower() for w in query.split())
    scored = []
    for s in sentences:
        s_words = set(w.lower().strip(",.") for w in s.split())
        overlap = len(query_words & s_words)
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


class ClaudeGenerator(Generator):
    """
    Production-shaped implementation showing how this pipeline would call
    Claude to generate a grounded answer. Not executed in this project --
    included to document the intended architecture.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key
        self.model = model

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source: {c['doc_id']}]\n{c['text']}" for c in retrieved_chunks
        )
        prompt = (
            "Answer the question using only the provided sources. "
            "Cite sources by name. If the sources don't contain the answer, say so.\n\n"
            f"Sources:\n{context}\n\nQuestion: {query}"
        )
        # In production this would be:
        #
        #   import anthropic
        #   client = anthropic.Anthropic(api_key=self.api_key)
        #   response = client.messages.create(
        #       model=self.model,
        #       max_tokens=500,
        #       messages=[{"role": "user", "content": prompt}],
        #   )
        #   return response.content[0].text
        #
        # Not called here since this sandbox has no API key wired in.
        raise NotImplementedError(
            "ClaudeGenerator is a documented reference implementation, "
            "not wired up to a live API key in this environment."
        )
