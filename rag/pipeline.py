"""
End-to-end RAG pipeline: retrieve relevant chunks, then generate a grounded
answer from them.
"""

import os

from rag.retriever import Retriever
from rag.generator import ExtractiveGenerator


class RAGPipeline:
    def __init__(self, corpus_dir: str, generator=None, top_k: int = 3):
        self.retriever = Retriever(corpus_dir)
        self.generator = generator or ExtractiveGenerator()
        self.top_k = top_k

    def query(self, question: str) -> dict:
        retrieved = self.retriever.retrieve(question, top_k=self.top_k)
        answer = self.generator.generate(question, retrieved)
        return {
            "question": question,
            "answer": answer,
            "sources": [{"doc_id": r["doc_id"], "score": round(r["score"], 3)} for r in retrieved],
        }


if __name__ == "__main__":
    corpus_path = os.path.join(os.path.dirname(__file__), "corpus")
    pipeline = RAGPipeline(corpus_path)

    test_questions = [
        "How do teams measure whether a shot was a good look regardless of whether it went in?",
        "What is drop coverage in pick-and-roll defense?",
        "How is player load monitored using tracking data?",
        "What is the offside rule in soccer?",  # deliberately out-of-domain
    ]

    for q in test_questions:
        result = pipeline.query(q)
        print("=" * 80)
        print(f"Q: {result['question']}")
        print(f"Sources: {result['sources']}")
        print(f"A: {result['answer']}\n")
