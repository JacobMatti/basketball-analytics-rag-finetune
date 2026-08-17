"""
Basic tests covering the RAG retrieval logic and the tokenizer, run without
needing the full pretraining/fine-tuning cycle (which is slow-ish and
covered separately by pretrain.py / finetune.py's own printed output).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from rag.retriever import Retriever, _chunk_document
from rag.generator import ExtractiveGenerator
from finetune.tokenizer import SimpleTokenizer, PAD, CLS


def test_chunking_covers_full_document():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = _chunk_document("doc", text, chunk_size=350, overlap=60)
    covered = set()
    for c in chunks:
        covered.update(c.text.split())
    all_words = set(text.split())
    assert all_words.issubset(covered), "Chunking dropped words from the document"
    print("test_chunking_covers_full_document: PASS")


def test_retriever_returns_relevant_doc():
    corpus_dir = os.path.join(os.path.dirname(__file__), "rag", "corpus")
    retriever = Retriever(corpus_dir)
    results = retriever.retrieve("pick and roll drop coverage defense", top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "pick_and_roll", f"Expected pick_and_roll, got {results[0]['doc_id']}"
    print("test_retriever_returns_relevant_doc: PASS")


def test_retriever_returns_nothing_for_unrelated_query():
    corpus_dir = os.path.join(os.path.dirname(__file__), "rag", "corpus")
    retriever = Retriever(corpus_dir)
    results = retriever.retrieve("chocolate chip cookie recipe flour sugar butter eggs", top_k=3)
    assert len(results) == 0, f"Expected no results for unrelated query, got {results}"
    print("test_retriever_returns_nothing_for_unrelated_query: PASS")


def test_extractive_generator_handles_empty_retrieval():
    gen = ExtractiveGenerator()
    answer = gen.generate("anything", [])
    assert "No relevant information" in answer
    print("test_extractive_generator_handles_empty_retrieval: PASS")


def test_tokenizer_roundtrip_length():
    tok = SimpleTokenizer.build(["Marcus Reeves makes three-pointer from 22 feet."])
    ids, attn = tok.encode("Marcus Reeves makes three-pointer from 22 feet.", max_len=16)
    assert len(ids) == 16
    assert len(attn) == 16
    assert ids[0] == tok.vocab[CLS]
    assert sum(attn) <= 16
    print("test_tokenizer_roundtrip_length: PASS")


def test_tokenizer_handles_unknown_word():
    tok = SimpleTokenizer.build(["hello world"])
    ids, attn = tok.encode("a completely different sentence entirely", max_len=8)
    assert all(i < len(tok) for i in ids)
    print("test_tokenizer_handles_unknown_word: PASS")


if __name__ == "__main__":
    test_chunking_covers_full_document()
    test_retriever_returns_relevant_doc()
    test_retriever_returns_nothing_for_unrelated_query()
    test_extractive_generator_handles_empty_retrieval()
    test_tokenizer_roundtrip_length()
    test_tokenizer_handles_unknown_word()
    print("\nAll tests passed.")
