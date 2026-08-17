"""Minimal Flask API exposing the RAG pipeline as a queryable HTTP endpoint."""

import os
from flask import Flask, request, jsonify

from rag.pipeline import RAGPipeline

app = Flask(__name__)
corpus_path = os.path.join(os.path.dirname(__file__), "corpus")
pipeline = RAGPipeline(corpus_path)


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    result = pipeline.query(question)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "documents_indexed": len(set(c.doc_id for c in pipeline.retriever.chunks))})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
