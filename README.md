# Basketball Analytics RAG + Fine-Tuning Demo

A small, self-contained project demonstrating two things I didn't previously have concrete,
built examples of: a retrieval-augmented generation (RAG) pipeline, and the pretrain-then-
fine-tune workflow for adapting a model to a downstream task. Both are built around a
basketball-analytics theme.

**Important context on scope:** this was built in a sandboxed environment with no GPU and
no access to download pretrained model weights (Hugging Face Hub isn't reachable from this
environment). So instead of faking scale I don't have, everything here is real but
deliberately small: a from-scratch tiny transformer instead of a downloaded foundation
model, and TF-IDF retrieval instead of a downloaded embedding model. The architecture and
workflow are genuine; the scale is a sandbox constraint, not a design choice I'd make with
real infrastructure.

## Part 1: RAG Pipeline (`rag/`)

A working retrieval-augmented generation system over a small hand-written knowledge base
of basketball analytics concepts (shot quality, player tracking, defensive matchups, lineup
efficiency, load management, pick-and-roll coverages).

**How it works:**
1. **Retrieval** (`retriever.py`): documents are chunked with overlap, then indexed with
   TF-IDF. A query is embedded the same way and ranked against all chunks by cosine
   similarity, with a minimum relevance threshold so irrelevant queries correctly return
   nothing instead of a low-confidence guess.
2. **Generation** (`generator.py`): two interchangeable backends behind the same interface.
   - `ExtractiveGenerator` (what actually runs): synthesizes an answer directly from the
     retrieved text, with citations, no external API call. Fully offline.
   - `ClaudeGenerator`: a documented but *not executed* reference implementation showing
     exactly how this pipeline would call Claude to generate a grounded answer from the
     same retrieved context, if a live API key were available. Swapping it in requires no
     changes anywhere else in the pipeline, since both implement the same `generate()`
     interface.
3. **API** (`app.py`): a small Flask service exposing `/query` (POST) and `/health` (GET).

**Verified behavior:**
- Correctly retrieves and cites the right source document for in-domain questions (e.g.
  asking about pick-and-roll coverage correctly returns the pick-and-roll document at 0.47
  cosine similarity, other documents don't clear the relevance threshold).
- Correctly returns "no relevant information found" for out-of-domain questions (tested with
  both basketball-adjacent-but-wrong questions like soccer's offside rule, and completely
  unrelated ones) instead of hallucinating an answer from unrelated chunks.

## Part 2: Fine-Tuning (`finetune/`)

Demonstrates the actual mechanics of adapting a pretrained model to a downstream task,
with a controlled comparison proving the pretraining step is doing something real rather
than just running the code path.

**How it works:**
1. **Data** (`generate_data.py`): a templated synthetic play-by-play dataset (real NBA
   play-by-play data wasn't available in this environment), covering 8 event types
   (made shot, missed shot, turnover, foul, rebound, assist, block, steal). A large
   unlabeled pool (3,200 examples) is used for pretraining; a deliberately small labeled
   set (5 examples per class, 40 total) is used for fine-tuning, which is the realistic
   scenario where pretraining should actually matter.
2. **Model** (`model.py`): a small transformer encoder (~40K params) built from scratch in
   PyTorch, small enough to train on CPU in minutes.
3. **Pretraining** (`pretrain.py`): trains the encoder with a masked-language-modeling
   objective on the unlabeled corpus, the same idea BERT-style models use. MLM loss drops
   from 4.41 to 1.38 over 25 epochs.
4. **Fine-tuning** (`finetune.py`): adds a classification head on top of the *same* encoder
   weights and fine-tunes on the small labeled set, then evaluates on a held-out test set of
   320 examples. Run against two conditions to isolate the pretraining effect:
   - Starting from the pretrained encoder weights.
   - Starting from randomly initialized weights (identical architecture, no pretraining).

**Result** (averaged over 5 random seeds, same data and training budget for both
conditions):

| Condition | Mean test accuracy |
|---|---|
| Fine-tuned from pretrained weights | **80.4%** |
| Trained from scratch (no pretraining) | 71.7% |

Pretraining wins in all 5 seeds, by an average of **+8.7 percentage points**. This is a
genuine, reproducible demonstration that the pretrain-then-fine-tune workflow works, at
small scale. (Worth noting honestly: at full scale with more pretraining epochs, this gap
was small — pretraining only showed its real advantage once the base model was trained
long enough to actually learn useful structure. That in itself was a useful, honest
finding about how much pretraining is needed before it pays off.)

## Running it

```bash
pip install torch scikit-learn rank_bm25 flask

# RAG pipeline
python3 -m rag.pipeline          # CLI demo with sample questions
python3 -m rag.app               # Flask API on :5050

# Fine-tuning
python3 finetune/generate_data.py
python3 -m finetune.pretrain     # ~1-2 min on CPU
python3 -m finetune.finetune     # runs both conditions across 5 seeds

# Tests
python3 test_pipeline.py
```

## What I'd do differently with real infrastructure

- Swap TF-IDF retrieval for a downloaded sentence-embedding model (dense retrieval usually
  outperforms TF-IDF, especially for queries that don't share exact vocabulary with the
  source text).
- Swap the from-scratch tiny transformer for an actual pretrained foundation model (e.g.
  fine-tuning a small open-weight model with LoRA), which is what "fine-tuning foundation
  models" means at production scale.
- Wire up `ClaudeGenerator` for real, and evaluate generation quality (not just retrieval
  quality) with a proper eval set.
- Use real play-by-play and tracking data instead of templated synthetic text, which would
  also make the fine-tuning task meaningfully harder and more realistic.
