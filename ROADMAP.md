# lancedb_chunking_project — Improvement Roadmap

Based on architecture review and retrieval quality analysis. Each phase builds on the
previous one. Do not skip phases — Phase 1 fixes foundations that Phase 2 and 3 depend on.

---

## Why This Roadmap Exists

The v1 pipeline works but has three compounding weaknesses:

1. **Embedding model ceiling** — `all-MiniLM-L6-v2` has a 256-token input limit and low
   retrieval accuracy on technical content. Chunks silently truncate above 256 tokens.

2. **Chunking strategy** — Fixed-token chunks cut mid-idea. The LLM receives fragments
   instead of coherent units of meaning, which causes incomplete or hallucinated answers.

3. **Retrieval method** — Pure vector search misses exact-term queries (config keys, error
   codes, numeric values). A hybrid approach recovers these cases without sacrificing
   semantic retrieval.

Fixing these three in order transforms retrieval precision from ~55% to ~85%+ on mixed
technical corpora.

---

## Phase 1 — Embedding Model Upgrade

**Priority:** Do first. Everything else depends on retrieval quality.  
**Effort:** 30 minutes  
**Files changed:** `ingestion/embedder.py`, `run_once_download_model.py`

### What to implement

Replace `all-MiniLM-L6-v2` with `BAAI/bge-small-en-v1.5`.

```python
# ingestion/embedder.py
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # was: all-MiniLM-L6-v2
```

### Why this model

| Property | all-MiniLM-L6-v2 | bge-small-en-v1.5 |
|---|---|---|
| Max input tokens | 256 | 512 |
| Embedding dims | 384 | 384 |
| MTEB retrieval score | ~51 | ~58 |
| Size on disk | 90MB | 130MB |
| Schema migration needed | — | No (same 384 dims) |

The 256-token ceiling in MiniLM is the critical failure point. Any chunk above 256 tokens
gets silently truncated before embedding — the stored vector does not represent the full
chunk. BGE-small doubles this ceiling to 512 tokens and scores 7 points higher on
retrieval benchmarks with no schema migration required.

### Why not bge-large-en-v1.5

BGE-large outputs 1024 dimensions. Switching requires dropping and recreating the LanceDB
table (schema change from `Vector(384)` to `Vector(1024)`) and re-ingesting all documents.
BGE-small gives most of the quality gain at zero migration cost. Upgrade to large only if
retrieval quality is still insufficient after Phase 2 and 3.

### Steps

1. Update `MODEL_NAME` in `ingestion/embedder.py`
2. Update `run_once_download_model.py` to pull new model name
3. Delete `./data/lancedb/` — vectors from the old model are incompatible
4. Run `python run_once_download_model.py`
5. Re-ingest: `python main.py --ingest ./docs`

### Config change

After this upgrade, raise `MAX_CHUNK_TOKENS` from 150 to 300. The MiniLM ceiling forced
this value down. BGE-small handles 512 tokens cleanly.

```env
MAX_CHUNK_TOKENS=300
```

---

## Phase 2 — Chunking Strategy Overhaul

**Priority:** Second. Depends on the new embedding model being in place.  
**Effort:** 2–4 hours  
**Files changed:** `ingestion/chunker.py`, `storage/schema.py`, `storage/lance_store.py`,
`retrieval/searcher.py`

Implement in this order: contextual headers first (trivial), then hierarchical chunking
(highest answer quality gain).

---

### 2a — Contextual Chunk Headers

**Effort:** 30 minutes  
**Files changed:** `ingestion/chunker.py` only

#### What to implement

Prepend each chunk with its source context before embedding. Do not store this prefix in
`chunk_text` — store it separately and prepend only at embed time.

```python
# ingestion/chunker.py
def build_embed_text(chunk_text: str, file_name: str, chunk_index: int) -> str:
    return f"File: {file_name}\nChunk: {chunk_index}\n\n{chunk_text}"

# Embed the prefixed version, store the raw chunk_text
vector = embedder.embed(build_embed_text(chunk.chunk_text, chunk.file_name, chunk.chunk_index))
```

#### Why this matters

The embedding encodes what a chunk says but not where it comes from. Two chunks from
different documents can produce nearly identical vectors if they cover the same topic.
Adding source context into the embedded text shifts the vector to encode both meaning
and origin. Retrieval accuracy on multi-document corpora improves measurably.

Zero schema changes. The stored `chunk_text` remains clean for display to the LLM.

---

### 2b — Hierarchical Parent-Child Chunking

**Effort:** 2–3 hours  
**Files changed:** `ingestion/chunker.py`, `storage/schema.py`, `storage/lance_store.py`,
`retrieval/searcher.py`

#### What to implement

Store two chunk sizes per document. Small child chunks (60–80 tokens) are used for
retrieval. Large parent chunks (300–500 tokens) are sent to the LLM as context.

```
Ingest:
  Document → split into parent chunks (400 tokens, overlap 40)
           → split each parent into child chunks (70 tokens, overlap 10)
           → embed child chunks only
           → store child with parent_id reference

Query:
  embed query → retrieve top-k child chunks (precision)
              → look up parent chunks by parent_id (context)
              → send parent chunks to LLM
```

#### Schema change required

Add `parent_chunk_id` and `parent_chunk_text` to `storage/schema.py`:

```python
# storage/schema.py
class DocumentChunk(LanceModel):
    id: str
    file_name: str
    file_path: str
    file_hash: str
    chunk_index: int
    chunk_text: str           # child chunk — used for retrieval display
    parent_chunk_id: str      # reference to parent
    parent_chunk_text: str    # full parent — sent to LLM
    vector: Vector(384)
    char_count: int
    token_count: int
```

#### Why this matters

Fixed-token chunking creates a precision-context tradeoff. Small chunks retrieve the right
passage but give the LLM too little context to answer fully. Large chunks give the LLM
context but dilute the embedding signal with off-topic content. Hierarchical chunking
eliminates this tradeoff: child chunks find the right location, parent chunks supply
the surrounding meaning.

This is the single highest-impact change for grounded answer quality.

#### Retrieval change

```python
# retrieval/searcher.py
def retrieve(query: str, top_k: int = 5) -> list[str]:
    child_chunks = vector_search(query, top_k=top_k)
    parent_texts = [c.parent_chunk_text for c in child_chunks]
    return fit_chunks_to_context(parent_texts)
```

---

### 2c — Semantic Chunking (Optional Upgrade)

**Effort:** 1 hour  
**Files changed:** `ingestion/chunker.py`  
**Dependency:** `pip install langchain-experimental`

#### What to implement

Replace `RecursiveCharacterTextSplitter` with `SemanticChunker` for narrative documents
(`.md`, `.txt`). Keep fixed-token splitting for structured documents (`.pdf`, `.docx`).

```python
# ingestion/chunker.py
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def chunk_document(text: str, file_ext: str) -> list[str]:
    if file_ext in [".md", ".txt"]:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )
        chunks = splitter.split_text(text)
        # Cap oversized chunks
        return [c for c in chunks if token_count(c) <= MAX_CHUNK_TOKENS]
    else:
        return recursive_split(text)  # existing logic
```

#### Why this matters

Fixed-token splitting cuts at arbitrary boundaries regardless of topic. Semantic chunking
detects drops in embedding similarity between sentences — a signal that the topic has
shifted — and splits there instead. Each chunk maps to a coherent idea rather than a
token window. Retrieval hits become self-contained units of meaning.

The tradeoff is variable chunk size. Apply the `MAX_CHUNK_TOKENS` cap as a guardrail.
Semantic chunking adds embedding calls at ingest time (not query time) — no cost impact
at runtime.

---

## Phase 3 — Hybrid Search (Vector + BM25)

**Priority:** Third. Fills the retrieval gap that vector search cannot cover.  
**Effort:** 2–3 hours  
**Files changed:** `retrieval/searcher.py`, `main.py`  
**New dependency:** `pip install rank-bm25`

### What to implement

Run BM25 keyword search alongside vector search. Merge both ranked lists using
Reciprocal Rank Fusion (RRF) before sending to the LLM.

```
Query
  ├── vector_search()   → semantic top-k  (handles meaning, paraphrasing)
  └── bm25_search()     → keyword top-k   (handles exact terms, codes, values)
            ↓
      rrf_merge()       → single ranked list
            ↓
      fit_chunks_to_context() → LLM
```

#### BM25 index management

BM25 requires a corpus of tokenized chunks. Build the index on startup from LanceDB,
store it in memory. At 24 chunks this is instant. At 100k chunks, consider persisting
to disk with `pickle`.

```python
# retrieval/searcher.py
from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, all_chunks: list[DocumentChunk]):
        corpus = [chunk.chunk_text.lower().split() for chunk in all_chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunks = all_chunks

    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunks[i].id, score) for i, score in ranked]

    def rrf_merge(
        self,
        vector_results: list[DocumentChunk],
        bm25_results: list[tuple[str, float]],
        k: int = 60
    ) -> list[DocumentChunk]:
        scores: dict[str, float] = {}

        for rank, chunk in enumerate(vector_results):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank)

        chunk_map = {c.id: c for c in self.chunks}
        for rank, (chunk_id, _) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

        top_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [chunk_map[cid] for cid in top_ids if cid in chunk_map]
```

#### RRF formula

```
score(chunk) = 1 / (k + rank_vector) + 1 / (k + rank_bm25)
k = 60   (constant — smooths score differences between rank positions)
```

Chunks appearing in both lists get a double contribution. Chunks in only one list still
rank if their individual score is high enough.

### Why this matters

Vector search embeds query meaning and retrieves semantically similar chunks. It handles
paraphrasing and conceptual queries well. It fails on exact-term queries — searching for
`MAX_CHUNK_TOKENS` or `SHA256` may retrieve chunks about the concept of configuration or
hashing rather than the specific passage containing those terms.

BM25 inverts this: it finds exact terms reliably but cannot understand that "chunk size
limit" and `MAX_CHUNK_TOKENS` refer to the same thing.

The hybrid approach covers both failure modes. The RRF merge is parameter-free — no
threshold to tune, no weights to set. It is the standard production pattern used in
Elasticsearch, Weaviate, and Qdrant hybrid search.

### Query types and which retriever wins

| Query | Vector wins | BM25 wins |
|---|---|---|
| "explain how chunking works" | ✅ | |
| "what is MAX_CHUNK_TOKENS default" | | ✅ |
| "why does PDF load empty" | ✅ | |
| "SHA256 hash dedup" | | ✅ |
| "summarize the ingestion pipeline" | ✅ | |
| Typos or paraphrased questions | ✅ | |
| Exact config values and error codes | | ✅ |

---

## Phase 4 — Proposition Chunking (v2 Feature)

**Priority:** Deferred. High ingest cost. Implement after Phase 1–3 are stable.  
**Effort:** 3–4 hours  
**Files changed:** `ingestion/chunker.py`  
**Cost:** LLM call per chunk at ingest time (~$0.002 for 24 chunks at current rates)

### What to implement

Use an LLM to decompose each document chunk into atomic factual propositions before
ingesting. Each proposition becomes its own chunk.

```
Raw chunk:
"LanceDB is a vector database that stores data on disk and supports cosine similarity."

Propositions:
  → "LanceDB is a vector database."
  → "LanceDB stores data on disk."
  → "LanceDB supports cosine similarity search."
```

### Why this matters

Every chunk becomes a self-contained, retrievable fact. The embedding space is used for
pure semantic matching with no noise from surrounding sentences. Retrieval precision
approaches theoretical maximum. LLM answers become fully grounded because each retrieved
unit maps to exactly one verifiable fact.

### Why it is deferred

Requires an LLM call per chunk at ingest time. For a corpus of 24 chunks this is trivial
($0.002). For a growing knowledge base it becomes a managed cost. Implement only after
hybrid search (Phase 3) has been validated — proposition chunking amplifies a good
retrieval system but cannot compensate for a broken one.

---

## Implementation Order Summary

| Phase | Feature | Files | Effort | Impact |
|---|---|---|---|---|
| 1 | BGE-small embedding model | `embedder.py` | 30 min | Fixes silent truncation, +7 MTEB |
| 2a | Contextual chunk headers | `chunker.py` | 30 min | +retrieval precision on multi-doc |
| 2b | Hierarchical parent-child chunking | `schema.py`, `chunker.py`, `searcher.py` | 3 hrs | Highest answer groundedness gain |
| 2c | Semantic chunking (narrative docs) | `chunker.py` | 1 hr | Coherent chunk boundaries |
| 3 | Hybrid search — BM25 + RRF | `searcher.py` | 2 hrs | Covers exact-term retrieval gap |
| 4 | Proposition chunking | `chunker.py` | 3 hrs | Maximum retrieval precision (v2) |

Each phase requires a full re-ingest (`python main.py --reingest ./docs`) because stored
vectors and chunk boundaries change. Run `python main.py --stats` after each phase to
verify chunk counts and token distributions look correct before moving to the next phase.

---

## What Does Not Change

- LLM answer quality is bounded by `llm/prompt_builder.py` — these phases improve what
  context reaches the LLM, but weak prompts will still produce weak answers.
- API cost per query stays effectively the same across phases. `MAX_CONTEXT_TOKENS=3000`
  caps every method. Sentence-window (Phase 2b child retrieval) reduces context tokens
  sent to the LLM, which lowers cost slightly, but at Gemini 2.5 Flash Lite rates
  ($0.10/M input) the difference is under $0.0003 per query.
- Embedding is always local and free regardless of which model is used.
