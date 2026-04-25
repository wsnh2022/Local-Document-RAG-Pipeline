# lancedb_chunking_project — Development Plan
> Local CLI RAG pipeline: ingest TXT/PDF/DOCX/MD → chunk → embed → store in LanceDB → query → OpenRouter grounded answer. Stack locked. Build vertically.

---

## LOCKED STACK

| Layer | Tool | Version | Notes |
|---|---|---|---|
| Language | Python | 3.11.x | Do NOT use 3.12 — torch/sentence-transformers break silently |
| Vector DB | lancedb | 0.6.13 | Pin exact minor — API changed every minor release |
| Embeddings | sentence-transformers | 2.7.0 | Uses `all-MiniLM-L6-v2` model — 384-dim vectors |
| Embedding model | all-MiniLM-L6-v2 | (auto-downloaded) | Pre-download in Phase 0 — never lazy-load at query time |
| PDF parsing | pymupdf (fitz) | 1.24.3 | Faster and more reliable than pypdf2 |
| DOCX parsing | python-docx | 1.1.2 | Only library that handles .docx reliably |
| MD parsing | markdown | 3.6 | Strips markdown syntax for clean text extraction |
| TXT parsing | built-in | — | Native Python `open()` — no extra library needed |
| Chunking | langchain-text-splitters | 0.2.2 | RecursiveCharacterTextSplitter — best for mixed docs |
| Token counting | tiktoken | 0.7.0 | Token-aware chunking + context budget enforcement |
| LLM API | openrouter via requests | 2.32.3 | REST calls to `https://openrouter.ai/api/v1/chat/completions` |
| LLM model | `mistralai/mistral-7b-instruct` | — | Free tier on OpenRouter — swap to better model later |
| CLI | argparse | built-in | No extra library needed |
| Hashing | hashlib | built-in | SHA256 per file — dedup and re-ingest tracking |
| Config | python-dotenv | 1.0.1 | `.env` for OpenRouter API key |

---

## PROJECT STRUCTURE

```
lancedb_chunking_project/
├── main.py                  # CLI entry point — argparse commands
├── .env                     # OPENROUTER_API_KEY (gitignored)
├── .env.example             # Template for .env (committed)
├── .gitignore
├── requirements.txt         # All pinned versions
│
├── ingestion/
│   ├── __init__.py
│   ├── file_loader.py       # Load raw text from TXT/PDF/DOCX/MD
│   ├── chunker.py           # Split text into chunks with metadata (token-aware)
│   ├── embedder.py          # sentence-transformers embedding logic
│   ├── hash_tracker.py      # SHA256 file hashing + skip logic
│   └── token_counter.py     # tiktoken token counting + context budget
│
├── storage/
│   ├── __init__.py
│   ├── lance_store.py       # LanceDB table create/insert/delete/query
│   └── schema.py            # LanceDB table schema definition
│
├── retrieval/
│   ├── __init__.py
│   ├── searcher.py          # Semantic search → top-k chunks
│   └── reranker.py          # Simple score threshold filter
│
├── llm/
│   ├── __init__.py
│   ├── openrouter_client.py # OpenRouter API calls with retry
│   └── prompt_builder.py    # RAG prompt + summarization prompt
│
└── data/
    └── lancedb/             # LanceDB files live here (gitignored)
```

**.gitignore entries that matter:**
```
.env
data/
__pycache__/
*.pyc
.venv/
*.egg-info/
```

---

## PHASE 0 — Environment Setup (Day 0)

### 1. Verify Python version
```bash
python --version
# Must show 3.11.x — if not, install via pyenv or python.org
```

### 2. Create virtual environment
```bash
cd lancedb_chunking_project
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Create requirements.txt
```txt
lancedb==0.6.13
sentence-transformers==2.7.0
pymupdf==1.24.3
python-docx==1.1.2
markdown==3.6
langchain-text-splitters==0.2.2
tiktoken==0.7.0
requests==2.32.3
python-dotenv==1.0.1
numpy==1.26.4
torch==2.3.1
```

### 4. Install all dependencies
```bash
pip install -r requirements.txt
```

### 5. Pre-download embedding model (CRITICAL — do this now)
```python
# run_once_download_model.py — run this once in Phase 0
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
print("Model downloaded and cached at ./models")
```
```bash
python run_once_download_model.py
```
> ⚠️ **Gotcha:** Never let the model download at query time. Always pre-download and point `cache_folder` to `./models`. Default downloads to `~/.cache` which is slow and unpredictable.

### 6. Create .env file
```bash
# .env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
LANCEDB_PATH=./data/lancedb
MODEL_CACHE=./models
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
MAX_CHUNK_TOKENS=150
MAX_CONTEXT_TOKENS=3000
TIKTOKEN_ENCODING=cl100k_base
```

### 7. Create .env.example
```bash
OPENROUTER_API_KEY=
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
LANCEDB_PATH=./data/lancedb
MODEL_CACHE=./models
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
MAX_CHUNK_TOKENS=150
MAX_CONTEXT_TOKENS=3000
TIKTOKEN_ENCODING=cl100k_base
```

### 8. Git init
```bash
git init
git add requirements.txt .env.example .gitignore
git commit -m "Phase 0: environment setup"
```

**Success Criteria:**
- [ ] `python --version` shows 3.11.x
- [ ] `pip install -r requirements.txt` completes with no errors
- [ ] `run_once_download_model.py` prints "Model downloaded" and `./models/` folder exists
- [ ] `.env` file exists with valid OpenRouter API key
- [ ] `data/` and `.env` are in `.gitignore`

---

## PHASE 1 — LanceDB Schema + Storage Layer (Day 1 AM)

Build the database layer first — everything else depends on it.

### storage/schema.py
```python
import lancedb
from lancedb.pydantic import LanceModel, Vector
from typing import Optional

VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension

class DocumentChunk(LanceModel):
    id: str                      # SHA256(file_path) + chunk_index
    file_name: str               # Original filename
    file_path: str               # Absolute path
    file_hash: str               # SHA256 of file content
    chunk_index: int             # Position in document
    chunk_text: str              # Raw chunk text
    vector: Vector(VECTOR_DIM)   # Embedding vector
    char_count: int              # Length of chunk
    token_count: int             # Token count via tiktoken
```

### storage/lance_store.py
```python
import os
import lancedb
from storage.schema import DocumentChunk
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "documents"

def get_db():
    path = os.getenv("LANCEDB_PATH", "./data/lancedb")
    os.makedirs(path, exist_ok=True)
    return lancedb.connect(path)

def get_or_create_table():
    db = get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=DocumentChunk)

def insert_chunks(chunks: list[dict]):
    table = get_or_create_table()
    table.add(chunks)
    print(f"  Inserted {len(chunks)} chunks into LanceDB")

def delete_by_file_hash(file_hash: str):
    table = get_or_create_table()
    table.delete(f"file_hash = '{file_hash}'")
    print(f"  Deleted chunks for file_hash={file_hash}")

def hash_exists(file_hash: str) -> bool:
    table = get_or_create_table()
    results = table.search().where(f"file_hash = '{file_hash}'").limit(1).to_list()
    return len(results) > 0

def search_chunks(query_vector: list[float], top_k: int = 5) -> list[dict]:
    table = get_or_create_table()
    results = (
        table.search(query_vector)
        .limit(top_k)
        .to_pandas()
    )
    return results.to_dict(orient="records")

def get_table_stats() -> dict:
    table = get_or_create_table()
    df = table.to_pandas()
    return {
        "total_chunks": len(df),
        "unique_files": df["file_name"].nunique() if len(df) > 0 else 0
    }
```

### Test Phase 1
```bash
python -c "
from storage.lance_store import get_or_create_table, get_table_stats
table = get_or_create_table()
print('Table created:', table)
print('Stats:', get_table_stats())
"
```

> ⚠️ **LanceDB Gotcha:** If you change the schema (add/remove columns), you MUST delete `./data/lancedb/` and recreate. There is no migration tooling.

**Success Criteria:**
- [ ] `get_or_create_table()` runs without error
- [ ] `./data/lancedb/` folder is created on disk
- [ ] `get_table_stats()` returns `{"total_chunks": 0, "unique_files": 0}`

---

## PHASE 2 — File Parsing Layer (Day 1 PM)

### ingestion/file_loader.py
```python
import os
import fitz  # pymupdf
import docx
import markdown
import re
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

def load_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    loaders = {
        ".txt": _load_txt,
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".md": _load_md,
    }
    return loaders[ext](file_path)

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _load_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

def _load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def _load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    html = markdown.markdown(raw)
    return re.sub(r"<[^>]+>", "", html)  # strip HTML tags

def get_all_files(folder_path: str) -> list[str]:
    files = []
    for root, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return sorted(files)
```

### Test Phase 2
```bash
# Create a test file
echo "Hello this is a test document about LanceDB vector databases." > /tmp/test.txt

python -c "
from ingestion.file_loader import load_file, get_all_files
text = load_file('/tmp/test.txt')
print('Loaded text:', text[:100])
"
```

**Success Criteria:**
- [ ] `load_file()` returns clean text for TXT, PDF, DOCX, and MD files
- [ ] `get_all_files()` walks a folder and returns only supported extensions
- [ ] No crash on empty paragraphs in DOCX

---

## PHASE 3 — Hashing + Chunking + Token Counter Layer (Day 2 AM)

### ingestion/hash_tracker.py
```python
import hashlib

def hash_file(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()

def make_chunk_id(file_hash: str, chunk_index: int) -> str:
    return f"{file_hash[:16]}_{chunk_index:04d}"
```

### ingestion/token_counter.py
```python
import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

_encoder = None

def get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        encoding = os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
        _encoder = tiktoken.get_encoding(encoding)
    return _encoder

def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text))

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])

def chunks_token_report(chunks: list[dict]) -> dict:
    counts = [count_tokens(c["chunk_text"]) for c in chunks]
    max_chunk_tokens = int(os.getenv("MAX_CHUNK_TOKENS", 150))
    return {
        "total_chunks": len(counts),
        "min_tokens": min(counts),
        "max_tokens": max(counts),
        "avg_tokens": round(sum(counts) / len(counts), 1),
        "oversized": sum(1 for c in counts if c > max_chunk_tokens)
    }
```

> ⚠️ **Gotcha:** `cl100k_base` is used by GPT-3.5/4 and Mistral — correct default for OpenRouter models. If you switch to a very different model family token counts may differ slightly but are close enough for chunking.

### ingestion/chunker.py
```python
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.hash_tracker import hash_file, make_chunk_id
from ingestion.file_loader import load_file_safe
from ingestion.token_counter import count_tokens
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def chunk_file(file_path: str) -> list[dict]:
    max_tokens = int(os.getenv("MAX_CHUNK_TOKENS", 150))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 20))

    file_hash = hash_file(file_path)
    raw_text = load_file_safe(file_path)

    if raw_text is None:
        return []

    # Token-aware splitter — chunk_size is in TOKENS not characters
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",   # uses cl100k_base encoding
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(raw_text)

    return [
        {
            "id": make_chunk_id(file_hash, i),
            "file_name": Path(file_path).name,
            "file_path": os.path.abspath(file_path),
            "file_hash": file_hash,
            "chunk_index": i,
            "chunk_text": chunk,
            "char_count": len(chunk),
            "token_count": count_tokens(chunk),
        }
        for i, chunk in enumerate(chunks)
        if chunk.strip()
    ]
```

### Test Phase 3
```bash
python -c "
from ingestion.chunker import chunk_file
chunks = chunk_file('/tmp/test.txt')
print(f'Total chunks: {len(chunks)}')
print('First chunk:', chunks[0])
"
```

**Success Criteria:**
- [ ] `chunk_file()` returns list of dicts with all required keys including `token_count`
- [ ] Each chunk has a unique `id`
- [ ] `file_hash` is consistent — same file always produces same hash
- [ ] `token_count` per chunk stays at or below `MAX_CHUNK_TOKENS`
- [ ] Empty chunks are filtered out

---

## PHASE 4 — Embedding Layer (Day 2 PM)

### ingestion/embedder.py
```python
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        cache = os.getenv("MODEL_CACHE", "./models")
        _model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    return [v.tolist() for v in vectors]

def embed_query(text: str) -> list[float]:
    model = get_model()
    vector = model.encode([text])[0]
    return vector.tolist()

def attach_embeddings(chunks: list[dict]) -> list[dict]:
    texts = [c["chunk_text"] for c in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    return chunks
```

### Test Phase 4
```bash
python -c "
from ingestion.embedder import embed_query, embed_texts
v = embed_query('What is LanceDB?')
print('Vector dim:', len(v))
print('First 5 values:', v[:5])
"
```

> ⚠️ **Gotcha:** Model loads from `./models` folder. If that folder is missing, run `run_once_download_model.py` again. Do NOT let it download silently at runtime.

**Success Criteria:**
- [ ] `embed_query()` returns a list of 384 floats
- [ ] `attach_embeddings()` adds `vector` key to every chunk dict
- [ ] Model loads from `./models` — no internet call at runtime

---

## PHASE 5 — Ingestion Pipeline (Day 3 AM)

This phase wires Phases 2 + 3 + 4 + 1 into one `ingest` command.

### main.py (ingestion section)
```python
import argparse
import os
from dotenv import load_dotenv
from ingestion.file_loader import get_all_files
from ingestion.chunker import chunk_file
from ingestion.embedder import attach_embeddings
from ingestion.hash_tracker import hash_file
from storage.lance_store import (
    insert_chunks, hash_exists, delete_by_file_hash, get_table_stats
)

load_dotenv()

def run_ingest(folder_path: str, force_reingest: bool = False):
    files = get_all_files(folder_path)
    if not files:
        print(f"No supported files found in: {folder_path}")
        return

    print(f"\nFound {len(files)} file(s) to process\n")
    skipped, ingested = 0, 0

    for file_path in files:
        file_hash = hash_file(file_path)
        fname = os.path.basename(file_path)

        if hash_exists(file_hash) and not force_reingest:
            print(f"  [SKIP] {fname} — already ingested")
            skipped += 1
            continue

        if force_reingest and hash_exists(file_hash):
            print(f"  [DELETE] {fname} — removing old chunks")
            delete_by_file_hash(file_hash)

        print(f"  [INGEST] {fname}")
        chunks = chunk_file(file_path)
        chunks = attach_embeddings(chunks)
        insert_chunks(chunks)
        ingested += 1

    stats = get_table_stats()
    print(f"\nDone. Ingested: {ingested} | Skipped: {skipped}")
    print(f"DB total: {stats['total_chunks']} chunks from {stats['unique_files']} files\n")
```

### Test Phase 5
```bash
# Create a small test folder with 2 files
mkdir -p /tmp/test_docs
echo "LanceDB is a vector database built on the Lance format." > /tmp/test_docs/intro.txt
echo "Semantic search finds documents by meaning not keywords." > /tmp/test_docs/search.txt

python main.py --ingest /tmp/test_docs
```

Expected output:
```
Found 2 file(s) to process
  [INGEST] intro.txt — Inserted 1 chunks
  [INGEST] search.txt — Inserted 1 chunks

Done. Ingested: 2 | Skipped: 0
DB total: 2 chunks from 2 files
```

Run again — both should be skipped:
```bash
python main.py --ingest /tmp/test_docs
# Both files: [SKIP]
```

**Success Criteria:**
- [ ] First run ingests all files and reports chunk count
- [ ] Second run on same folder skips all files
- [ ] `--reingest` flag deletes old chunks and re-embeds
- [ ] `data/lancedb/` grows in size after ingestion

---

## PHASE 6 — Semantic Search + Retrieval (Day 3 PM)

### retrieval/searcher.py
```python
import os
from ingestion.embedder import embed_query
from ingestion.token_counter import count_tokens, truncate_to_tokens
from storage.lance_store import search_chunks
from dotenv import load_dotenv

load_dotenv()

def retrieve(query: str) -> list[dict]:
    top_k = int(os.getenv("TOP_K", 5))
    query_vector = embed_query(query)
    results = search_chunks(query_vector, top_k=top_k)
    return results

def fit_chunks_to_context(chunks: list[dict]) -> list[dict]:
    """Trim chunk list to fit within MAX_CONTEXT_TOKENS budget."""
    max_context = int(os.getenv("MAX_CONTEXT_TOKENS", 3000))
    fitted, total = [], 0

    for chunk in chunks:
        tokens = count_tokens(chunk["chunk_text"])
        if total + tokens > max_context:
            remaining = max_context - total
            if remaining > 50:  # only include if at least 50 tokens fit
                chunk = chunk.copy()
                chunk["chunk_text"] = truncate_to_tokens(chunk["chunk_text"], remaining)
                fitted.append(chunk)
            break
        fitted.append(chunk)
        total += tokens

    return fitted

def format_results(results: list[dict]) -> str:
    if not results:
        return "No relevant chunks found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['file_name']} (chunk {r['chunk_index']}) — {r.get('token_count', '?')} tokens")
        lines.append(f"    {r['chunk_text'][:200]}...")
        lines.append("")
    return "\n".join(lines)
```

### Test Phase 6
```bash
python -c "
from retrieval.searcher import retrieve, format_results
results = retrieve('what is vector database')
print(format_results(results))
"
```

**Success Criteria:**
- [ ] `retrieve()` returns list of chunk dicts with `file_name` and `chunk_text`
- [ ] Results are ranked by similarity (most relevant first)
- [ ] Empty DB returns empty list gracefully

---

## PHASE 7 — OpenRouter LLM Integration (Day 4 AM)

### llm/openrouter_client.py
```python
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3

def call_openrouter(messages: list[dict]) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "lancedb_chunking_project",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.2,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=30
            )
            if response.status_code == 401:
                raise ValueError("Invalid OpenRouter API key — check .env")
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API error: {e}")

    raise RuntimeError("All retry attempts failed")
```

### llm/prompt_builder.py
```python
def build_rag_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join([
        f"[Source: {c['file_name']}, chunk {c['chunk_index']}]\n{c['chunk_text']}"
        for c in chunks
    ])
    system = (
        "You are a helpful assistant. Answer questions strictly based on the provided context. "
        "If the context doesn't contain enough information, say so clearly. "
        "Always mention which source file(s) you used in your answer."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def build_summary_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n".join([c["chunk_text"] for c in chunks])
    system = (
        "You are a summarization assistant. Summarize the provided content clearly and concisely. "
        "Focus on what is most relevant to the user's topic."
    )
    user = f"Topic: {query}\n\nContent to summarize:\n{context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def needs_summary(query: str) -> bool:
    summary_keywords = ["summarize", "summary", "overview", "brief", "tldr", "what does", "explain all"]
    return any(kw in query.lower() for kw in summary_keywords)
```

### Test Phase 7
```bash
python -c "
from llm.openrouter_client import call_openrouter
reply = call_openrouter([{'role': 'user', 'content': 'Say hello in one sentence.'}])
print('OpenRouter reply:', reply)
"
```

> ⚠️ **Gotcha:** Always set `timeout=30` — default requests timeout is too fast for LLM APIs. Check `status_code == 401` explicitly for key errors.

**Success Criteria:**
- [ ] `call_openrouter()` returns a string response
- [ ] 401 error gives a clear "Invalid API key" message
- [ ] Timeout retries 3 times with backoff before raising

---

## PHASE 8 — Full CLI Query Loop (Day 4 PM)

Wire everything into `main.py` with an interactive prompt.

### main.py (complete)
```python
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def run_ingest(folder_path: str, force_reingest: bool = False):
    from ingestion.file_loader import get_all_files
    from ingestion.chunker import chunk_file
    from ingestion.embedder import attach_embeddings
    from ingestion.hash_tracker import hash_file
    from storage.lance_store import (
        insert_chunks, hash_exists, delete_by_file_hash, get_table_stats
    )

    files = get_all_files(folder_path)
    if not files:
        print(f"No supported files found in: {folder_path}")
        return

    print(f"\nFound {len(files)} file(s) to process\n")
    skipped, ingested = 0, 0

    for file_path in files:
        file_hash = hash_file(file_path)
        fname = os.path.basename(file_path)

        if hash_exists(file_hash) and not force_reingest:
            print(f"  [SKIP] {fname} — already ingested")
            skipped += 1
            continue

        if force_reingest and hash_exists(file_hash):
            print(f"  [DELETE] {fname} — removing old chunks")
            delete_by_file_hash(file_hash)

        print(f"  [INGEST] {fname}")
        chunks = chunk_file(file_path)
        chunks = attach_embeddings(chunks)
        insert_chunks(chunks)
        ingested += 1

    from storage.lance_store import get_table_stats
    stats = get_table_stats()
    print(f"\nDone. Ingested: {ingested} | Skipped: {skipped}")
    print(f"DB total: {stats['total_chunks']} chunks from {stats['unique_files']} files\n")

def run_query_loop():
    from retrieval.searcher import retrieve, format_results, fit_chunks_to_context
    from llm.openrouter_client import call_openrouter
    from llm.prompt_builder import build_rag_prompt, build_summary_prompt, needs_summary
    from storage.lance_store import get_table_stats

    stats = get_table_stats()
    if stats["total_chunks"] == 0:
        print("No data ingested yet. Run: python main.py --ingest <folder>")
        return

    print(f"\nReady. DB has {stats['total_chunks']} chunks from {stats['unique_files']} files.")
    print("Type your question (or 'exit' to quit)\n")

    while True:
        try:
            query = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        print("\nSearching...\n")
        chunks = retrieve(query)
        chunks = fit_chunks_to_context(chunks)  # enforce context token budget

        if not chunks:
            print("No relevant content found in the database.\n")
            continue

        print("Top sources found:")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c['file_name']} — chunk {c['chunk_index']}")
        print()

        if needs_summary(query):
            messages = build_summary_prompt(query, chunks)
            print("[SUMMARY MODE]")
        else:
            messages = build_rag_prompt(query, chunks)
            print("[RAG MODE]")

        print("Calling OpenRouter...\n")
        try:
            answer = call_openrouter(messages)
            print("Answer:\n")
            print(answer)
            print("\n" + "─" * 60 + "\n")
        except Exception as e:
            print(f"LLM Error: {e}\n")

def run_stats():
    from storage.lance_store import get_table_stats, get_or_create_table
    from ingestion.token_counter import chunks_token_report
    import os
    stats = get_table_stats()
    print(f"\nDB Stats:")
    print(f"  Total chunks : {stats['total_chunks']}")
    print(f"  Unique files : {stats['unique_files']}")
    if stats["total_chunks"] > 0:
        table = get_or_create_table()
        df = table.to_pandas()
        print("\nIngested files:")
        for fname in df["file_name"].unique():
            count = len(df[df["file_name"] == fname])
            print(f"  {fname} — {count} chunks")
        chunks = df.to_dict(orient="records")
        report = chunks_token_report(chunks)
        print(f"\nToken Report:")
        print(f"  Min tokens/chunk : {report['min_tokens']}")
        print(f"  Max tokens/chunk : {report['max_tokens']}")
        print(f"  Avg tokens/chunk : {report['avg_tokens']}")
        print(f"  Oversized chunks : {report['oversized']}  (>{os.getenv('MAX_CHUNK_TOKENS', 150)} tokens)")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanceDB RAG CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="FOLDER", help="Ingest documents from folder")
    group.add_argument("--query", action="store_true", help="Start interactive query loop")
    group.add_argument("--stats", action="store_true", help="Show database statistics")
    group.add_argument("--reingest", metavar="FOLDER", help="Force re-ingest (deletes old chunks)")
    args = parser.parse_args()

    if args.ingest:
        run_ingest(args.ingest)
    elif args.reingest:
        run_ingest(args.reingest, force_reingest=True)
    elif args.query:
        run_query_loop()
    elif args.stats:
        run_stats()
```

### Test Phase 8 — Full end-to-end
```bash
# Step 1 — Ingest
python main.py --ingest /tmp/test_docs

# Step 2 — Query
python main.py --query
>> What is LanceDB?
>> summarize all the documents
>> exit

# Step 3 — Stats
python main.py --stats

# Step 4 — Re-ingest
python main.py --reingest /tmp/test_docs
```

**Success Criteria:**
- [ ] `--ingest` processes folder and reports skip/ingest counts
- [ ] `--query` launches interactive loop and returns LLM answers
- [ ] Summary mode triggers on "summarize", "overview" etc.
- [ ] Source filenames are shown for every answer
- [ ] `--reingest` deletes and re-embeds correctly
- [ ] `--stats` shows chunk count and file breakdown
- [ ] `exit` / `Ctrl+C` exits cleanly

---

## PHASE 9 — Error Handling + Stability (Day 5 AM)

### Global error cases to handle

| Symptom | Handler |
|---|---|
| Folder path not found | `--ingest` prints clear message and exits |
| Unsupported file type | `file_loader.py` raises ValueError with filename |
| Corrupted PDF | Wrap `fitz.open()` in try/except, skip file with warning |
| Empty file | Chunker returns empty list — ingest skips gracefully |
| OpenRouter 401 | Raise ValueError with "Check your API key in .env" |
| OpenRouter timeout | Retry 3x with exponential backoff |
| LanceDB schema mismatch | Print "Delete ./data/lancedb/ and re-ingest" |
| No data in DB at query time | Print message before entering query loop |
| KeyboardInterrupt in loop | Catch and exit cleanly |

### Add to file_loader.py
```python
def load_file_safe(file_path: str) -> str | None:
    try:
        return load_file(file_path)
    except Exception as e:
        print(f"  [WARN] Could not load {os.path.basename(file_path)}: {e}")
        return None
```

Update `chunk_file()` in `chunker.py` to use `load_file_safe()` and return `[]` on None.

**Success Criteria:**
- [ ] Corrupted PDF is skipped with warning — does not crash pipeline
- [ ] Missing folder path gives clear error message
- [ ] All OpenRouter errors show actionable messages
- [ ] Ctrl+C exits without traceback

---

## PHASE 10 — Final Verification (Day 5 PM)

```bash
# Full clean test on a real folder with mixed file types
mkdir -p /tmp/real_test
# Drop in 1 PDF, 1 DOCX, 1 MD, 1 TXT

python main.py --ingest /tmp/real_test
python main.py --stats
python main.py --query
>> [ask a question about the documents]
>> summarize all documents
>> exit

# Re-ingest test
python main.py --reingest /tmp/real_test
python main.py --stats  # chunk count should be same as first ingest
```

---

## TIMELINE

| Day | Phase | Key Output |
|---|---|---|
| Day 0 | Environment | Python 3.11, venv, all packages installed, model pre-downloaded |
| Day 1 AM | Phase 1 — LanceDB Schema | Table creates, hash_exists works, search_chunks works |
| Day 1 PM | Phase 2 — File Parsing | All 4 file types load clean text |
| Day 2 AM | Phase 3 — Hashing + Chunking | Chunks with IDs, hashes, metadata |
| Day 2 PM | Phase 4 — Embeddings | 384-dim vectors from local model |
| Day 3 AM | Phase 5 — Ingestion Pipeline | `--ingest` works, dedup works |
| Day 3 PM | Phase 6 — Semantic Search | `retrieve()` returns ranked chunks |
| Day 4 AM | Phase 7 — OpenRouter | LLM answers + summary mode |
| Day 4 PM | Phase 8 — Full CLI | End-to-end `--query` loop works |
| Day 5 | Phases 9-10 — Errors + Final Test | All edge cases handled, clean run on real docs |

---

## DEBUGGING MATRIX

| Symptom | Check First | Check Second |
|---|---|---|
| `ImportError: lancedb` | `pip install lancedb==0.6.13` in active venv | Confirm venv is activated: `which python` |
| `ValueError: Table schema mismatch` | Delete `./data/lancedb/` folder entirely | Schema changed — must recreate from scratch |
| Vector search returns empty results | Check `get_table_stats()` — is DB empty? | Run `--ingest` first before `--query` |
| Embedding dim mismatch error | `VECTOR_DIM` in schema.py must be 384 | Model was changed — delete DB and re-ingest |
| `401 Unauthorized` from OpenRouter | Check `OPENROUTER_API_KEY` in `.env` | Confirm `.env` is in project root, not subdirectory |
| OpenRouter timeout every call | Check internet connection | Increase timeout in `openrouter_client.py` |
| PDF loads as empty string | File is scanned image PDF — pymupdf can't OCR | Use OCR tool (tesseract) for scanned PDFs |
| Same file ingested twice | `hash_exists()` not being called | Check that `--reingest` flag is not set unintentionally |
| Model downloads at every run | `cache_folder` not set in `get_model()` | Check `MODEL_CACHE` env var and `./models/` folder exists |
| `torch` import error | Python version is 3.12 — use 3.11 | Reinstall torch: `pip install torch==2.3.1` |
| Chunks have no text | File encoding issue | Add `errors="ignore"` to all `open()` calls |
| LanceDB writes slow on first insert | Normal — Lance format compacts on first write | Subsequent writes are faster |
| LLM answer cuts off mid-sentence | `MAX_CONTEXT_TOKENS` too low — increase to 4000 | Check `fit_chunks_to_context()` is being called in query loop |
| Oversized chunks shown in `--stats` | `MAX_CHUNK_TOKENS` too high — lower to 100–150 | Confirm chunker uses `from_tiktoken_encoder` not character splitter |
| `tiktoken` import error in packaged build | Add `--hidden-import tiktoken_ext.openai_public` to PyInstaller | Also add `--hidden-import tiktoken_ext` |

---

## MVP LOCK — Do Not Build in Week 1

- Web UI or chat interface (learn the CLI flow first)
- Hybrid search combining keyword + vector (add after you understand pure vector search)
- Metadata filtering by date, file type, or custom tags
- Automated re-ingestion scheduler / file watcher
- Multi-user support or authentication
- Chat history / conversation memory
- Document versioning system
- Custom embedding model fine-tuning
- Reranking with a cross-encoder model

---

## FINAL CHECKLIST

- [ ] Python 3.11 confirmed — NOT 3.12
- [ ] All packages in `requirements.txt` with exact pinned versions
- [ ] `.env` is in `.gitignore` — API key never committed
- [ ] `.env.example` is committed with empty values
- [ ] Embedding model pre-downloaded to `./models/` in Phase 0
- [ ] `VECTOR_DIM = 384` matches `all-MiniLM-L6-v2` output dimension
- [ ] `data/lancedb/` is in `.gitignore` — DB files never committed
- [ ] `hash_exists()` called before every insert — no duplicate chunks
- [ ] `delete_by_file_hash()` called before re-ingest — no stale chunks
- [ ] OpenRouter timeout set to 30 seconds minimum
- [ ] 401 API key error handled with clear message
- [ ] `tiktoken==0.7.0` in requirements.txt — token-aware chunking active
- [ ] `MAX_CHUNK_TOKENS`, `MAX_CONTEXT_TOKENS`, `TIKTOKEN_ENCODING` set in `.env`
- [ ] `--stats` shows token report — no oversized chunks before calling done
- [ ] `fit_chunks_to_context()` called in query loop — context budget enforced
- [ ] Corrupted or unreadable files skip with warning — do not crash
- [ ] `--stats` shows current DB state at any time
- [ ] All 4 file types tested end-to-end before calling done
- [ ] Clean run verified: delete `./data/lancedb/`, re-ingest, re-query
