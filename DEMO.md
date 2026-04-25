# Demo Walkthrough — local-document-rag-pipeline (CLI)

A step-by-step showcase run of the pipeline. Use this as a script for a live demo, screen recording, or portfolio walkthrough.

---

## Prerequisites

- Setup complete (see [README.md](./README.md) Quick Start)
- `.env` configured with a valid `OPENROUTER_API_KEY`
- `scripts/setup_model.py` already run (embedding model cached locally)

## Before Every Session — Activate the Virtual Environment

**You must activate the venv before running any command.** If you skip this, Python will use the system interpreter and `import lancedb` (and everything else) will fail.

```bash
# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

Your terminal prompt will show `(.venv)` when it's active:
```
(.venv) PS C:\your-projects\local-document-rag-pipeline>
```

To deactivate when you're done:
```bash
deactivate
```

---

## Step 1 — Check the sample documents

The `examples/` folder contains three sample documents covering different formats and topics:

```bash
ls examples/
```

```
what_is_rag.txt         — Plain text: explanation of RAG systems
vector_databases.md     — Markdown: deep dive on vector DBs and LanceDB
chunking_strategies.txt — Plain text: comparison of chunking approaches
```

---

## Step 2 — Ingest the documents

```bash
python main.py --ingest ./examples
```

Expected output:
```
[INGEST] what_is_rag.txt          →  4 chunks   (421 tokens)
[INGEST] vector_databases.md      →  8 chunks   (892 tokens)
[INGEST] chunking_strategies.txt  →  6 chunks   (618 tokens)
Done. 18 new chunks stored.
```

**What just happened:**
1. Each file was SHA256-hashed
2. Text extracted by the appropriate parser
3. Split into token-aware chunks (≤ 150 tokens each)
4. Each chunk embedded into a 384-dim vector via `all-MiniLM-L6-v2` (locally)
5. All chunks + vectors stored in LanceDB on disk

---

## Step 3 — View database stats

```bash
python main.py --stats
```

Expected output:
```
DB Stats:
  Total chunks : 18
  Unique files :  3

Ingested files:
  what_is_rag.txt          —  4 chunks
  vector_databases.md      —  8 chunks
  chunking_strategies.txt  —  6 chunks

Token Report:
  Min tokens/chunk :  32
  Max tokens/chunk : 149
  Avg tokens/chunk : 107.6
  Oversized chunks :   0  (> 150 tokens)
```

Notice **0 oversized chunks** — the token-aware chunker kept every chunk within budget.

---

## Step 4 — Run a RAG query

```bash
python main.py --query
```

Try this question:
```
>> what is retrieval augmented generation?
```

Expected flow:
```
[RAG MODE] Retrieved 5 chunks (538 tokens)

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval
system with a language model. Instead of relying solely on the model's training
data, RAG fetches relevant documents at query time and uses them as context...

Sources: what_is_rag.txt, vector_databases.md
```

The LLM answers **strictly from the retrieved context** and cites which files the answer came from.

---

## Step 5 — Try a factual precision query

```
>> what are the differences between character-based and token-based chunking?
```

This tests semantic retrieval precision — the answer should come from `chunking_strategies.txt`:

```
[RAG MODE] Retrieved 5 chunks (612 tokens)

Character-based chunking splits text at fixed character counts, which can
produce chunks with very different token counts depending on content density...
Token-based chunking uses tiktoken to count tokens directly, ensuring each
chunk stays within a predictable token budget for the LLM...

Sources: chunking_strategies.txt
```

---

## Step 6 — Switch to summary mode

```
>> summarize all documents
```

The keyword `summarize` triggers **summary mode** automatically:

```
[SUMMARY MODE] Retrieved 5 chunks (756 tokens)

The documents cover three interconnected topics in modern NLP pipelines:
1. RAG systems — how retrieval improves LLM accuracy by grounding answers...
2. Vector databases — how LanceDB stores and searches embeddings on disk...
3. Chunking strategies — why token-aware chunking outperforms character-based...
```

---

## Step 7 — Demonstrate deduplication (re-ingest safety)

Exit the query loop (`exit` or `Ctrl+C`) and re-run ingest on the same folder:

```bash
python main.py --ingest ./examples
```

```
[SKIP] what_is_rag.txt          already ingested (hash match)
[SKIP] vector_databases.md      already ingested (hash match)
[SKIP] chunking_strategies.txt  already ingested (hash match)
Done. 0 new chunks stored.
```

**Zero duplicates.** The SHA256 hash of each file matches what's already in LanceDB, so all files are skipped. Re-running ingestion on the same folder is completely safe.

---

## Step 8 — Force re-ingest

```bash
python main.py --reingest ./examples
```

```
[DELETE] what_is_rag.txt          removed 4 old chunks
[INGEST] what_is_rag.txt          → 4 chunks  (421 tokens)
[DELETE] vector_databases.md      removed 8 old chunks
[INGEST] vector_databases.md      → 8 chunks  (892 tokens)
...
Done. 18 chunks re-stored.
```

Use `--reingest` when document content has actually changed and you need fresh embeddings.

---

## Step 9 — Delete a single file (interactive)

```bash
python main.py --delete
```

The command lists every ingested file so you can pick by number — no need to type the full name manually:

```
Ingested files:
  [1] what_is_rag.txt — 4 chunks
  [2] vector_databases.md — 8 chunks
  [3] chunking_strategies.txt — 6 chunks

  Enter numbers to delete (e.g. 1,3,5  or  1-3  or  all)
  Selection: 1

Files to delete (1):
  - what_is_rag.txt (4 chunks)

Delete 1 file(s) / 4 chunks? [y/N]: y
  Deleted 'what_is_rag.txt' — 4 chunks removed

Done. 1 file(s) deleted.
```

Verify it's gone:

```bash
python main.py --stats
```

```
DB Stats:
  Total chunks : 14
  Unique files :  2

Ingested files:
  vector_databases.md      —  8 chunks
  chunking_strategies.txt  —  6 chunks
```

---

## Step 10 — Delete multiple files using a list file

This works like `requirements.txt` — create `delete_files.txt` in the project root:

```
# delete_files.txt
# These are outdated — removing from DB
vector_databases.md
chunking_strategies.txt
```

Then run:

```bash
python main.py --delete-from delete_files.txt
```

```
Files to delete (2):
  - vector_databases.md (8 chunks)
  - chunking_strategies.txt (6 chunks)

Delete 2 file(s) / 14 chunks? [y/N]: y
  Deleted 'vector_databases.md' — 8 chunks removed
  Deleted 'chunking_strategies.txt' — 6 chunks removed

Done. 2 file(s) deleted.
```

If a name in the list isn't in the DB, it's reported and skipped — no error:

```
Not found in DB (1) — will be skipped:
  - old_report.pdf
```

This is useful when you keep a running `delete_files.txt` and re-run it after ingesting fresh versions of the same documents.

---

## Step 11 — Windows launcher (start.bat)

Double-click `start.bat` to access all commands through a numbered menu — no terminal flags needed:

```
================================
  lancedb_chunking - chatbot
================================

 [1] Query (interactive)
 [2] Ingest docs folder
 [3] Re-ingest docs folder
 [4] View DB stats
 [5] Delete ingested file
 [6] Delete from delete_files.txt
 [7] Exit
```

Each option prints a loading message before Python starts — the screen never goes blank during the cold-start import phase.

---

## Key talking points for a presentation

- **Why token-aware chunking matters**: Show Step 3 — every chunk is under 150 tokens. Compare to character-based chunking where the same setting might produce chunks from 30 to 300 tokens.
- **Why deduplication matters**: Show Step 7 — re-running `--ingest` is safe and instant.
- **Context budget in action**: The `(538 tokens)` in Step 4 output means the LLM received exactly that many tokens of context — always under `MAX_CONTEXT_TOKENS=3000`, regardless of how many chunks matched.
- **Auto mode detection**: Step 6 requires no extra flag — just use the word "summarize" naturally.
- **Fully offline ingestion**: The embedding model runs locally. Only the final LLM call hits the network.
- **Granular delete without full wipe**: Step 9 shows targeted removal of one file — the rest of the DB stays intact. No need to wipe and re-ingest everything just to remove one stale document.
- **Bulk delete via file list**: Step 10 mirrors the `requirements.txt` pattern — maintain a `delete_files.txt`, run one command, done. Missing names are skipped gracefully so the list can stay in the repo without breaking anything.

---

## Sample queries to explore further

```
>> what is a vector database?
>> how does cosine similarity search work?
>> explain the difference between RAG and fine-tuning
>> what is LanceDB and how is it different from other vector stores?
>> what chunking strategy should I use for long documents?
>> tldr all documents
>> overview of the main topics
```
