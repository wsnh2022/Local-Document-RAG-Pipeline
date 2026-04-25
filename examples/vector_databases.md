# Vector Databases and LanceDB

## What is a Vector Database?

A vector database is a database system designed to store, index, and search high-dimensional vectors efficiently. Unlike relational databases that store structured rows and columns, or document stores that handle JSON, vector databases are optimised for a specific operation: finding the nearest neighbours of a given query vector in a high-dimensional space.

This operation is called Approximate Nearest Neighbour (ANN) search. Given a query vector, the database returns the k vectors in its index that are most similar according to a distance metric — typically cosine similarity or Euclidean distance.

Vector databases are the backbone of modern semantic search, recommendation systems, and retrieval-augmented generation (RAG) pipelines.

## How Vectors Are Produced

Vectors are produced by embedding models — neural networks trained to map text, images, or other data into a dense numerical representation. For text, models like `all-MiniLM-L6-v2` from the sentence-transformers library produce 384-dimensional vectors. Larger models like OpenAI's `text-embedding-3-large` produce 3072-dimensional vectors.

The key property of these embeddings is that semantically similar inputs map to nearby points in the vector space. "What is a neural network?" and "Explain deep learning" will produce vectors that are close together, even though they share almost no words.

## Common Vector Databases

| Database | Storage | Key Feature |
|---|---|---|
| LanceDB | On-disk (Lance format) | Zero server setup, columnar storage |
| Pinecone | Cloud-hosted | Fully managed, high scale |
| Weaviate | Self-hosted or cloud | Built-in object storage |
| Chroma | In-memory or disk | Lightweight, easy to set up |
| Qdrant | Self-hosted or cloud | Payload filtering, Rust-based |
| pgvector | PostgreSQL extension | Familiar SQL interface |

## LanceDB

LanceDB is an open-source, embedded vector database that runs directly on disk using the Lance columnar format. It requires no server process — it runs as a library inside your application, writing files to a local path you specify.

### Key Properties

**Serverless**: LanceDB opens directly from a file path. There is no daemon, no port, no network call. This makes it ideal for local development, CLI tools, and research pipelines where you want zero infrastructure overhead.

**Columnar storage**: The Lance file format stores data in columns rather than rows. This makes reading a specific field (such as the vector column) very fast without loading unrelated data.

**Versioning**: Lance supports dataset versioning out of the box. Every write creates a new version, and older versions can be accessed or rolled back to.

**Python-native**: LanceDB has a clean Python API with native Pandas and Arrow integration.

### Schema in This Project

In this RAG pipeline, each LanceDB record represents one document chunk:

- `id` — unique identifier derived from file hash and chunk index
- `file_name` — original filename
- `file_path` — absolute path on disk
- `file_hash` — SHA256 hash of the source file
- `chunk_index` — position of this chunk within the document
- `chunk_text` — raw text of the chunk
- `vector` — 384-dimensional embedding (Float32 array)
- `char_count` — number of characters in the chunk
- `token_count` — number of tokens according to tiktoken

### Cosine Similarity Search

When a user submits a query, the query text is embedded into a 384-dim vector. LanceDB then performs a cosine similarity search across all stored vectors, returning the top-k most similar chunks.

Cosine similarity measures the angle between two vectors rather than their absolute distance. This makes it robust to differences in text length — a short, dense chunk and a long, verbose chunk covering the same topic will still score highly against a matching query.
