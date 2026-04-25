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