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