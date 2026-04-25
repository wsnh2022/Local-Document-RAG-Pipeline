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