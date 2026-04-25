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