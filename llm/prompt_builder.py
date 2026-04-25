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