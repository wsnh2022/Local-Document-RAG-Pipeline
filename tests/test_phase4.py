from ingestion.embedder import embed_query, attach_embeddings
from ingestion.chunker import chunk_file

# Test single query embedding
v = embed_query("What is LanceDB?")
print("Vector dim:", len(v))
print("First 5 values:", [round(x, 4) for x in v[:5]])

# Test attaching embeddings to chunks
chunks = chunk_file("docs/intro.txt")
chunks = attach_embeddings(chunks)
print("Chunk has vector:", "vector" in chunks[0])
print("Vector dim on chunk:", len(chunks[0]["vector"]))