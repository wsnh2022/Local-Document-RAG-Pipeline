from ingestion.chunker import chunk_file

chunks = chunk_file("docs/intro.txt")
print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"  id={c['id']}  tokens={c['token_count']}  text={c['chunk_text'][:60]}")