from retrieval.searcher import retrieve, format_results, fit_chunks_to_context

# Test retrieval
query = "what is vector database"
print(f"Query: {query}\n")

results = retrieve(query)
print(f"Raw results: {len(results)} chunks found")

# Test context budget enforcement
fitted = fit_chunks_to_context(results)
print(f"After context fit: {len(fitted)} chunks")

# Test formatted output
print("\nFormatted results:")
print(format_results(fitted))