import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        cache = os.getenv("MODEL_CACHE", "./models")
        _model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache)
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    return [v.tolist() for v in vectors]

def embed_query(text: str) -> list[float]:
    model = get_model()
    vector = model.encode([text])[0]
    return vector.tolist()

def attach_embeddings(chunks: list[dict]) -> list[dict]:
    texts = [c["chunk_text"] for c in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector
    return chunks