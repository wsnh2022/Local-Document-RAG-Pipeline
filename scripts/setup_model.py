# run_once_download_model.py — run this once in Phase 0
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./models")
print("Model downloaded and cached at ./models")