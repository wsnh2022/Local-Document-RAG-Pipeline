import os
import lancedb
from storage.schema import DocumentChunk
from dotenv import load_dotenv

load_dotenv()

TABLE_NAME = "documents"

def get_db():
    path = os.getenv("LANCEDB_PATH", "./data/lancedb")
    os.makedirs(path, exist_ok=True)
    return lancedb.connect(path)

def get_or_create_table():
    db = get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=DocumentChunk)

def insert_chunks(chunks: list[dict]):
    table = get_or_create_table()
    table.add(chunks)
    print(f"  Inserted {len(chunks)} chunks into LanceDB")

def delete_by_file_hash(file_hash: str):
    table = get_or_create_table()
    safe_hash = file_hash.replace("'", "''")
    table.delete(f"file_hash = '{safe_hash}'")
    print(f"  Deleted chunks for file_hash={file_hash}")

def delete_by_file_name(file_name: str) -> int:
    table = get_or_create_table()
    df = table.to_pandas()
    matched = df[df["file_name"] == file_name]
    count = len(matched)
    if count == 0:
        return 0
    safe_name = file_name.replace("'", "''")
    table.delete(f"file_name = '{safe_name}'")
    return count

def list_ingested_files() -> list[dict]:
    table = get_or_create_table()
    df = table.to_pandas()
    if df.empty:
        return []
    return [
        {"file_name": fname, "chunks": len(df[df["file_name"] == fname])}
        for fname in df["file_name"].unique()
    ]

def hash_exists(file_hash: str) -> bool:
    table = get_or_create_table()
    safe_hash = file_hash.replace("'", "''")
    results = table.search().where(f"file_hash = '{safe_hash}'").limit(1).to_list()
    return len(results) > 0

def search_chunks(query_vector: list[float], top_k: int = 5) -> list[dict]:
    table = get_or_create_table()
    results = (
        table.search(query_vector)
        .limit(top_k)
        .to_pandas()
    )
    return results.to_dict(orient="records")

def get_table_stats() -> dict:
    table = get_or_create_table()
    df = table.to_pandas()
    return {
        "total_chunks": len(df),
        "unique_files": df["file_name"].nunique() if len(df) > 0 else 0
    }