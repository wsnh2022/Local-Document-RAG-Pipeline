import hashlib

def hash_file(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()

def make_chunk_id(file_hash: str, chunk_index: int) -> str:
    return f"{file_hash[:16]}_{chunk_index:04d}"