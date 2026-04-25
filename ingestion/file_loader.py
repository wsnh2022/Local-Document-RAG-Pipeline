import os
import fitz  # pymupdf
import docx
import markdown
import re
from pathlib import Path

EXCLUDED_DIRS = {".venv", "models", "data", "__pycache__", ".git"}

def get_all_files(folder_path: str) -> list[str]:
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        # Skip excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return sorted(files)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

def load_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    loaders = {
        ".txt": _load_txt,
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".md": _load_md,
    }
    return loaders[ext](file_path)

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _load_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

def _load_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def _load_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    html = markdown.markdown(raw)
    return re.sub(r"<[^>]+>", "", html)  # strip HTML tags

def get_all_files(folder_path: str) -> list[str]:
    files = []
    for root, _, filenames in os.walk(folder_path):
        for fname in filenames:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(os.path.join(root, fname))
    return sorted(files)

def load_file_safe(file_path: str) -> str | None:
    try:
        return load_file(file_path)
    except Exception as e:
        print(f"  [WARN] Could not load {os.path.basename(file_path)}: {e}")
        return None