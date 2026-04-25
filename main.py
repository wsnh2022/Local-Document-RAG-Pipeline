import argparse
import os
from dotenv import load_dotenv

load_dotenv()

def run_ingest(folder_path: str, force_reingest: bool = False):
    from ingestion.file_loader import get_all_files
    from ingestion.chunker import chunk_file
    from ingestion.embedder import attach_embeddings
    from ingestion.hash_tracker import hash_file
    from storage.lance_store import (
        insert_chunks, hash_exists, delete_by_file_name, get_table_stats
    )

    files = get_all_files(folder_path)
    if not files:
        print(f"No supported files found in: {folder_path}")
        return

    print(f"\nFound {len(files)} file(s) to process\n")
    skipped, ingested = 0, 0

    for file_path in files:
        file_hash = hash_file(file_path)
        fname = os.path.basename(file_path)

        if hash_exists(file_hash) and not force_reingest:
            print(f"  [SKIP] {fname} — already ingested")
            skipped += 1
            continue

        if force_reingest:
            # delete by file_name so old chunks are removed even if content changed (different hash)
            deleted = delete_by_file_name(fname)
            if deleted:
                print(f"  [DELETE] {fname} — removed {deleted} old chunks")

        print(f"  [INGEST] {fname}")
        chunks = chunk_file(file_path)
        chunks = attach_embeddings(chunks)
        insert_chunks(chunks)
        ingested += 1

    stats = get_table_stats()
    print(f"\nDone. Ingested: {ingested} | Skipped: {skipped}")
    print(f"DB total: {stats['total_chunks']} chunks from {stats['unique_files']} files\n")

def run_query_loop():
    from retrieval.searcher import retrieve, fit_chunks_to_context
    from llm.openrouter_client import call_openrouter
    from llm.prompt_builder import build_rag_prompt, build_summary_prompt, needs_summary
    from storage.lance_store import get_table_stats

    stats = get_table_stats()
    if stats["total_chunks"] == 0:
        print("No data ingested yet. Run: python main.py --ingest <folder>")
        return

    print(f"\nReady. DB has {stats['total_chunks']} chunks from {stats['unique_files']} files.")
    print("Type your question (or 'exit' to quit)\n")

    while True:
        try:
            query = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye.")
            break

        print("\nSearching...\n")
        chunks = retrieve(query)
        chunks = fit_chunks_to_context(chunks)  # enforce context token budget

        if not chunks:
            print("No relevant content found in the database.\n")
            continue

        print("Top sources found:")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] {c['file_name']} — chunk {c['chunk_index']}")
        print()

        if needs_summary(query):
            messages = build_summary_prompt(query, chunks)
            print("[SUMMARY MODE]")
        else:
            messages = build_rag_prompt(query, chunks)
            print("[RAG MODE]")

        print("Calling OpenRouter...\n")
        try:
            answer = call_openrouter(messages)
            print("Answer:\n")
            print(answer)
            print("\n" + "─" * 60 + "\n")
        except Exception as e:
            print(f"LLM Error: {e}\n")

def run_delete(file_name: str = None):
    from storage.lance_store import list_ingested_files, delete_by_file_name

    files = list_ingested_files()
    if not files:
        print("\nNo ingested files found in DB.\n")
        return

    print("\nIngested files:")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f['file_name']} — {f['chunks']} chunks")

    # resolve targets: passed via CLI arg, or interactive prompt
    if file_name:
        targets = [f for f in files if f["file_name"] == file_name]
        if not targets:
            print(f"\nNo file named '{file_name}' found in DB.\n")
            return
    else:
        print()
        print("  Enter numbers to delete (e.g. 1,3,5  or  1-3  or  all)")
        raw = input("  Selection: ").strip()
        if not raw:
            print("Cancelled.\n")
            return

        indices = set()
        if raw.lower() == "all":
            indices = set(range(1, len(files) + 1))
        else:
            for part in raw.split(","):
                part = part.strip()
                if "-" in part:
                    try:
                        lo, hi = part.split("-", 1)
                        indices.update(range(int(lo), int(hi) + 1))
                    except ValueError:
                        print(f"  Invalid range: '{part}' — skipping")
                elif part.isdigit():
                    indices.add(int(part))
                else:
                    print(f"  Invalid entry: '{part}' — skipping")

        targets = []
        for idx in sorted(indices):
            if 1 <= idx <= len(files):
                targets.append(files[idx - 1])
            else:
                print(f"  [{idx}] out of range — skipping")

        if not targets:
            print("No valid files selected. Cancelled.\n")
            return

    print(f"\nFiles to delete ({len(targets)}):")
    total_chunks = 0
    for t in targets:
        print(f"  - {t['file_name']} ({t['chunks']} chunks)")
        total_chunks += t["chunks"]

    confirm = input(f"\nDelete {len(targets)} file(s) / {total_chunks} chunks? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.\n")
        return

    for t in targets:
        deleted = delete_by_file_name(t["file_name"])
        print(f"  Deleted '{t['file_name']}' — {deleted} chunks removed")
    print(f"\nDone. {len(targets)} file(s) deleted.\n")

def run_delete_from_file(list_path: str):
    from storage.lance_store import list_ingested_files, delete_by_file_name

    if not os.path.isfile(list_path):
        print(f"\nFile not found: {list_path}\n")
        return

    requested = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.split("#")[0].strip()
            if name:
                requested.append(name)

    if not requested:
        print(f"\n{list_path} is empty or has only comments. Nothing to delete.\n")
        return

    ingested = {f["file_name"]: f for f in list_ingested_files()}

    found, missing = [], []
    for name in requested:
        if name in ingested:
            found.append(ingested[name])
        else:
            missing.append(name)

    if missing:
        print(f"\nNot found in DB ({len(missing)}) — will be skipped:")
        for m in missing:
            print(f"  - {m}")

    if not found:
        print("\nNo matching files to delete.\n")
        return

    total_chunks = sum(f["chunks"] for f in found)
    print(f"\nFiles to delete ({len(found)}):")
    for f in found:
        print(f"  - {f['file_name']} ({f['chunks']} chunks)")

    confirm = input(f"\nDelete {len(found)} file(s) / {total_chunks} chunks? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.\n")
        return

    for f in found:
        deleted = delete_by_file_name(f["file_name"])
        print(f"  Deleted '{f['file_name']}' — {deleted} chunks removed")
    print(f"\nDone. {len(found)} file(s) deleted.\n")

def run_stats():
    from storage.lance_store import get_table_stats, get_or_create_table
    from ingestion.token_counter import chunks_token_report
    stats = get_table_stats()
    print(f"\nDB Stats:")
    print(f"  Total chunks : {stats['total_chunks']}")
    print(f"  Unique files : {stats['unique_files']}")
    if stats["total_chunks"] > 0:
        table = get_or_create_table()
        df = table.to_pandas()
        print("\nIngested files:")
        for fname in df["file_name"].unique():
            count = len(df[df["file_name"] == fname])
            print(f"  {fname} — {count} chunks")
        chunks = df.to_dict(orient="records")
        report = chunks_token_report(chunks)
        print(f"\nToken Report:")
        print(f"  Min tokens/chunk : {report['min_tokens']}")
        print(f"  Max tokens/chunk : {report['max_tokens']}")
        print(f"  Avg tokens/chunk : {report['avg_tokens']}")
        print(f"  Oversized chunks : {report['oversized']}  (>{os.getenv('MAX_CHUNK_TOKENS', 150)} tokens)")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanceDB RAG CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", metavar="FOLDER", help="Ingest documents from folder")
    group.add_argument("--query", action="store_true", help="Start interactive query loop")
    group.add_argument("--stats", action="store_true", help="Show database statistics")
    group.add_argument("--reingest", metavar="FOLDER", help="Force re-ingest (deletes old chunks)")
    group.add_argument("--delete", metavar="FILE_NAME", nargs="?", const="", help="Delete a specific ingested file by name")
    group.add_argument("--delete-from", metavar="LIST_FILE", help="Delete files listed in a text file (one name per line)")
    args = parser.parse_args()

    if args.ingest:
        run_ingest(args.ingest)
    elif args.reingest:
        run_ingest(args.reingest, force_reingest=True)
    elif args.query:
        run_query_loop()
    elif args.stats:
        run_stats()
    elif args.delete is not None:
        run_delete(args.delete if args.delete else None)
    elif args.delete_from:
        run_delete_from_file(args.delete_from)