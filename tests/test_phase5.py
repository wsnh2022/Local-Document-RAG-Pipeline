from main import run_ingest

# First run — should ingest
print("=== RUN 1 (should ingest) ===")
run_ingest("docs")

# Second run — should skip
print("=== RUN 2 (should skip) ===")
run_ingest("docs")