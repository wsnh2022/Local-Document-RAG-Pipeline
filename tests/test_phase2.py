from ingestion.file_loader import load_file, get_all_files

# Test single file load
text = load_file("test_sample.txt")
print("Loaded text:", text[:100])

# Test folder scan — point to a real docs folder, not "."
files = get_all_files("docs")  # create a docs/ folder with sample files
print("Files found:", files)