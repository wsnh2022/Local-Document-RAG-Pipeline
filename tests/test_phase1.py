from storage.lance_store import get_or_create_table, get_table_stats

table = get_or_create_table()
print("Table created:", table)
print("Stats:", get_table_stats())