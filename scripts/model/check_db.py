import sqlite3
from pathlib import Path

db_path = r'C:\Users\ykovalova\msbuild\data\msbuild_analytics.db'

# Check if database exists
if not Path(db_path).exists():
    print(f"Database not found at: {db_path}")
    exit(1)

print(f"Checking database: {db_path}")
print("=" * 50)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"\nTables found: {tables}")
    
    # Check each table
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"\n{table}: {count} records")
        
        # Show columns
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        print(f"  Columns: {', '.join(columns)}")
        
        # Show sample record
        if count > 0:
            cursor.execute(f"SELECT * FROM {table} LIMIT 1")
            sample = cursor.fetchone()
            print(f"  Sample record ID: {sample[0] if sample else 'None'}")
    
    conn.close()
    print("\nDatabase check complete!")
    
except Exception as e:
    print(f"Error: {e}")