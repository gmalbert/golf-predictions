"""Check consolidated ESPN file"""
import pandas as pd

df = pd.read_parquet('data_files/espn_pga_2018_2025.parquet')

print(f"Consolidated file info:")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")
print(f"  Unique tournaments: {df['tournament'].nunique()}")
print(f"  Unique players: {df['name'].nunique()}")

if 'source_file' in df.columns:
    print(f"\n  Source files:")
    for f in sorted(df['source_file'].unique()):
        count = (df['source_file'] == f).sum()
        print(f"    {f}: {count:,} rows")

# Check if ALL PLAYERS have same scores in each tournament
print(f"\n\nChecking data integrity:")
tourn = '2018 Masters Tournament'
sample = df[df['tournament'] == tourn].copy()

print(f"\n{tourn}:")
print(f"  Total players: {len(sample)}")
print(f"  Unique scores: {sample['total_score'].nunique()}")
print(f"\n  Score distribution:")
print(sample['total_score'].value_counts().head(10).to_string())
