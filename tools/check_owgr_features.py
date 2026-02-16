import pandas as pd

df = pd.read_parquet('data_files/espn_with_owgr_features.parquet')
print(f'Rows: {len(df):,}')
owgr_cols = [c for c in df.columns if 'owgr' in c.lower()]
print(f'OWGR columns: {owgr_cols}')
print(f'\nSample:')
print(df[['name', 'tournament', 'year', 'owgr_rank_current', 'owgr_rank_change_4w', 'tournament_rank']].head(10))
print(f'\nCoverage:')
print(f"Records with OWGR rank: {df['owgr_rank_current'].notna().sum():,} / {len(df):,} ({df['owgr_rank_current'].notna().sum()/len(df)*100:.1f}%)")
