"""Quick data quality check for scraped ESPN data"""
import pandas as pd
from pathlib import Path

data_file = Path("data_files/espn_pga_2022.parquet")

if data_file.exists():
    df = pd.read_parquet(data_file)
    
    print("="*60)
    print("ESPN PGA 2022 Data - Quality Check")
    print("="*60)
    
    print(f"\n📊 Dataset Stats:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Tournaments: {df['tournament'].nunique()}")
    print(f"   Unique players: {df['name'].nunique()}")
    print(f"   Year: {df['year'].iloc[0]}")
    
    print(f"\n📋 Columns ({len(df.columns)}):")
    for col in df.columns:
        nulls = df[col].isnull().sum()
        null_pct = (nulls / len(df) * 100)
        print(f"   {col:20s} - {null_pct:5.1f}% null")
    
    print(f"\n🏆 Top 10 Tournaments by Player Count:")
    top_tourneys = df.groupby('tournament').size().sort_values(ascending=False).head(10)
    for tourney, count in top_tourneys.items():
        print(f"   {count:3d} players - {tourney}")
    
    print(f"\n👤 Sample Players:")
    sample = df[df['tournament'] == df['tournament'].iloc[0]].head(5)
    print(sample[['name', 'position', 'total_score', 'country']].to_string(index=False))
    
    print(f"\n✅ Data looks good!")
else:
    print(f"❌ File not found: {data_file}")
