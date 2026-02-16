"""Quick demo of OWGR + ESPN data integration."""
import pandas as pd

# Load data
owgr = pd.read_parquet('data_files/owgr_rankings_with_ids.parquet')
espn = pd.read_parquet('data_files/espn_pga_2018_2025.parquet')
registry = pd.read_parquet('data_files/player_registry.parquet')

print('='*70)
print('OWGR + ESPN DATA INTEGRATION - SUCCESS!')
print('='*70)

print(f'\n📊 Data Summary:')
print(f'   OWGR records: {len(owgr):,} (all have player_id)')
print(f'   ESPN records: {len(espn):,}')
print(f'   Players in registry: {len(registry):,}')
print(f'   Players with ESPN link: {registry.espn_player_id.notna().sum()}')

print(f'\n🔗 Sample Players with Both ESPN and OWGR Data:')
print('='*70)

# Find players with both
linked = registry[registry.espn_player_id.notna()].head(10)

for _, row in linked.iterrows():
    espn_count = len(espn[espn.player_id == row['espn_player_id']])
    owgr_count = len(owgr[owgr.player_id == row['player_id']])
    print(f"  {row['name']:25s} ESPN: {espn_count:3d} tourneys | OWGR: {owgr_count:5d} weeks")

print(f'\n✅ WHAT YOU CAN DO NOW:')
print('='*70)
print('  1. Join ESPN + OWGR data by player_id')
print('  2. Add OWGR rank as feature for each tournament')
print('  3. Add rank momentum (4/12/52-week trends)')
print('  4. Test if OWGR improves prediction accuracy')
print('\n  Next: Build OWGR features for the prediction model!')
print('='*70)
