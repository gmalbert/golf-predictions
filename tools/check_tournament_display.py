import pandas as pd
import re

p = 'data_files/espn_with_owgr_features.parquet'
df = pd.read_parquet(p)
nty = df[['tournament','year']].drop_duplicates()
nty = nty[nty['tournament'].notna()]

from utils.tournament_display import format_tournament_display, tournament_sort_key

nty['display'] = nty['tournament'].apply(format_tournament_display)
# Create sort key that ignores leading articles (The/A/An)
nty['sort'] = nty['display'].apply(tournament_sort_key)
nty = nty.sort_values(['sort','year'], ascending=[True, False])
masters = [f"{r['display']} ({int(r['year'])})" for _, r in nty.iterrows() if 'masters' in r['display'].lower()]
memorials = [f"{r['display']} ({int(r['year'])})" for _, r in nty.iterrows() if 'memorial' in r['display'].lower()]
cj_entries = [f"{r['display']} ({int(r['year'])})" for _, r in nty.iterrows() if 'cj' in r['display'].lower()]
print('Sample Masters dropdown entries (first 8):')
for x in masters[:8]:
    print(' -', x)
print('\nSample Memorial dropdown entries (first 8):')
for x in memorials[:8]:
    print(' -', x)
print('\nSample CJ dropdown entries (first 8):')
for x in cj_entries[:8]:
    print(' -', x)

# Show a small slice of the sorted list around Memorial to prove sorting ignores leading articles
all_display = [f"{r['display']} ({int(r['year'])})" for _, r in nty.iterrows()]
idx = next((i for i,s in enumerate(all_display) if 'Memorial Tournament' in s), None)
print('\nSorted list around Memorial:')
for s in all_display[max(0, idx-5): (idx or 0)+6]:
    print(' -', s)
