import pandas as pd
from pathlib import Path
p = Path('data_files/espn_player_tournament_features.parquet')
df = pd.read_parquet(p)
print('sample date UTC:', df['date'].iloc[0])
display = df.head(1).copy()
display['date'] = pd.to_datetime(display['date'], utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M %Z')
print('converted to ET:', display['date'].iloc[0])
