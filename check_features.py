"""Compare feature values for elite vs mid-tier players."""
import pandas as pd
from pathlib import Path

df = pd.read_parquet(Path('data_files/espn_with_owgr_features.parquet'))
latest = df.sort_values('date').groupby('player_id').tail(1)

jt = latest[latest['name'] == 'Justin Thomas']
ct = latest[latest['name'] == 'Cameron Tringale']

print('Justin Thomas (OWGR #5):')
if len(jt) > 0:
    print(f"  owgr_points_current: {jt['owgr_points_current'].values[0] if 'owgr_points_current' in jt.columns else 'N/A'}")
    print(f"  owgr_rank_current: {jt['owgr_rank_current'].values[0] if 'owgr_rank_current' in jt.columns else 'N/A'}")
    print(f"  prior_avg_score: {jt['prior_avg_score'].values[0] if 'prior_avg_score' in jt.columns else 'N/A'}")
    print(f"  last_event_rank: {jt['last_event_rank'].values[0] if 'last_event_rank' in jt.columns else 'N/A'}")
    print(f"  prior_top10_rate_10: {jt['prior_top10_rate_10'].values[0] if 'prior_top10_rate_10' in jt.columns else 'N/A'}")

print('\nCameron Tringale (OWGR #53):')
if len(ct) > 0:
    print(f"  owgr_points_current: {ct['owgr_points_current'].values[0] if 'owgr_points_current' in ct.columns else 'N/A'}")
    print(f"  owgr_rank_current: {ct['owgr_rank_current'].values[0] if 'owgr_rank_current' in ct.columns else 'N/A'}")
    print(f"  prior_avg_score: {ct['prior_avg_score'].values[0] if 'prior_avg_score' in ct.columns else 'N/A'}")
    print(f"  last_event_rank: {ct['last_event_rank'].values[0] if 'last_event_rank' in ct.columns else 'N/A'}")
    print(f"  prior_top10_rate_10: {ct['prior_top10_rate_10'].values[0] if 'prior_top10_rate_10' in ct.columns else 'N/A'}")

print('\n--- Analysis ---')
if len(jt) > 0 and len(ct) > 0:
    jt_points = jt['owgr_points_current'].values[0]
    ct_points = ct['owgr_points_current'].values[0]
    print(f"JT has {jt_points/ct_points:.2f}x more OWGR points than CT")
    print(f"But JT win prob (5.16%) is only {5.16/5.51:.2f}x CT's win prob (5.51%)")
    print("\n✗ Model is not properly weighting OWGR points despite it being top feature!")
