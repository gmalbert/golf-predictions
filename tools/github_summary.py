"""Generate a short Markdown summary for GitHub Actions step summary.

Intended to be safe (never raise) and produce human-friendly output that
will be appended to $GITHUB_STEP_SUMMARY in the workflow.

Outputs (markdown):
- Latest OWGR week/year and counts
- Whether player IDs were attached
- Feature rebuild status (presence of espn_with_owgr_features.parquet)
- Small diagnostics when files are missing
"""
from pathlib import Path
import sys

try:
    import pandas as pd
except Exception:
    pd = None

ROOT = Path(__file__).resolve().parent.parent
OWGR_IDS = ROOT / 'data_files' / 'owgr_rankings_with_ids.parquet'
OWGR_RAW = ROOT / 'data_files' / 'owgr_rankings.parquet'
FEATURES = ROOT / 'data_files' / 'espn_with_owgr_features.parquet'
PLAYER_REG = ROOT / 'data_files' / 'player_registry.parquet'


def safe_read_parquet(p: Path):
    if not p.exists():
        return None
    if pd is None:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def fmt_k(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def main():
    print("### OWGR Update Report\n")

    # OWGR (with IDs)
    df_ids = safe_read_parquet(OWGR_IDS)
    if df_ids is not None:
        sy = int(df_ids['source_year'].max()) if 'source_year' in df_ids.columns else None
        sw = int(df_ids[df_ids['source_year'] == sy]['source_week'].max()) if sy is not None else None
        total = len(df_ids)
        unique_players = df_ids['player_id'].nunique() if 'player_id' in df_ids.columns else 'N/A'

        print(f"**OWGR (with IDs):** {fmt_k(total)} records · {fmt_k(unique_players)} unique players")
        if sy and sw:
            print(f"- Latest week/year: **Week {sw}, {sy}**")
    else:
        # Try raw OWGR
        df_raw = safe_read_parquet(OWGR_RAW)
        if df_raw is not None:
            total = len(df_raw)
            print(f"**OWGR (raw):** {fmt_k(total)} records (no player IDs attached)")
        else:
            print("**OWGR data:** Not found (no parquet files)")

    # Features
    df_feats = safe_read_parquet(FEATURES)
    if df_feats is not None:
        rows = len(df_feats)
        players = df_feats['player_id'].nunique() if 'player_id' in df_feats.columns else 'N/A'
        owgr_cov = None
        if 'owgr_rank_current' in df_feats.columns:
            try:
                owgr_cov = df_feats['owgr_rank_current'].notna().sum() / len(df_feats) * 100
            except Exception:
                owgr_cov = None

        print(f"\n**Features rebuilt:** {fmt_k(rows)} rows · Players: {fmt_k(players)}")
        if owgr_cov is not None:
            print(f"- OWGR coverage in features: **{owgr_cov:.1f}%**")
    else:
        print("\n**Features:** `espn_with_owgr_features.parquet` not found")

    # Player registry
    df_reg = safe_read_parquet(PLAYER_REG)
    if df_reg is not None:
        print(f"\n**Player registry:** {fmt_k(len(df_reg))} entries")
    else:
        print(f"\n**Player registry:** Not found (`player_registry.parquet`) or empty")

    # Suggest next steps if something missing
    missing = []
    if df_ids is None and safe_read_parquet(OWGR_RAW) is None:
        missing.append('OWGR parquet')
    if df_feats is None:
        missing.append('Rebuilt features')

    if missing:
        print('\n**Notes:**')
        for m in missing:
            print(f"- Missing: {m}")
        print('\n> If expected files are missing, check earlier workflow steps for download or parsing failures.')

    # Exit successfully
    return 0


if __name__ == '__main__':
    code = main()
    sys.exit(code)
