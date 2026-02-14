"""
Fairway Oracle - PGA Tournament Predictions
Predict winners of upcoming PGA tournaments for betting insights.
"""

import streamlit as st
from pathlib import Path

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fairway Oracle – PGA Predictions",
    page_icon="⛳",
    layout="wide",
)

# ── Logo ─────────────────────────────────────────────────────────────────────
logo_path = Path(__file__).parent / "data_files" / "logo.png"
if logo_path.exists():
    st.image(str(logo_path), width=200)
else:
    st.warning("Logo not found – expected at data_files/logo.png")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("Fairway Oracle")
st.markdown("### ⛳ PGA Tournament Winner Predictions for Smarter Betting")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
tournament = st.sidebar.selectbox(
    "Select Tournament",
    [
        "The Masters",
        "PGA Championship",
        "U.S. Open",
        "The Open Championship",
        "The Players Championship",
        "Arnold Palmer Invitational",
        "Memorial Tournament",
        "WM Phoenix Open",
        "Genesis Invitational",
        "RBC Heritage",
    ],
)

num_predictions = st.sidebar.slider(
    "Top-N Predictions", min_value=5, max_value=50, value=20, step=5
)

# ── Main Content ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🏌️ Predictions – {tournament}")
    st.info(
        "Model not yet trained. Connect historical data and train a model to "
        "see predictions here. See the **docs/** roadmap for next steps."
    )

    # Placeholder for future predictions table
    # Example structure once the model is ready:
    # predictions_df = model.predict(tournament)
    # st.dataframe(predictions_df.head(num_predictions))

with col2:
    st.subheader("📊 Model Confidence")
    st.info(
        "Confidence metrics will appear here once a model is trained."
    )

# ── Upcoming Tournaments ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 Upcoming PGA Tour Events")
st.markdown(
    """
| Date | Tournament | Course | Purse |
|------|-----------|--------|-------|
| TBD  | *Connect live schedule data* | — | — |

> **Next step:** Scrape the PGA Tour schedule and populate this table
> automatically. See `docs/02_data_sources.md` for details.
"""
)

# ── Recent Form / Stats & Feature Preview ───────────────────────────────────
st.markdown("---")
st.subheader("📈 Player Stats & Features")

# Load features dataset if available and show quick interactive preview
import pandas as pd
features_path = Path(__file__).parent / "data_files" / "espn_player_tournament_features.parquet"

if features_path.exists():
    # Show feature dataset and interactive filters on the main page (not inside an expander)
    df_feats = pd.read_parquet(features_path)
    years_list = sorted(df_feats['year'].astype(int).unique().tolist())
    st.write(f"**Features:** {len(df_feats):,} rows · Years: {years_list} · Players: {df_feats['player_id'].nunique():,}")

    # Filters
    fcol1, fcol2 = st.columns([1, 3])
    with fcol1:
        year_opts = ["All"] + [str(y) for y in years_list]
        sel_year = st.selectbox("Year", year_opts, index=len(year_opts)-1)

        if sel_year != "All":
            tourn_opts = ["All"] + sorted(df_feats[df_feats['year'] == int(sel_year)]['tournament'].unique().tolist())
        else:
            tourn_opts = ["All"] + sorted(df_feats['tournament'].unique().tolist())
        sel_tourn = st.selectbox("Tournament", tourn_opts)

        player_sample = sorted(df_feats['name'].unique().tolist())[:200]
        sel_player = st.selectbox("Player (optional)", ["All"] + player_sample)

    with fcol2:
        # Apply filters
        view = df_feats.copy()
        if sel_year != "All":
            view = view[view['year'] == int(sel_year)]
        if sel_tourn != "All":
            view = view[view['tournament'] == sel_tourn]
        if sel_player != "All":
            view = view[view['name'] == sel_player]

        # Prepare display-only DataFrame: drop internal IDs and source columns
        view_display = view.copy()
        drop_cols = [c for c in ['player_id', 'tournament_id', 'source_file'] if c in view_display.columns]
        if drop_cols:
            view_display = view_display.drop(columns=drop_cols)

        # Convert UTC datetimes to Eastern Time for display
        if 'date' in view_display.columns:
            view_display['date'] = pd.to_datetime(view_display['date'], utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M %Z')

        # Friendly column names (only rename columns that exist)
        display_names = {
            'name': 'Player',
            'country': 'Country',
            'date': 'Date (ET)',
            'tournament': 'Tournament',
            'year': 'Year',
            'numeric_total_score': 'Total Score (to Par)',
            'tournament_rank': 'Tournament Rank',
            'prior_count': 'Previous Tournaments',
            'prior_avg_score': 'Prior Avg Score',
            'prior_std_score': 'Prior Score StdDev',
            'prior_avg_score_5': 'Avg Score (Last 5)',
            'prior_avg_score_10': 'Avg Score (Last 10)',
            'prior_std_score_5': 'StdDev (Last 5)',
            'prior_std_score_10': 'StdDev (Last 10)',
            'prior_top10_rate_5': 'Top-10 % (Last 5)',
            'prior_top10_rate_10': 'Top-10 % (Last 10)',
            'last_event_score': 'Last Event Score',
            'last_event_rank': 'Last Event Rank',
            'days_since_last_event': 'Days Since Last Event',
            'tournaments_last_365d': 'Tournaments (Last 365d)',
            'season_to_date_avg_score': 'Season-to-date Avg Score',
            'course_history_avg_score': 'Course Avg (Prior)',
            'career_best_rank': 'Career Best Rank',
            'played_last_30d': 'Played (Last 30d)'
        }
        rename_map = {k: v for k, v in display_names.items() if k in view_display.columns}
        view_display = view_display.rename(columns=rename_map)

        # Show filtered table without index
        st.dataframe(view_display.head(200), hide_index=True, use_container_width=True)

    # Quick aggregate leaderboards (rename for display)
    st.markdown("**Quick leaderboards (all years)**")
    try:
        top_form = df_feats.dropna(subset=['prior_avg_score']).nsmallest(10, 'prior_avg_score')[['name','prior_avg_score','prior_count']].drop_duplicates()
        top_form = top_form.rename(columns={
            'name': 'Player',
            'prior_avg_score': 'Prior Avg Score',
            'prior_count': 'Previous Tournaments'
        })
        st.dataframe(top_form, hide_index=True)
    except Exception:
        st.write("No form leaderboard available yet.")
else:
    st.info("Features dataset not found. Run `python features/build_features.py` to create `espn_player_tournament_features.parquet`.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Fairway Oracle © 2026")
