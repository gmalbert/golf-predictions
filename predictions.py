"""
Fairway Oracle - PGA Tournament Predictions
Predict winners of upcoming PGA tournaments for betting insights.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from utils.tournament_display import format_tournament_display, tournament_sort_key


# ── Tournament name → odds event_label mapping ───────────────────────────────
# Keys are substrings that appear in ESPN tournament names (case-insensitive)
TOURNAMENT_TO_EVENT_LABEL = {
    "masters":        "masters",
    "pga championship": "pga_champ",
    "u.s. open":      "us_open",
    "us open":        "us_open",
    "the open":       "open",
    "open championship": "open",
}


def resolve_event_label(tournament_name: str) -> str | None:
    """Return the odds event_label for a tournament name, or None if not a major."""
    if not tournament_name:
        return None
    name_lower = tournament_name.lower()
    for key, label in TOURNAMENT_TO_EVENT_LABEL.items():
        if key in name_lower:
            return label
    return None


def enrich_predictions_with_odds(
    predictions: pd.DataFrame,
    tournament_name: str,
) -> pd.DataFrame:
    """
    Merge consensus market odds into a predictions DataFrame and add:
      - DK Odds (american)
      - Best Odds / Best Book
      - Mkt NoVig%
      - Edge (pp)   (model_win% - market novig%)
      - Value Bet   ("YES +Xpp" / "no (-Xpp)")

    Supports both schema versions:
      - New (RotoWire): columns include player_name, event_name, event_id
      - Old (Odds API): columns include player, event_label

    Matches records by tournament name substring (case-insensitive).
    Returns the enriched DataFrame.
    """
    odds_path = Path(__file__).parent / "data_files" / "odds_consensus_latest.parquet"
    if not odds_path.exists():
        return predictions

    try:
        odds = pd.read_parquet(odds_path)
        cols = set(odds.columns)

        # ── Normalise to common schema ──────────────────────────────────────
        # New RotoWire schema
        if "player_name" in cols and "event_name" in cols:
            # Filter to matching event by tournament name substring
            t_lower = (tournament_name or "").lower()
            mask = odds["event_name"].str.lower().apply(
                lambda en: any(part in en for part in t_lower.split() if len(part) > 3)
                or any(part in t_lower for part in en.lower().split() if len(part) > 3)
            )
            odds_filtered = odds[mask].copy()
            if odds_filtered.empty:
                # No event match — use whatever is in the file (single-event file)
                odds_filtered = odds.copy()
            odds_filtered = odds_filtered.rename(columns={"player_name": "player"})

        # Old Odds API schema
        elif "player" in cols and "event_label" in cols:
            event_label = resolve_event_label(tournament_name)
            if not event_label:
                return predictions
            odds_filtered = odds[odds["event_label"] == event_label].copy()
            if odds_filtered.empty:
                return predictions
        else:
            return predictions

        if odds_filtered.empty:
            return predictions

        # Normalize `best_book` values to user-friendly names (e.g. 'betrivers' -> 'BetRivers')
        def _normalize_book(name):
            if pd.isna(name):
                return name
            key = str(name).strip().lower()
            mapping = {
                'mgm': 'BetMGM', 'betmgm': 'BetMGM', 'bet_mgm': 'BetMGM',
                'betrivers': 'BetRivers', 'bet_rivers': 'BetRivers',
                'draftkings': 'DraftKings', 'draft_kings': 'DraftKings',
                'fanduel': 'FanDuel', 'fan_duel': 'FanDuel',
                'caesars': 'Caesars',
                'hardrock': 'Hard Rock',
                'thescore': 'TheScore', 'the_score': 'TheScore', 'score': 'TheScore'
            }
            return mapping.get(key, str(name).strip().title())

        if 'best_book' in odds_filtered.columns:
            odds_filtered['best_book'] = odds_filtered['best_book'].apply(_normalize_book)

        odds_filtered["player_key"] = odds_filtered["player"].str.strip().str.lower()

        # Ensure needed columns exist
        for col in ["dk_odds", "best_odds", "best_book", "avg_novig_prob"]:
            if col not in odds_filtered.columns:
                odds_filtered[col] = None

        preds = predictions.copy()
        preds["player_key"] = preds["name"].str.strip().str.lower()

        merge_cols = ["player_key", "dk_odds", "best_odds", "best_book", "avg_novig_prob"]
        merge_cols = [c for c in merge_cols if c in odds_filtered.columns]

        merged = preds.merge(
            odds_filtered[merge_cols],
            on="player_key",
            how="left",
        ).drop(columns=["player_key"])

        merged["edge_pp"] = (
            merged["win_probability"] - merged["avg_novig_prob"]
        ) * 100

        def value_label(row):
            if pd.isna(row.get("avg_novig_prob")):
                return "–"
            if row["edge_pp"] > 0:
                return f"YES +{row['edge_pp']:.1f}pp"
            return f"no ({row['edge_pp']:.1f}pp)"

        merged["Value Bet"] = merged.apply(value_label, axis=1)
        return merged
    except Exception:
        return predictions


# ── Helper Functions ─────────────────────────────────────────────────────────
def get_tournament_options(features_path: Path | None = None):
    """Return (options_list, mapping) for the tournament selectbox.

    This is CI-friendly (callable from tests) and mirrors the UI logic.
    """
    defaults = [
        "Masters Tournament (2025)",
        "PGA Championship (2025)",
        "U.S. Open (2025)",
        "The Open (2025)",
        "THE PLAYERS Championship (2025)",
    ]

    if features_path is None:
        for candidate in [
            Path(__file__).parent / "data_files" / "espn_with_extended_features.parquet",
            Path(__file__).parent / "data_files" / "espn_with_owgr_features.parquet",
            Path(__file__).parent / "data_files" / "espn_player_tournament_features.parquet",
        ]:
            if candidate.exists():
                features_path = candidate
                break

    if features_path is None or not features_path.exists():
        return defaults, {}

    df_temp = pd.read_parquet(features_path)
    tournament_years = df_temp[['tournament', 'year']].drop_duplicates()
    tournament_years = tournament_years[tournament_years['tournament'].notna()]

    # Use canonical formatter + sort-key helper
    tournament_years['display_name'] = tournament_years['tournament'].apply(format_tournament_display)
    tournament_years['sort_name'] = tournament_years['display_name'].apply(tournament_sort_key)

    tournament_years = tournament_years.sort_values(['sort_name', 'year'], ascending=[True, False])

    options = [f"{row['display_name']} ({int(row['year'])})" for _, row in tournament_years.iterrows()]
    mapping = {f"{row['display_name']} ({int(row['year'])})": (row['tournament'], int(row['year'])) for _, row in tournament_years.iterrows()}

    return options, mapping


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
# st.title("Fairway Oracle")
# st.markdown("### ⛳ PGA Tournament Winner Predictions for Smarter Betting")
# st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")

# Mode selector: Historical vs Upcoming
prediction_mode = st.sidebar.radio(
    "Prediction Mode",
    ["📊 Historical Tournaments", "🔮 Upcoming Tournaments"],
    index=0
)

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

# Mode-specific variables
tournament = None
is_upcoming = False
selected_year = None
tournament_display = None

# Mode-specific tournament selection
if prediction_mode == "🔮 Upcoming Tournaments":
    # Fetch upcoming tournaments
    try:
        from models.predict_upcoming import get_upcoming_tournaments
        with st.spinner("Fetching upcoming tournaments..."):
            upcoming_df = get_upcoming_tournaments(days_ahead=90)
        
        if len(upcoming_df) > 0:
            # Format as selectbox options — tag majors with an odds indicator
            upcoming_options = []
            upcoming_mapping = {}
            
            for _, row in upcoming_df.iterrows():
                date_str = row['date'].strftime('%b %d, %Y')
                has_odds_tag = " 💰" if resolve_event_label(row['name']) else ""
                display = f"{row['name']}{has_odds_tag} - {date_str}"
                upcoming_options.append(display)
                upcoming_mapping[display] = {
                    'name': row['name'],
                    'id': row['id'],
                    'date': row['date']
                }
            
            selected_upcoming = st.sidebar.selectbox(
                "Select Upcoming Tournament",
                upcoming_options,
            )
            st.sidebar.caption("💰 = DraftKings odds available")
            
            # Extract selected tournament info
            selected_info = upcoming_mapping[selected_upcoming]
            tournament = selected_info['name']
            tournament_id = selected_info['id']
            tournament_date = selected_info['date']
            is_upcoming = True
            tournament_display = selected_upcoming
        else:
            st.sidebar.warning("No upcoming tournaments found in the next 90 days")

    except Exception as e:
        st.sidebar.error(f"Could not fetch upcoming tournaments: {e}")

else:
    # Historical tournament browser - load tournament list from actual data
    # Populate tournament options (callable from tests)
    try:
        tournament_options, tournament_mapping = get_tournament_options()
    except Exception:
        tournament_options = [
            "Masters Tournament (2025)",
            "PGA Championship (2025)", 
            "U.S. Open (2025)",
            "The Open (2025)",
            "THE PLAYERS Championship (2025)",
        ]
        tournament_mapping = {}

    tournament_display = st.sidebar.selectbox(
        "Select Tournament",
        tournament_options,
    )

    # Extract tournament name and year from selected option
    if tournament_display in tournament_mapping:
        tournament, selected_year = tournament_mapping[tournament_display]
    else:
        # Fallback parsing for default options
        import re
        match = re.match(r"^(.+?)\s*\((\d{4})\)$", tournament_display)
        if match:
            tournament = match.group(1)
            selected_year = int(match.group(2))
        else:
            tournament = tournament_display
            selected_year = 2025

num_predictions = st.sidebar.slider(
    "Top-N Predictions", min_value=5, max_value=50, value=20, step=5
)

# ── Main Content ─────────────────────────────────────────────────────────────
# Define model path (needed for both prediction and confidence display)
model_path = Path(__file__).parent / "models" / "saved_models" / "winner_predictor_v2.joblib"

st.subheader(f"🏌️ Predictions – {tournament_display}")

if is_upcoming:
    # Handle upcoming tournament predictions
    try:
        from models.predict_upcoming import predict_upcoming_tournament
        
        with st.spinner("Building predictions for upcoming tournament..."):
            predictions = predict_upcoming_tournament(tournament, tournament_id, tournament_date)
        
        if not predictions.empty:
            # Try to enrich with market odds (works for any PGA Tour event via RotoWire)
            predictions = enrich_predictions_with_odds(predictions, tournament)

            has_odds = "avg_novig_prob" in predictions.columns and predictions["avg_novig_prob"].notna().any()
            has_dk = has_odds and "dk_odds" in predictions.columns and predictions["dk_odds"].notna().any()

            # ── Refresh Odds button ────────────────────────────────────
            if not has_odds:
                refresh_col, info_col = st.columns([1, 3])
                with refresh_col:
                    if st.button("Refresh Odds", key="refresh_odds_upcoming"):
                        try:
                            import subprocess, sys
                            with st.spinner("Fetching latest odds from RotoWire..."):
                                result = subprocess.run(
                                    [sys.executable, "scrapers/rotowire_odds.py", "--no-save"],
                                    capture_output=True, text=True, timeout=20
                                )
                                # Re-run the scraper to save
                                subprocess.run(
                                    [sys.executable, "scrapers/rotowire_odds.py"],
                                    capture_output=True, text=True, timeout=20
                                )
                            st.success("Odds refreshed – reload page to see updated values.")
                            st.rerun()
                        except Exception as refresh_err:
                            st.error(f"Could not refresh odds: {refresh_err}")
                with info_col:
                    st.info(
                        "Market odds not loaded yet for this event. "
                        "Click **Refresh Odds** to pull the latest lines from RotoWire "
                        "(works for any current PGA Tour event).",
                        icon="\u2139\ufe0f",
                    )

            # Build display columns
            display_cols = ['name', 'win_probability']
            col_renames = {'name': 'Player', 'win_probability': 'Win Prob'}

            if 'owgr_rank_current' in predictions.columns:
                display_cols.append('owgr_rank_current')
                col_renames['owgr_rank_current'] = 'OWGR'

            if has_odds:
                if has_dk:
                    display_cols.append('dk_odds')
                    col_renames['dk_odds'] = 'DK Odds'
                display_cols += ['best_odds', 'best_book', 'avg_novig_prob', 'Value Bet']
                col_renames.update({
                    'best_odds':     'Best Odds',
                    'best_book':     'Best Book',
                    'avg_novig_prob': 'Mkt NoVig%',
                })

            # Respect Top‑N slider for upcoming predictions (don't show entire field)
            rows_to_show = predictions[[c for c in display_cols if c in predictions.columns]].head(num_predictions).copy()
            pred_display = rows_to_show.rename(columns=col_renames)

            # Add predicted rank (1 = top predicted winner, 2 = second, ...)
            pred_display.insert(0, 'Rank', range(1, len(pred_display) + 1))

            pred_display['Win Prob'] = pred_display['Win Prob'].apply(lambda x: f"{x*100:.2f}%")
            if 'OWGR' in pred_display.columns:
                pred_display['OWGR'] = pred_display['OWGR'].apply(
                    lambda x: str(int(x)) if pd.notna(x) else 'N/A'
                )
            if has_odds:
                def fmt_odds(x):
                    if pd.isna(x):
                        return 'N/A'
                    return f"+{int(x)}" if int(x) >= 0 else str(int(x))
                if has_dk and 'DK Odds' in pred_display.columns:
                    pred_display['DK Odds'] = pred_display['DK Odds'].apply(fmt_odds)
                if 'Best Odds' in pred_display.columns:
                    pred_display['Best Odds'] = pred_display['Best Odds'].apply(fmt_odds)
                pred_display['Mkt NoVig%'] = pred_display['Mkt NoVig%'].apply(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A'
                )

            tbl_height = min(len(pred_display) * 35 + 40, 2200)
            st.dataframe(pred_display, height=tbl_height, hide_index=True)

            total_field = len(predictions)
            top_n_prob = predictions['win_probability'].head(num_predictions).sum()
            if has_odds:
                books_str = "DraftKings" if has_dk else "BetMGM/BetRivers"
                st.caption(f"Top {num_predictions} of {total_field} players ({top_n_prob*100:.1f}% of win prob) | Odds: {books_str} via RotoWire")
                st.caption(
                    "**Value Bet** — `YES +Xpp` means the model's win probability exceeds the market no‑vig implied probability by X percentage points (positive edge). "
                    "`no (‑Xpp)` means the market is priced stronger than the model. Use this as a quick value indicator."
                )
            else:
                st.caption(f"Top {num_predictions} of {total_field} players ({top_n_prob*100:.1f}% of win prob)")
        else:
            st.warning(f"No field data available for {tournament}")
            
    except Exception as e:
        st.error(f"Could not generate predictions: {e}")

else:
    # Handle historical tournament predictions
    if model_path.exists():
        try:
            import joblib
            from models.predict_tournament import predict_field
            
            # Load features for selected tournament (prefer extended, then OWGR, then base)
            for fp_candidate in [
                Path(__file__).parent / "data_files" / "espn_with_extended_features.parquet",
                Path(__file__).parent / "data_files" / "espn_with_owgr_features.parquet",
                Path(__file__).parent / "data_files" / "espn_player_tournament_features.parquet",
            ]:
                if fp_candidate.exists():
                    features_path = fp_candidate
                    break
            
            if features_path.exists():
                df_all = pd.read_parquet(features_path)
                
                # Filter by selected tournament and year
                tournament_data = df_all[
                    (df_all['tournament'] == tournament) & 
                    (df_all['year'] == selected_year)
                ]
                
                if not tournament_data.empty:
                    field = tournament_data.copy()

                    # Make predictions
                    predictions = predict_field(field)

                    # Try to enrich with market odds
                    predictions = enrich_predictions_with_odds(predictions, tournament)

                    has_odds = "avg_novig_prob" in predictions.columns and predictions["avg_novig_prob"].notna().any()
                    has_dk = has_odds and "dk_odds" in predictions.columns and predictions["dk_odds"].notna().any()

                    # Build display columns
                    display_cols = ['name', 'win_probability']
                    col_renames = {'name': 'Player', 'win_probability': 'Win Prob'}

                    if 'owgr_rank_current' in predictions.columns:
                        display_cols.append('owgr_rank_current')
                        col_renames['owgr_rank_current'] = 'OWGR'

                    if 'tournament_rank' in predictions.columns:
                        display_cols.append('tournament_rank')
                        col_renames['tournament_rank'] = 'Actual Finish'

                    if has_odds:
                        if has_dk:
                            display_cols.append('dk_odds')
                            col_renames['dk_odds'] = 'DK Odds'
                        display_cols += ['best_odds', 'best_book', 'avg_novig_prob', 'Value Bet']
                        col_renames.update({
                            'best_odds':     'Best Odds',
                            'best_book':     'Best Book',
                            'avg_novig_prob': 'Mkt NoVig%',
                        })

                    # When odds are loaded, show full field; otherwise respect the slider
                    if has_odds:
                        hist_rows = predictions[[c for c in display_cols if c in predictions.columns]].copy()
                    else:
                        hist_rows = predictions[[c for c in display_cols if c in predictions.columns]].head(num_predictions).copy()
                    pred_display = hist_rows.rename(columns=col_renames)

                    # Add predicted rank (1 = top predicted winner)
                    pred_display.insert(0, 'Rank', range(1, len(pred_display) + 1))

                    pred_display['Win Prob'] = pred_display['Win Prob'].apply(lambda x: f"{x:.2%}")
                    if 'OWGR' in pred_display.columns:
                        pred_display['OWGR'] = pred_display['OWGR'].apply(
                            lambda x: str(int(x)) if pd.notna(x) else 'N/A'
                        )
                    if has_odds:
                        def fmt_odds_hist(x):
                            if pd.isna(x): return 'N/A'
                            return f"+{int(x)}" if int(x) >= 0 else str(int(x))
                        if has_dk and 'DK Odds' in pred_display.columns:
                            pred_display['DK Odds'] = pred_display['DK Odds'].apply(fmt_odds_hist)
                        if 'Best Odds' in pred_display.columns:
                            pred_display['Best Odds'] = pred_display['Best Odds'].apply(fmt_odds_hist)
                        pred_display['Mkt NoVig%'] = pred_display['Mkt NoVig%'].apply(
                            lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A'
                        )

                    tbl_height_hist = min(len(pred_display) * 35 + 40, 2200)
                    st.dataframe(pred_display, hide_index=True, height=tbl_height_hist)

                    total_field = len(predictions)
                    if has_odds:
                        shown = len(pred_display)
                        st.caption(f"Showing all {shown} players with odds | Odds: BetMGM/BetRivers via RotoWire")
                        st.caption(
                            "**Value Bet** — `YES +Xpp` means the model's win probability exceeds the market no‑vig implied probability by X percentage points (positive edge). "
                            "`no (‑Xpp)` means the market is priced stronger than the model. Use this as a quick value indicator."
                        )
                    else:
                        top_n_prob = predictions['win_probability'].head(num_predictions).sum()
                        st.caption(f"Top {num_predictions} of {total_field} players ({top_n_prob*100:.1f}% of win prob)")
                    st.caption(f"Based on {tournament} {selected_year} field")

                    if 'owgr_rank_current' not in predictions.columns:
                        st.caption("OWGR features not present — run `python features/build_owgr_features.py` to add.")
                else:
                    st.warning(f"No historical data available for {tournament} ({selected_year})")
            else:
                st.warning("Feature data not found. Please build features first.")
                
        except Exception as e:
            st.error(f"Error loading model predictions: {e}")
            st.info("Run `python models/train_improved_model.py` to train the model.")
    else:
        st.info(
            "Model not yet trained. Run `python models/train_improved_model.py` "
            "to train the winner prediction model."
        )

# ── Model Confidence ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Confidence")

if model_path.exists():
    # Show feature importance
    importance_path = Path(__file__).parent / "models" / "saved_models" / "feature_importance_v2.csv"
    if importance_path.exists():
        feat_imp = pd.read_csv(importance_path).head(20)
        feat_imp.columns = ['Feature', 'Importance']
        st.dataframe(feat_imp, hide_index=True, height=get_dataframe_height(feat_imp))
        st.caption("Top 20 Most Important Features")
else:
    st.info(
        "Confidence metrics will appear here once a model is trained."
    )

# ── Model Quality Statistics ────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Quality Statistics")

with st.expander("View Detailed Model Performance Metrics", expanded=False):
    try:
        # Import evaluation functions
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from models.evaluate_model_stats import (
            load_model_and_features,
            load_and_prepare_data,
            get_predictions,
            compute_overall_metrics,
            compute_top_n_accuracy,
            compute_calibration_bins,
            get_feature_importance
        )
        
        with st.spinner("Computing model quality metrics..."):
            # Load model and data
            model, feature_cols = load_model_and_features()
            _, val_df, test_df, _ = load_and_prepare_data(feature_cols)
            
            # Get predictions (validation + test only — train metrics are
            # always inflated because the model was fit to that data)
            val_proba, y_val = get_predictions(model, val_df, feature_cols)
            test_proba, y_test = get_predictions(model, test_df, feature_cols)
        
        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overall Metrics", "🎯 Top-N Accuracy", "⚖️ Calibration", "🔍 Feature Importance"])
        
        with tab1:
            st.markdown("### Overall Performance Metrics")
            
            overall_metrics_df = pd.DataFrame({
                'Validation': compute_overall_metrics(y_val, val_proba),
                'Test': compute_overall_metrics(y_test, test_proba)
            }).T
            
            # Format for display
            metrics_display = overall_metrics_df[['AUC-ROC', 'Log Loss', 'Average Precision', 'Brier Score']].copy()
            st.dataframe(metrics_display.style.format({
                'AUC-ROC': '{:.4f}',
                'Log Loss': '{:.4f}',
                'Average Precision': '{:.4f}',
                'Brier Score': '{:.4f}'
            }))
            
            st.markdown("""
            **Metric Definitions:**
            - **AUC-ROC**: Area under ROC curve (0.5-1.0, higher is better). Measures discrimination ability.
            - **Log Loss**: Logarithmic loss (lower is better). Measures prediction accuracy.
            - **Average Precision**: Summary of precision-recall curve (higher is better).
            - **Brier Score**: Mean squared error of predictions (lower is better). Measures calibration.
            """)
            
            # Summary stats
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Test AUC-ROC", f"{overall_metrics_df.loc['Test', 'AUC-ROC']:.4f}")
            with col_b:
                st.metric("Total Test Samples", f"{int(overall_metrics_df.loc['Test', 'Total Samples']):,}")
            with col_c:
                st.metric("Test Winners", f"{int(overall_metrics_df.loc['Test', 'Positive Samples'])}")
        
        with tab2:
            st.markdown("### Top-N Accuracy")
            st.markdown("Does the actual winner appear in the model's top N predictions per tournament?")
            
            top_n_df = compute_top_n_accuracy(test_df, test_proba, feature_cols)
            
            # Format for display
            display_top_n = top_n_df.copy()
            display_top_n['Top-N Accuracy'] = display_top_n['Top-N Accuracy'].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(display_top_n, hide_index=True, height=get_dataframe_height(display_top_n))
            
            # Highlight key metrics
            top5_acc = top_n_df[top_n_df['Top N'] == 5]['Top-N Accuracy'].values[0]
            top10_acc = top_n_df[top_n_df['Top N'] == 10]['Top-N Accuracy'].values[0]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Top-5 Accuracy", f"{top5_acc*100:.1f}%", 
                         help="Winner appears in top 5 predictions")
            with col_b:
                st.metric("Top-10 Accuracy", f"{top10_acc*100:.1f}%",
                         help="Winner appears in top 10 predictions")
        
        with tab3:
            st.markdown("### Calibration Analysis")
            st.markdown("How well do predicted probabilities match actual win rates?")
            
            cal_df, ece = compute_calibration_bins(y_test, test_proba, n_bins=10)
            
            # Format for display
            display_cal = cal_df.copy()
            display_cal['Mean Predicted Prob'] = display_cal['Mean Predicted Prob'].apply(lambda x: f"{x*100:.2f}%")
            display_cal['Actual Win Rate'] = display_cal['Actual Win Rate'].apply(lambda x: f"{x*100:.2f}%")
            display_cal['Calibration Error'] = display_cal['Calibration Error'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(display_cal, hide_index=True, height=get_dataframe_height(display_cal))
            
            st.metric("Expected Calibration Error (ECE)", f"{ece:.4f}",
                     help="Lower is better; <0.1 indicates good calibration")
            
            st.markdown("""
            **Interpretation**: A well-calibrated model should have predicted probabilities 
            that closely match actual win rates. For example, if the model predicts 20% win 
            probability for a group of players, approximately 20% of them should actually win.
            """)
        
        with tab4:
            st.markdown("### Feature Importance")
            st.markdown("Top features driving model predictions (by gain)")
            
            importance_df = get_feature_importance(model, feature_cols)
            
            # Show top 20
            display_importance = importance_df.head(20)[['Feature', 'Importance (%)', 'Cumulative (%)']].copy()
            display_importance['Importance (%)'] = display_importance['Importance (%)'].apply(lambda x: f"{x:.2f}%")
            display_importance['Cumulative (%)'] = display_importance['Cumulative (%)'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_importance, hide_index=True, height=get_dataframe_height(display_importance))
            
            st.caption("Higher percentage = more important for predictions")
    
    except Exception as e:
        st.error(f"Could not compute model quality statistics: {e}")
        st.info("Run `python models/evaluate_model_stats.py` in terminal to see full statistics.")

# ── Upcoming Tournaments ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 Upcoming PGA Tour Events")

try:
    from models.predict_upcoming import get_upcoming_tournaments
    upcoming_df = get_upcoming_tournaments(days_ahead=90)
    
    if len(upcoming_df) > 0:
        # Format for display
        display_upcoming = upcoming_df[['date', 'name']].copy()
        display_upcoming['date'] = display_upcoming['date'].dt.strftime('%b %d, %Y')
        display_upcoming.columns = ['Date', 'Tournament']
        
        st.dataframe(display_upcoming, hide_index=True, height=get_dataframe_height(display_upcoming))
        st.caption(f"Showing {len(display_upcoming)} upcoming tournaments in the next 90 days")
    else:
        st.info("No upcoming tournaments found in the next 90 days")
except Exception as e:
    st.warning(f"Could not fetch upcoming tournaments: {e}")
    st.markdown(
        """
    > **Note:** Enable upcoming tournament display by ensuring `scrapers/espn_golf.py` 
    > is accessible and ESPN API is available.
    """
    )

# ── Recent Form / Stats & Feature Preview ───────────────────────────────────
st.markdown("---")
st.subheader("📈 Player Stats & Features")

# Load features dataset (prefer OWGR-enhanced version)
features_path = Path(__file__).parent / "data_files" / "espn_with_owgr_features.parquet"
if not features_path.exists():
    features_path = Path(__file__).parent / "data_files" / "espn_player_tournament_features.parquet"

if features_path.exists():
    # Show feature dataset and interactive filters on the main page (not inside an expander)
    df_feats = pd.read_parquet(features_path)
    
    # Check if OWGR features are present
    has_owgr = 'owgr_rank_current' in df_feats.columns
    owgr_badge = "🌍 **WITH OWGR DATA**" if has_owgr else ""
    
    years_list = sorted(df_feats['year'].astype(int).unique().tolist())
    st.write(f"**Features:** {len(df_feats):,} rows · Years: {years_list} · Players: {df_feats['player_id'].nunique():,} {owgr_badge}")
    
    if has_owgr:
        owgr_coverage = (df_feats['owgr_rank_current'].notna().sum() / len(df_feats) * 100)
        st.caption(f"OWGR Coverage: {owgr_coverage:.1f}% of records have world ranking data")

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

        player_sample = sorted(
            df_feats['name'].dropna().astype(str).unique().tolist(),
            key=lambda s: s.lower()
        )[:200]
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
            # Prior performance features
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
            'played_last_30d': 'Played (Last 30d)',
            # OWGR features
            'owgr_rank_current': 'World Rank',
            'owgr_rank_4w_ago': 'World Rank (4w ago)',
            'owgr_rank_12w_ago': 'World Rank (12w ago)',
            'owgr_rank_52w_ago': 'World Rank (52w ago)',
            'owgr_points_current': 'OWGR Points',
            'owgr_rank_change_4w': 'Rank Δ (4w)',
            'owgr_rank_change_12w': 'Rank Δ (12w)',
            'owgr_rank_change_52w': 'Rank Δ (52w)',
            'owgr_data_staleness_days': 'OWGR Data Age (days)'
        }
        rename_map = {k: v for k, v in display_names.items() if k in view_display.columns}
        view_display = view_display.rename(columns=rename_map)

        # Show filtered table without index
        st.dataframe(view_display.head(200), hide_index=True, height=get_dataframe_height(view_display.head(200)))

    # Quick aggregate leaderboards (rename for display)
    st.markdown("**Quick leaderboards (all years)**")
    try:
        top_form = df_feats.dropna(subset=['prior_avg_score']).nsmallest(10, 'prior_avg_score')[['name','prior_avg_score','prior_count']].drop_duplicates()
        top_form = top_form.rename(columns={
            'name': 'Player',
            'prior_avg_score': 'Prior Avg Score',
            'prior_count': 'Previous Tournaments'
        })
        st.dataframe(top_form, height=get_dataframe_height(top_form), hide_index=True)
    except Exception:
        st.write("No form leaderboard available yet.")
else:
    st.info("Features dataset not found. Run `python features/build_features.py` to create features, then optionally run `python features/build_owgr_features.py` to add OWGR data.")

# ── Value Bets (Odds Comparison) ─────────────────────────────────────────────
st.markdown("---")
st.subheader("💰 Value Bets – Model vs Market Odds")

MAJOR_OPTIONS = {
    "Masters (Apr 2026)":        "masters",
    "PGA Championship (May 2026)": "pga_champ",
    "US Open (Jun 2026)":        "us_open",
    "The Open (Jul 2026)":       "open",
    "All Majors":                "all",
}

odds_col1, odds_col2 = st.columns([2, 1])
with odds_col1:
    selected_major = st.selectbox(
        "Select Major", list(MAJOR_OPTIONS.keys()), index=4
    )
with odds_col2:
    refresh_odds = st.button("🔄 Refresh Odds (uses 4 API credits)", type="secondary")

event_key = MAJOR_OPTIONS[selected_major]

odds_data_path = Path(__file__).parent / "data_files" / "odds_consensus_latest.parquet"
odds_available = odds_data_path.exists()

if refresh_odds:
    with st.spinner("Fetching latest odds from The Odds API…"):
        try:
            from scrapers.odds_api import fetch_all_golf_odds
            events = None if event_key == "all" else [event_key]
            fetch_all_golf_odds(events=events)
            odds_available = True
            st.success("Odds refreshed!")
        except Exception as e:
            st.error(f"Could not refresh odds: {e}")

if not odds_available:
    st.info(
        "No odds data yet. Click **Refresh Odds** above or run "
        "`python scrapers/odds_api.py` in your terminal."
    )
else:
    try:
        from tools.odds_comparison import load_model_predictions, load_odds, build_comparison

        with st.spinner("Building value-bet comparison…"):
            preds = load_model_predictions(event_key)
            odds_df = load_odds(event_key)
            comp = build_comparison(preds, odds_df)

        # Separate positive-edge and negative-edge rows
        has_odds = comp[comp["avg_novig_prob"].notna()].copy()
        positive = has_odds[has_odds["edge_pp"] > 0].copy()
        negative = has_odds[has_odds["edge_pp"] < 0].copy()

        # ── Summary metrics ────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Value Bets Found", len(positive))
        m2.metric("Players w/ Market Lines", len(has_odds))
        m3.metric(
            "Best Edge",
            f"{positive['edge_pp'].max():.1f}pp" if not positive.empty else "—",
            delta=None,
        )

        # ── Value bets table ───────────────────────────────────────────────
        if positive.empty:
            st.info("No positive edges found for the selected events.")
        else:
            def fmt_american(val):
                if pd.isna(val):
                    return "N/A"
                return f"+{int(val)}" if val > 0 else str(int(val))

            display = positive[[
                "event_label", "name", "model_prob", "avg_novig_prob",
                "edge_pp", "dk_odds", "best_odds", "best_book", "half_kelly_pct",
            ]].copy()
            display = display.rename(columns={
                "event_label":   "Event",
                "name":          "Player",
                "model_prob":    "Model %",
                "avg_novig_prob": "Mkt NoVig %",
                "edge_pp":       "Edge (pp)",
                "dk_odds":       "DK Odds",
                "best_odds":     "Best Odds",
                "best_book":     "Best Book",
                "half_kelly_pct": "Half-Kelly %",
            })
            display["Model %"]    = display["Model %"].apply(lambda x: f"{x*100:.2f}%")
            display["Mkt NoVig %"] = display["Mkt NoVig %"].apply(lambda x: f"{x*100:.2f}%")
            display["Edge (pp)"]  = display["Edge (pp)"].apply(lambda x: f"+{x:.2f}")
            display["DK Odds"]    = display["DK Odds"].apply(fmt_american)
            display["Best Odds"]  = display["Best Odds"].apply(fmt_american)
            display["Half-Kelly %"] = display["Half-Kelly %"].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
            )

            st.dataframe(
                display,
                hide_index=True,
                height=get_dataframe_height(display, max_height=500),
            )
            st.caption(
                "**Edge (pp)** = Model win% minus market no-vig implied probability. "
                "**Half-Kelly %** = suggested stake as % of bankroll (half-Kelly criterion). "
                "Positive edge = model sees more value than the market price implies."
            )

        # ── Model vs Market divergence ─────────────────────────────────────
        with st.expander("Model underweights these players (market favours them)"):
            if negative.empty:
                st.write("None.")
            else:
                neg_display = negative[[
                    "event_label", "name", "model_prob", "avg_novig_prob", "edge_pp", "dk_odds"
                ]].head(20).copy()
                neg_display = neg_display.rename(columns={
                    "event_label": "Event", "name": "Player",
                    "model_prob": "Model %", "avg_novig_prob": "Mkt NoVig %",
                    "edge_pp": "Edge (pp)", "dk_odds": "DK Odds",
                })
                neg_display["Model %"]    = neg_display["Model %"].apply(lambda x: f"{x*100:.2f}%")
                neg_display["Mkt NoVig %"] = neg_display["Mkt NoVig %"].apply(lambda x: f"{x*100:.2f}%")
                neg_display["Edge (pp)"]  = neg_display["Edge (pp)"].apply(lambda x: f"{x:.2f}")
                neg_display["DK Odds"]    = neg_display["DK Odds"].apply(fmt_american)
                st.dataframe(neg_display, hide_index=True)

        if odds_data_path.exists():
            import os
            mtime = pd.Timestamp(os.path.getmtime(odds_data_path), unit='s', tz='UTC')
            st.caption(f"Odds data last updated: {mtime.strftime('%Y-%m-%d %H:%M UTC')}")

    except Exception as e:
        st.error(f"Could not build odds comparison: {e}")

# ── Footer ───────────────────────────────────────────────────────────────────

# Include shared footer component if available
try:
    from footer import add_betting_oracle_footer
    add_betting_oracle_footer()
except Exception:
    # If footer component is missing or fails, silently continue
    pass
