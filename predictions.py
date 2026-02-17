"""
Fairway Oracle - PGA Tournament Predictions
Predict winners of upcoming PGA tournaments for betting insights.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from utils.tournament_display import format_tournament_display, tournament_sort_key


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
        features_path = Path(__file__).parent / "data_files" / "espn_with_owgr_features.parquet"
        if not features_path.exists():
            features_path = Path(__file__).parent / "data_files" / "espn_player_tournament_features.parquet"

    if not features_path.exists():
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
            # Format as selectbox options
            upcoming_options = []
            upcoming_mapping = {}
            
            for _, row in upcoming_df.iterrows():
                date_str = row['date'].strftime('%b %d, %Y')
                display = f"{row['name']} - {date_str}"
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

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🏌️ Predictions – {tournament_display}")
    
    if is_upcoming:
        # Handle upcoming tournament predictions
        try:
            from models.predict_upcoming import predict_upcoming_tournament
            
            with st.spinner("Building predictions for upcoming tournament..."):
                predictions = predict_upcoming_tournament(tournament, tournament_id, tournament_date)
            
            if not predictions.empty:
                # Build display for upcoming predictions
                display_cols = ['name', 'win_probability']
                col_renames = {'name': 'Player', 'win_probability': 'Win Prob'}
                
                if 'owgr_rank_current' in predictions.columns:
                    display_cols.append('owgr_rank_current')
                    col_renames['owgr_rank_current'] = 'OWGR Rank'
                
                pred_display = predictions[display_cols].head(num_predictions).copy()
                pred_display = pred_display.rename(columns=col_renames)
                
                # Format percentages
                pred_display['Win Prob'] = pred_display['Win Prob'].apply(lambda x: f"{x*100:.2f}%")
                
                st.dataframe(pred_display, height=get_dataframe_height(pred_display), hide_index=True)
                
                # Show probability coverage
                top_n_prob = predictions['win_probability'].head(num_predictions).sum()
                total_field = len(predictions)
                st.caption(f"Showing top {num_predictions} of {total_field} players (covering {top_n_prob*100:.1f}% of total win probability)")
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
                
                # Load features for selected tournament (prefer OWGR-enhanced version)
                features_path = Path(__file__).parent / "data_files" / "espn_with_owgr_features.parquet"
                if not features_path.exists():
                    features_path = Path(__file__).parent / "data_files" / "espn_player_tournament_features.parquet"
                
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
                        
                        # Build display columns dynamically (OWGR may be absent)
                        display_cols = ['name', 'win_probability']
                        col_renames = {'name': 'Player', 'win_probability': 'Win Prob'}

                        if 'owgr_rank_current' in predictions.columns:
                            display_cols.append('owgr_rank_current')
                            col_renames['owgr_rank_current'] = 'OWGR Rank'

                        if 'tournament_rank' in predictions.columns:
                            display_cols.append('tournament_rank')
                            col_renames['tournament_rank'] = 'Actual Finish'

                        pred_display = predictions[display_cols].head(num_predictions).copy()
                        pred_display = pred_display.rename(columns=col_renames)

                        # Format columns
                        pred_display['Win Prob'] = pred_display['Win Prob'].apply(lambda x: f"{x:.2%}")
                        if 'OWGR Rank' in pred_display.columns:
                            # Convert to int for non-null values, then to string, then fill NaN
                            pred_display['OWGR Rank'] = pred_display['OWGR Rank'].apply(
                                lambda x: str(int(x)) if pd.notna(x) else 'N/A'
                            )

                        st.dataframe(pred_display, hide_index=True, height=get_dataframe_height(pred_display))
                        
                        # Show probability coverage
                        top_n_prob = predictions['win_probability'].head(num_predictions).sum()
                        total_field = len(predictions)
                        st.caption(f"Showing top {num_predictions} of {total_field} players (covering {top_n_prob*100:.1f}% of total win probability)")
                        st.caption(f"Based on {tournament} {selected_year} field")

                        # Informative note when OWGR features are missing
                        if 'owgr_rank_current' not in predictions.columns:
                            st.caption("OWGR features not present for this dataset — run `python features/build_owgr_features.py` to add world ranking data (optional).")
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

with col2:
    st.subheader("📊 Model Confidence")
    
    if model_path.exists():
        # Show feature importance
        importance_path = Path(__file__).parent / "models" / "saved_models" / "feature_importance_v2.csv"
        if importance_path.exists():
            feat_imp = pd.read_csv(importance_path).head(10)
            feat_imp.columns = ['Feature', 'Importance']
            st.dataframe(feat_imp, hide_index=True, height=get_dataframe_height(feat_imp))
            st.caption("Top 10 Most Important Features")
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
            train_df, val_df, test_df, _ = load_and_prepare_data(feature_cols)
            
            # Get predictions
            train_proba, y_train = get_predictions(model, train_df, feature_cols)
            val_proba, y_val = get_predictions(model, val_df, feature_cols)
            test_proba, y_test = get_predictions(model, test_df, feature_cols)
        
        # Create tabs for different metric categories
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overall Metrics", "🎯 Top-N Accuracy", "⚖️ Calibration", "🔍 Feature Importance"])
        
        with tab1:
            st.markdown("### Overall Performance Metrics")
            
            overall_metrics_df = pd.DataFrame({
                'Train': compute_overall_metrics(y_train, train_proba),
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

# ── Footer ───────────────────────────────────────────────────────────────────

# Include shared footer component if available
try:
    from footer import add_betting_oracle_footer
    add_betting_oracle_footer()
except Exception:
    # If footer component is missing or fails, silently continue
    pass
