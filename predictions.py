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

# ── Recent Form / Stats Preview ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Player Stats Preview")
st.markdown(
    "Once data pipelines are connected, recent player form and key stats "
    "(SG:Tee-to-Green, SG:Putting, Top-10 %, etc.) will appear here."
)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Fairway Oracle © 2026 · Not financial advice · For entertainment & research only")
