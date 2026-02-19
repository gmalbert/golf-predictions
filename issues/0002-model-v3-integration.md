Title: Integrate model-v3 into Streamlit UI and prediction pipeline

Checklist:
- [ ] Add model-v3 to model registry (`models/saved_models/`) and include version selector in the UI (`predictions.py`).
- [ ] Add end-to-end tests that compare v2 vs v3 outputs on a fixed dataset (`tests/test_model_v3_integration.py`).
- [ ] Update `models/train_improved_model.py` to produce V3 artifacts and update `models/saved_models/model_features_v3.txt`.
- [ ] Add migration docs describing differences between v2 and v3 (features, expected performance uplift).
- [ ] Add an A/B validation job to compute lift metrics (AUC, top‑N accuracy, ROI on value bets).
- [ ] Update Streamlit help text and README with instructions to select model version.

Notes:
- Priority: Medium
- Owner: @you (assign in GitHub issue)
- ETA: 2–3 weeks

Related files: `models/train_improved_model.py`, `models/saved_models/`, `predictions.py`, `tests/`

Suggested GitHub issue body (use with `gh issue create`):
> Wire the new model-v3 into the app: save artifacts, add UI model selector, create integration tests, and run an A/B comparison against v2 to quantify improvements.