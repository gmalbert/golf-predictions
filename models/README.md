# Models

This directory contains machine learning models for predicting golf tournament winners.

## Current Models

### 1. Baseline Winner Predictor
**File:** `train_baseline_model.py`

A binary classification model that predicts the probability of each player winning a tournament.

**Features Used:**
- Historical performance (prior avg score, top-10 rates, etc.)
- Recent form (last 5/10 events)
- OWGR world ranking data (current rank, rank changes, points)
- Course history
- Career statistics

**Model:** XGBoost Classifier with class imbalance handling

**Training:**
```bash
python models/train_baseline_model.py
```

**Output:**
- `saved_models/baseline_winner_predictor.joblib` - Trained model
- `saved_models/model_features.txt` - List of features used
- `saved_models/feature_importance.csv` - Feature importance scores

## Making Predictions

### Predict a specific tournament:
```bash
python models/predict_tournament.py
```

### Use in Python code:
```python
from models.predict_tournament import predict_tournament

# Predict most recent Masters
results = predict_tournament("The Masters")

# Predict specific year
results = predict_tournament("The Masters", year=2024)
```

### Results include:
- Player name
- Win probability (0-100%)
- OWGR rank
- Actual finish (if available)

## Model Performance

After training, check:
- `saved_models/feature_importance.csv` - Which features matter most
- Training output - AUC scores on train/test sets

**Current performance (baseline model):**
- Uses OWGR + historical features
- Binary classification (win/loss)
- Handles severe class imbalance (~0.1% win rate)

## Next Steps

1. **Improve features:**
   - Add Strokes Gained data (SG:Total, SG:Putting, etc.)
   - Course-specific statistics
   - Weather/conditions data

2. **Try different models:**
   - LightGBM
   - Neural networks
   - Ensemble methods

3. **Better evaluation:**
   - Top-N accuracy (is winner in top 10 predictions?)
   - Calibration curves
   - Cross-validation by year

4. **Deploy:**
   - API endpoint for predictions
   - Real-time updates as tournament progresses
   - Odds comparison (model vs betting markets)
