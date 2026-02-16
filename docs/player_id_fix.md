# Player ID Assignment Fix

## Problem

Model was predicting uniform probabilities (49.76% for all players) despite having valid tournament data.

### Root Cause
- **player_id**: 100% NULL in features file
- **Feature completeness**: 0% for all player-dependent features (prior_avg_score, last_event_rank, career_best_rank, etc.)
- **Model impact**: AUC 0.5000 (random guessing), all feature importances 0.00%

## Investigation

1. **ESPN scraper worked correctly** - Data clean with 134 unique winners
2. **Consolidation worked** - 37,442 rows, 107 tournaments  
3. **Feature building broken** - build_features.py didn't assign player_id
4. **Missing step** - apply_player_ids.py wasn't being run in pipeline

## Solution

### Fixed Pipeline Order
```bash
# Correct order (now documented):
python features/apply_player_ids.py    # CRITICAL: Must run FIRST
python features/build_features.py       # Can only work with valid player_id
python features/build_owgr_features.py  # Optional enhancement
python models/train_improved_model.py   # Train on valid features
```

### Results After Fix

**Player ID Assignment:**
- player_id: 0% NULL (was 100%)
- Unique players: 1,650 (was 0)
- Feature completeness: 95.6% (was 0%)

**Model Performance:**
- AUC-ROC: Train=0.9859, Val=0.7355, Test=0.6874 (was 0.5000)
- Top feature importance: prior_avg_score 10.47% (was 0.00%)

**Predictions:**
- Range: 0.06% to 88.99% (was uniform 49.76%)
- Median: 11.07%
- Realistic discrimination between players

### Files Updated

1. **models/train_improved_model.py** - Use espn_player_tournament_features.parquet
2. **models/test_model_v2.py** - Use espn_player_tournament_features.parquet  
3. **models/predict_tournament.py** - Use winner_predictor_v2.joblib
4. **predictions.py** - Use correct data/model files

## Lessons Learned

1. **Player IDs are foundational** - All historical features require player grouping
2. **Pipeline order matters** - apply_player_ids.py must run before build_features.py
3. **AUC 0.5 = broken features** - Not overfitting, features provide zero signal
4. **NULL propagates** - One missing step breaks entire downstream pipeline

## Next Steps

1. ✅ Player IDs applied
2. ✅ Features rebuilt with 95.6% completeness
3. ✅ Model retrained (AUC 0.69 on test)
4. ✅ Predictions discriminative
5. ⏳ Add OWGR features (build_owgr_features.py needs optimization)
6. ⏳ Deploy to UI
