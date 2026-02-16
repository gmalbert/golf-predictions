"""
Test the improved model on actual tournaments
"""
import pandas as pd
from pathlib import Path

from models.predict_tournament import load_model

MODEL_DIR = Path(__file__).parent / 'saved_models'
DATA_DIR = Path(__file__).parent.parent / 'data_files'

# Load model via runtime loader (prefers XGBoost-native file)
model, features = load_model()

print("="*70)
print("TESTING MODEL ON 2025 TOURNAMENTS")
print("="*70)

# Load data (prefer OWGR-enhanced dataset if available)
owgr_path = DATA_DIR / 'espn_with_owgr_features.parquet'
base_path = DATA_DIR / 'espn_player_tournament_features.parquet'
if owgr_path.exists():
    df = pd.read_parquet(owgr_path)
    print("[OK] Using OWGR-enhanced features for testing")
else:
    df = pd.read_parquet(base_path)
    print("[OK] Using base features for testing (no OWGR)")

# Test on 2025 tournaments (unseen during training)
test_data = df[df['year'] == 2025].copy()

print(f"\nTesting on {test_data['tournament'].nunique()} tournaments from 2025")

# Make predictions
X_test = test_data[features].fillna(0)
test_data['win_probability'] = model.predict_proba(X_test)[:, 1]

# Check a specific tournament
tournaments = test_data['tournament'].unique()
print(f"\nAvailable 2025 tournaments: {len(tournaments)}")

for i, tourn in enumerate(sorted(tournaments)[:3]):  # Show first 3
    print(f"\n{'-'*70}")
    print(f"TOURNAMENT: {tourn}")
    print(f"{'-'*70}")
    
    tourn_data = test_data[test_data['tournament'] == tourn].copy()
    tourn_data = tourn_data.sort_values('win_probability', ascending=False)
    
    print(f"\nTotal field: {len(tourn_data)} players")
    print(f"\nTop 10 predictions:")
    
    cols = ['name', 'win_probability', 'tournament_rank', 'last_event_rank', 'career_best_rank', 'prior_avg_score']
    display = tourn_data[cols].head(10).copy()
    display['win_probability'] = display['win_probability'].apply(lambda x: f"{x*100:.2f}%")
    print(display.to_string(index=False))
    
    # Show actual winner
    winner = tourn_data[tourn_data['tournament_rank'] == 1.0]
    if not winner.empty:
        winner = winner.iloc[0]
        winner_rank_in_pred = tourn_data[tourn_data['name'] == winner['name']].index[0]
        model_rank = tourn_data.index.get_loc(winner_rank_in_pred) + 1
        
        print(f"\nACTUAL WINNER: {winner['name']}")
        print(f"  Model predicted rank: #{model_rank}")
        print(f"  Win probability: {winner['win_probability']*100:.2f}%")
        print(f"  Last event rank: {winner['last_event_rank']}")
        print(f"  Career best rank: {winner['career_best_rank']}")

print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")

# Check prediction distribution
print(f"\nWin probability distribution:")
print(f"  Min:    {test_data['win_probability'].min()*100:.4f}%")
print(f"  Median: {test_data['win_probability'].median()*100:.4f}%")
print(f"  Mean:   {test_data['win_probability'].mean()*100:.4f}%")
print(f"  Max:    {test_data['win_probability'].max()*100:.4f}%")

# Check if model gives unrealistic 100% predictions
perfect_pred = test_data[test_data['win_probability'] > 0.99]
print(f"\nPredictions > 99%: {len(perfect_pred)} / {len(test_data)} ({len(perfect_pred)/len(test_data)*100:.1f}%)")

if len(perfect_pred) > 0:
    print("\nSAMPLE OF >99% PREDICTIONS:")
    perfect_sample = perfect_pred[['name', 'tournament', 'win_probability', 'tournament_rank']].head(10)
    perfect_sample['win_probability'] = perfect_sample['win_probability'].apply(lambda x: f"{x*100:.4f}%")
    print(perfect_sample.to_string())
