import sys
sys.path.insert(0, 'c:\\Users\\gmalb\\Downloads\\golf-predictions')

from features.build_features import parse_score
import pandas as pd

# Test parse_score
scores = ['+1', '-10', 'E', '+3', '-5', '-1', '+10']
print("Testing parse_score:")
for s in scores:
    parsed = parse_score(s)
    print(f"  {s:>5} -> {parsed:>5}")

# Test ranking
print("\nTesting ranking:")
df = pd.DataFrame({
    'score_str': ['+1', '-10', '-5', '+3', 'E'],
    'numeric': [parse_score(s) for s in ['+1', '-10', '-5', '+3', 'E']]
})
df['rank_asc'] = df['numeric'].rank(method='min', ascending=True)
df['rank_desc'] = df['numeric'].rank(method='min', ascending=False)
print(df.to_string(index=False))

print("\nLower scores should get rank 1 (winner)")
print("ascending=True is CORRECT for golf")
