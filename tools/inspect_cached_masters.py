import json
from pathlib import Path

# Read the cached Masters tournament data
cache_file = Path('data_files/cache/07ec1b47ef6c48f6aa8578b211c070d4.html')

with open(cache_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

event = data['events'][0]
competition = event['competitions'][0]
competitors = competition['competitors']

print(f"Total competitors in API response: {len(competitors)}\n")

print("First 10 competitors from ESPN API:")
for i, comp in enumerate(competitors[:10]):
    athlete = comp.get('athlete', {})
    name = athlete.get('displayName', 'Unknown')
    score_obj = comp.get('score', {})
    
    if isinstance(score_obj, dict):
        score = score_obj.get('displayValue', 'N/A')
    else:
        score = str(score_obj) if score_obj else 'N/A'
    
    print(f"{i+1}. {name:30} Score: {score}")
