import pandas as pd
from models.predict_tournament import predict_field, load_model

DF = pd.read_parquet('data_files/espn_player_tournament_features.parquet')
print('owgr present in base features:', 'owgr_rank_current' in DF.columns)

field = DF[(DF.tournament=='3M Open') & (DF.year==2025)].copy()
print('field rows', len(field))

model, feature_cols = load_model()
print('model features count', len(feature_cols))

preds = predict_field(field, model=model, feature_cols=feature_cols)
print('preds columns contain owgr_rank_current?', 'owgr_rank_current' in preds.columns)
print(preds[['name','win_probability']].head())
