import xgboost as xgb
import traceback

clf = xgb.XGBClassifier()
try:
    clf.load_model('models/saved_models/winner_predictor_v2.json')
    print('SUCCESS: Native model loaded')
    print(f'Booster type: {type(clf.get_booster())}')
except Exception as e:
    print(f'ERROR loading native model: {e}')
    traceback.print_exc()
