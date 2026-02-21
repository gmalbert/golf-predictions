"""Train models with and without new features and report metrics.
"""

import sys, os
# ensure workspace root is on the import path (script lives in tools/)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if root not in sys.path:
    sys.path.insert(0, root)
from models import train_improved_model as tim
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score


def run_experiment(feature_list):
    df = tim.load_and_prepare_data()
    Xtr, Xv, Xt, ytr, yv, yt = tim.create_time_based_splits(df, feature_list)
    model = tim.train_model(Xtr, ytr, Xv, yv, feature_list)
    results = {}
    for name, X, y in [('Train', Xtr, ytr), ('Val', Xv, yv), ('Test', Xt, yt)]:
        p = model.predict_proba(X)[:, 1]
        results[name] = {
            'auc': roc_auc_score(y, p),
            'logloss': log_loss(y, p),
            'ap': average_precision_score(y, p),
        }
        print(f"{name}: AUC {results[name]['auc']:.4f}  logloss {results[name]['logloss']:.4f}  ap {results[name]['ap']:.4f}")
    return results


if __name__ == '__main__':
    df = tim.load_and_prepare_data()
    full_feats = tim.select_features(df)
    new_feats = ['purse_size_m', 'course_length_fit', 'grass_fit', 'course_yardage']
    reduced_feats = [f for f in full_feats if f not in new_feats]

    print(f"full features count: {len(full_feats)}")
    print(f"reduced features count: {len(reduced_feats)}")

    print("\n=== Full model ===")
    full_res = run_experiment(full_feats)

    print("\n=== Reduced model ===")
    red_res = run_experiment(reduced_feats)

    print("\n=== Differences (full - reduced) ===")
    for split in ['Train', 'Val', 'Test']:
        diff = full_res[split]['auc'] - red_res[split]['auc']
        print(f"{split} AUC delta: {diff:.4f}")
