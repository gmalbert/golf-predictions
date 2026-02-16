import xgboost as xgb
from pathlib import Path


def test_native_xgb_model_exists_and_loads():
    """Assert the XGBoost-native model file is present and can be loaded."""
    model_path = Path(__file__).parent.parent / "models" / "saved_models" / "winner_predictor_v2.json"
    assert model_path.exists(), f"Native XGBoost model not found: {model_path}"

    # Load via sklearn wrapper (preferred load path used in runtime)
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    booster = clf.get_booster()
    dump = booster.get_dump()
    assert isinstance(dump, list) and len(dump) > 0

    # Also verify direct Booster load works
    b = xgb.Booster()
    b.load_model(str(model_path))
    assert len(b.get_dump()) > 0
