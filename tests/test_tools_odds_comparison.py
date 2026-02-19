import pandas as pd
from pathlib import Path

import pytest

from tools import odds_comparison as oc


def _write_parquet(path: Path, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_load_odds_accepts_rotowire_schema(tmp_path, monkeypatch):
    # Create a temporary data_files dir and point the module at it
    temp_data = tmp_path / "data_files"
    temp_data.mkdir()
    monkeypatch.setattr(oc, "DATA_DIR", temp_data)

    # Rotowire-style schema: player_name, event_name
    df = pd.DataFrame([
        {
            "player_name": "Scottie Scheffler",
            "event_name": "Masters Tournament",
            "best_book": "betrivers",
            "avg_novig_prob": 0.025,
            "dk_odds": 320,
            "best_odds": 320,
        },
        {
            "player_name": "Rory McIlroy",
            "event_name": "Masters Tournament",
            "best_book": "mgm",
            "avg_novig_prob": 0.008,
            "dk_odds": None,
            "best_odds": 1200,
        },
    ])

    _write_parquet(temp_data / "odds_consensus_latest.parquet", df)

    # Should not raise and should normalize to provide `player` + `event_label`
    out = oc.load_odds("masters")
    assert not out.empty
    assert "player" in out.columns
    assert "event_label" in out.columns
    assert all(out["event_label"].fillna("") == "masters")


def test_load_odds_accepts_legacy_schema(tmp_path, monkeypatch):
    temp_data = tmp_path / "data_files"
    temp_data.mkdir()
    monkeypatch.setattr(oc, "DATA_DIR", temp_data)

    # Legacy schema: player, event_label
    df = pd.DataFrame([
        {"player": "Scottie Scheffler", "event_label": "masters", "avg_novig_prob": 0.03},
        {"player": "Rory McIlroy", "event_label": "masters", "avg_novig_prob": 0.01},
    ])

    _write_parquet(temp_data / "odds_consensus_latest.parquet", df)

    out = oc.load_odds("masters")
    assert not out.empty
    assert "player" in out.columns
    # event_label should be preserved
    assert set(out["event_label"]) == {"masters"}
