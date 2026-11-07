import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from scripts import export_best_bets as feed

NOW = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)


class GolfFeedTests(unittest.TestCase):
    def run_export(self, quote=None, model=None, csv_only=False):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            (data / "predictions_cache").mkdir()
            quote = quote or {"player_name": "Player", "event_name": "Event", "best_odds": 900, "best_book": "Book", "avg_novig_prob": .9, "fetched_at": NOW.isoformat()}
            model = model or {"name": "Player", "tournament": "Event", "date": "2026-10-29", "win_probability": .15}
            if csv_only:
                pd.DataFrame([quote]).to_csv(data / "odds_consensus_latest.csv", index=False)
            else:
                pd.DataFrame([quote]).to_parquet(data / "odds_consensus_latest.parquet")
            pd.DataFrame([model]).to_parquet(data / "predictions_cache/event.parquet")
            with patch.object(feed, "DATA", data):
                feed.export(now=NOW)
            return json.loads((data / "best_bets_today.json").read_text())

    def test_model_probability_event_date_and_best_price_are_used(self):
        result = self.run_export()
        bet = result["bets"][0]
        self.assertEqual((bet["confidence"], bet["edge"], bet["tier"]), (.15, .05, "Strong"))
        self.assertEqual(bet["game_date"], "2026-10-29")
        self.assertEqual(bet["odds"], 900)

    def test_undated_legacy_csv_cannot_be_stamped_as_current(self):
        result = self.run_export(quote={"player": "Player", "tournament": "Event", "best_odds": 900}, csv_only=True)
        self.assertEqual(result["meta"]["status"], "pipeline_pending")
        self.assertEqual(result["bets"], [])

    def test_expired_quotes_and_historical_forecasts_are_withheld(self):
        for quote, model in [({"fetched_at": "2026-06-01"}, None), (None, {"name": "Player", "tournament": "Event", "date": "2024-04-11", "win_probability": .99})]:
            self.assertEqual(self.run_export(quote=quote, model=model)["meta"]["status"], "pipeline_pending")

    def test_different_events_are_never_joined(self):
        result = self.run_export(model={"name": "Player", "tournament": "Other Event", "date": "2026-10-29", "win_probability": .99})
        self.assertEqual(result["meta"]["status"], "pipeline_pending")

    def test_canonical_tiers(self):
        self.assertEqual([feed._tier_from_edge(v) for v in [.009, .01, .03, .06]], ["Standard", "Good", "Strong", "Elite"])


if __name__ == "__main__":
    unittest.main()
