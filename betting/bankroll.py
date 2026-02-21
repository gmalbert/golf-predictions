"""
Bankroll Manager – fractional Kelly criterion bet-sizing with history tracking.

Usage
-----
    from betting.bankroll import BankrollManager

    mgr = BankrollManager(initial_bankroll=1000, kelly_fraction=0.25)

    bet = mgr.calculate_bet_size(model_prob=0.12, decimal_odds=9.0)
    mgr.place_bet("Scottie Scheffler", amount=bet, odds=9.0, result=True)
    mgr.summary()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HISTORY_PATH = Path(__file__).resolve().parent.parent / "data_files" / "bet_history.csv"


class BankrollManager:
    """
    Implements fractional Kelly criterion for bet-sizing.

    Parameters
    ----------
    initial_bankroll : float
        Starting capital (dollars, or any consistent unit).
    kelly_fraction : float
        Fraction of full Kelly to use (0.25 = quarter-Kelly recommended).
    max_bet_fraction : float
        Maximum fraction of current bankroll per bet (safety cap, default 5 %).
    """

    def __init__(
        self,
        initial_bankroll: float = 1_000.0,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.05,
    ):
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.history: list[dict] = []

    # ── Core calculation ──────────────────────────────────────────────────────

    def kelly_bet(self, model_prob: float, decimal_odds: float) -> float:
        """
        Compute the full-Kelly fraction.

        f = (b·p − q) / b
          where b = decimal_odds − 1, p = model_prob, q = 1 − p
        """
        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - model_prob
        return (b * model_prob - q) / b

    def calculate_bet_size(self, model_prob: float, decimal_odds: float) -> float:
        """
        Return the recommended bet amount in bankroll units.

        Returns 0 if the model sees no edge (negative Kelly).
        """
        k = self.kelly_bet(model_prob, decimal_odds)
        if k <= 0:
            return 0.0

        fractional = k * self.kelly_fraction
        max_bet = self.bankroll * self.max_bet_fraction
        return round(min(fractional * self.bankroll, max_bet), 2)

    def edge(self, model_prob: float, decimal_odds: float) -> float:
        """Return the raw edge: model_prob − implied_prob."""
        implied = 1.0 / decimal_odds
        return model_prob - implied

    # ── Record keeping ────────────────────────────────────────────────────────

    def place_bet(
        self,
        player: str,
        amount: float,
        decimal_odds: float,
        result: bool,
        tournament: str = "",
        model_prob: float | None = None,
        notes: str = "",
    ) -> dict:
        """
        Record a completed bet and update the bankroll.

        Parameters
        ----------
        player      : Player name.
        amount      : Amount wagered.
        decimal_odds: Decimal odds at time of bet.
        result      : True if the bet won, False otherwise.
        tournament  : Optional tournament name for reference.
        model_prob  : Model win probability at time of bet.
        notes       : Free-form notes.

        Returns
        -------
        dict with bet details including profit/loss and updated bankroll.
        """
        payout = amount * decimal_odds if result else 0.0
        profit = payout - amount
        self.bankroll = round(self.bankroll + profit, 2)

        record = {
            "timestamp":    pd.Timestamp.now().isoformat(),
            "player":       player,
            "tournament":   tournament,
            "amount":       amount,
            "decimal_odds": decimal_odds,
            "implied_prob": round(1.0 / decimal_odds, 4) if decimal_odds else None,
            "model_prob":   model_prob,
            "edge":         round(self.edge(model_prob, decimal_odds), 4) if model_prob else None,
            "won":          result,
            "payout":       round(payout, 2),
            "profit":       round(profit, 2),
            "bankroll":     self.bankroll,
            "notes":        notes,
        }
        self.history.append(record)
        return record

    # ── Analytics ────────────────────────────────────────────────────────────

    def summary(self, print_results: bool = True) -> pd.DataFrame:
        """Compute and display performance summary. Returns history DataFrame."""
        if not self.history:
            print("No bets recorded yet.")
            return pd.DataFrame()

        df = pd.DataFrame(self.history)
        total_bets    = len(df)
        wins          = int(df["won"].sum())
        total_wagered = df["amount"].sum()
        total_profit  = df["profit"].sum()
        roi           = total_profit / total_wagered * 100 if total_wagered else 0.0
        peak          = df["bankroll"].max()
        drawdown      = (peak - df["bankroll"].min()) / peak * 100

        if print_results:
            print("\n── Bankroll Manager Summary ──")
            print(f"  Starting bankroll  : ${self.initial_bankroll:,.2f}")
            print(f"  Current bankroll   : ${self.bankroll:,.2f}")
            print(f"  Total bets         : {total_bets}   (wins: {wins}  losses: {total_bets - wins})")
            print(f"  Win rate           : {wins/total_bets:.1%}")
            print(f"  Total wagered      : ${total_wagered:,.2f}")
            print(f"  Total profit/loss  : ${total_profit:+,.2f}")
            print(f"  ROI                : {roi:+.1f}%")
            print(f"  Peak bankroll      : ${peak:,.2f}")
            print(f"  Max drawdown       : {drawdown:.1f}%")

        return df

    def save_history(self, path: Path | str | None = None) -> Path:
        """Persist bet history to CSV."""
        out = Path(path or HISTORY_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history)
        df.to_csv(out, index=False)
        return out

    def load_history(self, path: Path | str | None = None) -> None:
        """Load bet history from CSV and restore bankroll state."""
        src = Path(path or HISTORY_PATH)
        if not src.exists():
            return
        df = pd.read_csv(src)
        self.history = df.to_dict("records")
        if self.history:
            self.bankroll = self.history[-1]["bankroll"]
