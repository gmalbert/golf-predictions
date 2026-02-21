"""
Monitoring & Alerts – email and Discord notifications for predictions and value bets.

Usage
-----
    from monitoring.alerts import send_value_bet_alert, send_discord_alert

    # Email: requires SMTP_HOST / SMTP_USER / SMTP_PASS / ALERT_TO_EMAIL env vars.
    send_value_bet_alert(value_bets_df, tournament="Genesis Invitational")

    # Discord: requires DISCORD_WEBHOOK_URL env var (or pass directly).
    send_discord_alert("🏌️ Top pick: Scottie Scheffler @+450")
"""

from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd
import requests

# ── Discord ───────────────────────────────────────────────────────────────────

def send_discord_alert(
    message: str,
    webhook_url: str | None = None,
    username: str = "Fairway Oracle",
) -> bool:
    """
    Send a plain-text message to a Discord channel via webhook.

    Parameters
    ----------
    message     : The text content (supports markdown).
    webhook_url : Discord webhook URL. Falls back to DISCORD_WEBHOOK_URL env var.
    username    : Display name for the bot post.

    Returns True on success, False on failure.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
    if not url:
        print("[WARN] DISCORD_WEBHOOK_URL not set – skipping Discord alert.")
        return False

    try:
        resp = requests.post(
            url,
            json={"content": message, "username": username},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        print(f"[ERROR] Discord alert failed: {exc}")
        return False


# ── Email ─────────────────────────────────────────────────────────────────────

def _build_value_bet_email(
    value_bets: pd.DataFrame, tournament: str
) -> tuple[str, str]:
    """Return (subject, html_body) for a value-bet alert email."""
    subject = f"🏌️ Fairway Oracle – Value Bets: {tournament}"

    rows_html = ""
    for _, r in value_bets.iterrows():
        edge_str = f"+{r['edge_pp']:.1f}pp" if r.get("edge_pp") else "–"
        rows_html += (
            f"<tr>"
            f"<td>{r.get('name','–')}</td>"
            f"<td>{r.get('win_probability',0)*100:.1f}%</td>"
            f"<td>{r.get('avg_novig_prob',0)*100:.1f}%</td>"
            f"<td><b>{edge_str}</b></td>"
            f"<td>{r.get('best_odds','–')}</td>"
            f"<td>{r.get('best_book','–')}</td>"
            f"</tr>"
        )

    html = f"""
    <html><body style="font-family:sans-serif;color:#222">
    <h2>🏌️ Fairway Oracle – Value Bets</h2>
    <p><b>Tournament:</b> {tournament} &nbsp;|&nbsp; <b>Generated:</b> {datetime.utcnow():%Y-%m-%d %H:%M UTC}</p>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
      <thead style="background:#1a5c2a;color:#fff">
        <tr>
          <th>Player</th><th>Model Win%</th><th>Market NoVig%</th>
          <th>Edge</th><th>Best Odds</th><th>Best Book</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style="color:#888;font-size:12px">
      Edge = model probability − market implied probability. Positive edge = value bet.
      Always gamble responsibly. This is for informational purposes only.
    </p>
    </body></html>
    """
    return subject, html


def send_email_alert(
    subject: str,
    html_body: str,
    to_email: str | None = None,
    from_email: str | None = None,
    smtp_host: str | None = None,
    smtp_port: int | None = None,
    smtp_user: str | None = None,
    smtp_pass: str | None = None,
) -> bool:
    """
    Send an HTML email.  All parameters fall back to environment variables:
        SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS,
        ALERT_FROM_EMAIL, ALERT_TO_EMAIL.

    Returns True on success, False on failure.
    """
    _host  = smtp_host  or os.getenv("SMTP_HOST",       "smtp.gmail.com")
    _port  = smtp_port  or int(os.getenv("SMTP_PORT",   "587"))
    _user  = smtp_user  or os.getenv("SMTP_USER",       "")
    _pass  = smtp_pass  or os.getenv("SMTP_PASS",       "")
    _from  = from_email or os.getenv("ALERT_FROM_EMAIL", _user)
    _to    = to_email   or os.getenv("ALERT_TO_EMAIL",   "")

    if not (_user and _pass and _to):
        print("[WARN] Email credentials incomplete – set SMTP_USER, SMTP_PASS, ALERT_TO_EMAIL.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = _from
    msg["To"]      = _to
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(_host, _port) as server:
            server.starttls()
            server.login(_user, _pass)
            server.send_message(msg)
        print(f"[OK] Email sent to {_to}")
        return True
    except Exception as exc:
        print(f"[ERROR] Email failed: {exc}")
        return False


# ── Convenience wrappers ──────────────────────────────────────────────────────

def send_value_bet_alert(
    value_bets: pd.DataFrame,
    tournament: str = "Upcoming Tournament",
    channels: list[str] | None = None,
) -> None:
    """
    Send value-bet notifications to all configured channels.

    Parameters
    ----------
    value_bets : DataFrame with columns: name, win_probability, avg_novig_prob,
                 edge_pp, best_odds, best_book.
    tournament : Tournament name for the message header.
    channels   : List of channels to send to. Defaults to ['discord', 'email'].
    """
    channels = channels or ["discord", "email"]

    if value_bets.empty:
        print("[INFO] No value bets to alert.")
        return

    top_bets = value_bets[value_bets.get("Value Bet", pd.Series(dtype=str)).str.startswith("YES", na=False)]
    if top_bets.empty:
        top_bets = value_bets.head(5)

    if "discord" in channels:
        lines = [f"🏌️ **Fairway Oracle – Value Bets: {tournament}**", ""]
        for _, r in top_bets.head(10).iterrows():
            edge = f"+{r.get('edge_pp',0):.1f}pp" if r.get("edge_pp") else "?"
            lines.append(
                f"• **{r.get('name','?')}** – Model {r.get('win_probability',0)*100:.1f}%"
                f" | Market {r.get('avg_novig_prob',0)*100:.1f}% | Edge **{edge}**"
                f" | Odds {r.get('best_odds','?')} ({r.get('best_book','?')})"
            )
        send_discord_alert("\n".join(lines))

    if "email" in channels:
        subject, html = _build_value_bet_email(top_bets, tournament)
        send_email_alert(subject, html)


def send_predictions_ready_alert(
    tournament: str,
    top_pick: str,
    top_prob: float,
    num_value_bets: int = 0,
) -> None:
    """Quick notification that fresh predictions are available."""
    msg = (
        f"🏌️ **{tournament}** predictions are ready!\n"
        f"  Top pick: **{top_pick}** ({top_prob*100:.1f}% win probability)\n"
        f"  Value bets found: **{num_value_bets}**\n"
        f"  Open Fairway Oracle to view full rankings."
    )
    send_discord_alert(msg)

    subject = f"Fairway Oracle – {tournament} Predictions Ready"
    html = f"""
    <html><body style="font-family:sans-serif">
    <h2>🏌️ {tournament} – Predictions Ready</h2>
    <p><b>Top pick:</b> {top_pick} ({top_prob*100:.1f}% win probability)</p>
    <p><b>Value bets found:</b> {num_value_bets}</p>
    </body></html>
    """
    send_email_alert(subject, html)
