# Fairway Oracle — 6-Month Feature Roadmap

## Month 1: Tournament Hub

- **This Week page** — Tournament overview: field, course details, purse, recent winners, weather forecast.
- **Tee time tracker** — Import tee times from PGA Tour API; show which players are in the early/late wave.
- **Cut line estimator** — Live cut projection based on scoring through round 2.
- **Leaderboard integration** — Live leaderboard via ESPN API during tournament rounds.

## Month 2: Player Profiles

- **Player card** — OWGR, season stats, SG breakdown, course history, major wins.
- **Course fit score** — Visual indicator of how well a player's game matches the course characteristics.
- **Hot/cold form badge** — "Last 3 events: 2 top-10s" type summary card.
- **Career major results** — All-time majors history table.

## Month 3: Betting Intelligence

- **Value bet table** — Sorted by edge; shows model probability, best market odds, no-vig probability.
- **Outright vs. place bet comparison** — Show both win (top 1) and each-way (top 5) implied probabilities.
- **Field size filter** — Filter by "featured group" (top 50 players) to avoid thin odds on weak fields.

## Month 4: Analytics

- **Course DNA page** — Historical statistics for each PGA Tour course: scoring average, birdie rate, champion profile.
- **World ranking tracker** — Interactive chart of top-50 OWGR changes week over week.
- **Model accuracy log** — Tournament-by-tournament model accuracy and CLV tracking.

## Month 5: Notifications & Automation

- **Tuesday morning email** — Value bets for the upcoming tournament with course notes.
- **Live alerts** — Email/Discord alert when a value-bet player takes the lead or shoots 62.
- **OWGR auto-refresh** — GitHub Action runs `update_owgr_weekly.py` every Monday.

## Month 6: Advanced Features

- **Major simulator** — Monte Carlo simulation of remaining majors for the calendar year.
- **Futures tracker** — Track outright futures prices (e.g., Masters winner) across the full season.
- **Weather impact model** — Wind speed and rain affect scoring; surface Open-Meteo forecast on the tournament page.
