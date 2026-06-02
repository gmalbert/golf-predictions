# Golf Oracle — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Course History Feature

**Why:** A player's historical performance at a specific course is the highest-value feature in golf prediction models. Some players are "course specialists" — they consistently outperform their ranking at certain venues due to course geometry, grass type, or elevation. This feature exists conceptually but may not yet be in the model.

**How:**
1. Build a per-player, per-course history from the processed Parquet data: `course_avg_finish_l5`, `course_made_cut_pct`, `course_avg_strokes_vs_field`
2. Use `shift(1)` — only include prior appearances (not the current tournament)
3. Add these to the feature matrix in `check_features.py` / the model training script
4. Validate improvement: course history should be among the top-3 features in SHAP importance

**Complexity:** Low

---

## Feature 2: Strokes-Gained Category Breakdown

**Why:** Using SG: Total as a single feature averages away category-specific advantages. SG: Off-Tee matters most at long courses; SG: Putting matters most at tight courses where approach accuracy is equalized. Using the four SG categories as separate features would be a significant model improvement.

**How:**
1. Fetch `sg_off_tee`, `sg_approach`, `sg_around_green`, `sg_putting` per player from the DataGolf or PGA Tour ShotLink API
2. Compute rolling 6-tournament averages per SG category (using `shift(1)` to prevent leakage)
3. Replace `sg_total_l6` with four separate SG category features
4. Compute a `course_sg_profile_match` score: how well a player's SG strengths match the course's SG demands

**Complexity:** Medium

---

## Feature 3: Round-by-Round Win Probability Update

**Why:** After each completed round, the leaderboard position changes dramatically. `live/tournament_tracker.py` already exists as a scaffold. Updating win probability predictions after each round using current leaderboard position would make the app far more engaging during live tournaments.

**How:**
1. In `live/tournament_tracker.py`, fetch live leaderboard from PGA Tour's JSON endpoint after each round
2. Update the model's predicted win probability using: current rank, strokes behind leader, holes remaining, historical R2→finish correlation per course
3. Render an updating Plotly chart showing each player's win probability over the 4 rounds
4. Display "Round 2 Win Probability" alongside the original pre-tournament prediction

**Complexity:** Medium

---

## Feature 4: Player Withdrawal / WD Detection

**Why:** Late player withdrawals from PGA Tour events disrupt Pick 6 and outright winner predictions. A WD detection feed that flags affected players and removes them from active pick pools would prevent recommending players who have already withdrawn.

**How:**
1. Add `scripts/fetch_withdrawals.py` that monitors the PGA Tour's official tee time/entry list for `WD`, `MDF`, or `DQ` status changes
2. Alternatively, scrape ESPN Golf news for withdrawal announcements using BeautifulSoup keyword matching
3. Store `data_files/current_tournament_wds.json` with affected player names + withdrawal reason
4. On the predictions page, display a `⚠ Withdrawn` badge and exclude WD players from the top-pick recommendations

**Complexity:** Medium

---

## Feature 5: DK Pick 6 Calculator

**Why:** DraftKings Pick 6 is the most popular daily fantasy golf format. Users need to know whether to go OVER or UNDER on props like birdies made, fairways hit, and greens in regulation. The baseball app already has a similar calculator — adapting the pattern for golf would be consistent with the suite.

**How:**
1. Add a "DK Pick 6" tab to the predictions page
2. Allow users to input a DraftKings Pick 6 line (e.g., "Tiger Woods — 3.5 Birdies")
3. Use the model's predicted hole-by-hole scoring distribution (from SG data) to compute the probability of going OVER the line
4. Display: Player | Stat | Line | Model OVER% | Recommendation (OVER/UNDER)
5. Use the same tier logic as other Betting Oracle apps: ≥60% → OVER, ≤40% → UNDER, 40–60% → PASS

**Complexity:** Medium
