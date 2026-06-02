# Fairway Oracle — Model Suggested Enhancements

## Priority 1: Win Probability Model

### Strokes Gained Integration
- Replace OWGR rank as the primary feature with **Strokes Gained: Total** (SG:T) and its components (SG:OTT, SG:APP, SG:ARG, SG:P).
- DataGolf API provides SG decomposition free for the current week.

### Course Fit Features
- Course fit is among the strongest predictors in golf. Add `course_history_avg_finish`, `course_fit_score` (based on SG categories that match the course layout).
- Example: Augusta favours high SG:APP; Pebble Beach favours SG:ARG and SG:P.

### Recent Form Weighting
- Add exponentially decayed recent form: last 3 events weighted more than last 20. Players entering tournaments on cold streaks vs. hot streaks behave very differently.

### Cut-Making Probability
- Add a separate classification model for `makes_cut` (binary). Players who miss cuts frequently are poor outright bets regardless of price.

## Priority 2: Field Strength Adjustment

### Field Quality Normalisation
- A top-10 at a full-strength event (Players, Masters) is more predictive than a top-10 at a weak-field event.
- Encode `field_avg_owgr` as a normalisation factor for recent finish stats.

### Major vs. Regular Event Split
- Train separate models for major championships vs. regular tour events. Strategy, pressure, and course setup differ significantly.

## Priority 3: Calibration & Infrastructure

### Implied Probability Calibration
- Apply Platt scaling to the model output. Favourite golfers (e.g., Scottie Scheffler at −200) are frequently over-bet; model should reflect this.

### Closing Line Value Tracking
- Compare `avg_novig_prob` at time of post to closing DraftKings price. Track CLV weekly to validate model edge.

### DataGolf Integration
- DataGolf's live model outputs can serve as a cross-validation benchmark. Add a comparison column in the dashboard.
