# Decision log

Spec: `OPUS_BUILD_PROMPT.md`. All tuning on external matches; the BAR-ATH public window is used
only as a wiring check and is never used to choose between versions or set a value.

## 2026-11-05

- **Data layer.** Whole-file vectorised ATD parse (`np.fromstring`) cached to
  `cache/BAR-ATH_tactical.npz` (≈4 min once, <2 s after). Coordinates scaled to 105.3 × 68.0 m
  (games.py). Parity test: nearest-player labels from the cache reproduce `baseline_raw` and
  `baseline_smooth50` with 0 mismatched frames.
- **Ball outlier gate.** A visible ball point is removed when both its links to visible
  neighbours imply > 40 m/s, or it is an isolated single-frame detection with one impossible link.
  Removes 1,911 of 113,488 visible frames (1.7%). 40 m/s is a physical bound, not fitted.
- **Ball velocity** from 5-frame moving-average positions inside visible runs, centred difference
  ±2 frames. The provided speed field is ignored (populated on 8.7% of frames).
- **Architecture.** Per-frame team log-odds `e` (exclusive proximity + last-touch bias while the
  ball is free or missing) and dead-ball log-odds `z`, decoded by a 3-state Viterbi with constant
  switch penalties, each half independently. The last-touch bias makes the Viterbi break ties
  towards placing a change of possession on the first frame of the new team (GT convention, fact 12).
- **No minimum duration** anywhere (fact 9).
- **Second-half rule.** Frames from the half's first graded frame to the published kickoff
  (113,742) are forced D; the kickoff frame is forced alive. Same rule for external matches using
  their own kickoff frames.
- **No X on graded frames** (spellF1 forward-fills X; macroF1 charges it).
- **v0** (untuned defaults in params.py): 242 possessions, 97 dead stretches, dead share 0.31,
  mean possession 18 s, too long against the sanity range of 10–15 s. Public-window numbers are
  not recorded here as results.

## Disagreements with the spec

- None of substance so far. `scipy` turns out to be installed (the spec says it is not).

## 2026-11-05 (later): external validation and learned evidence

- **External data.** IDSSE/DFL 7 matches (CC BY 4.0) and Metrica Sample Games 1–2 (public,
  "acknowledge the source"). Metrica Game 3 skipped for now: its EPTS file changes the player
  channel layout at every substitution segment, so it needs a segment-aware parser.
- **IDSSE labels** = per-frame `BallStatus` / `BallPossession` from the DFL position file.
  Measured: median possession 3–4.7 s, over half under 4 s, dead share 42–52%. That is choppier
  and more dead than BAR-ATH's public window (median 7.6 s, about 20% dead), most likely
  because DFL credits contacts more freely than Metrica's "controlled" possession. Consequence:
  tuning targets the mean over IDSSE and Metrica, and BAR-ATH counts are checked against the
  sanity ranges.
- **Metrica labels** derived from events by rule (`loaders.metrica_labels`): owner = team of
  the last PASS/RECOVERY/SHOT/SET PIECE/BALL LOST/CARRY/BALL OUT; dead from a stoppage (ball
  out, shot out or goal, foul, offside) to the next restart (set piece or goal kick), cancelled
  if a live event comes first. Metrica's ball is untracked almost exactly while dead (ball
  visible on 1.2% of dead frames), so Metrica is never used to *train* the dead-ball model.
- **Noise simulator, first version was wrong.** AR(1) jitter with rho 0.9 added about 1 m/s of
  fake player speed. Measured: the ATD is smoother frame to frame than the clean sources (median
  second difference 1.4 mm vs 10 mm for IDSSE), and clean external speeds already match BAR-ATH
  (mean 1.82 / 1.73 vs 1.70 m/s; standing fraction 0.17 / 0.20 vs 0.22). Fixed: drift-only
  jitter (rho 0.999). Also, IDSSE keeps the ball tracked off the pitch (16% of frames) where the
  ATD loses it (4.2% of visible frames): off-pitch runs are now hidden to match. With the old
  simulator the dead-ball model put 54% of BAR-ATH graded time in D; after the fix, 38%.
- **Learned evidence, all scored out-of-fold** (two folds over the 9 external matches):
  - `teamgeom` (player-only MLP): 85–89% frame accuracy on held-out IDSSE, also on missing-ball
    frames. External spellF1 0.232 → 0.300 at w_geom 0.2 (before the other models).
  - `deadnet` (ball + kinematics MLP, IDSSE only): spellF1 0.300 → 0.436.
  - `teamnet` (geometry + ball MLP): 87–89% IDSSE, 82–85% Metrica; spellF1 0.446 → 0.516.
- **Pseudo-label adaptation (spec 3.4), tried and dropped.** External test: retrain the team
  model on the other fold's GT plus the match's own pipeline labels on visible-ball frames.
  Mean spellF1 0.5154 → 0.5149 over the 9 matches. No gain, so the submission's models are
  trained on external labels only (`possession/validation/pseudo_test.py`).
- **Smart Tagging not used for now.** Neither its set-piece windows nor its phase team tags
  can be validated externally (the external matches have no tags), and the spec forbids fitting
  their weights on the public window. `w_tag` stays 0.

## 2026-11-05 (evening): tuning

- **First tuning objective was IDSSE-heavy** (4 IDSSE : 1 Metrica). Its parameters scored 0.553
  externally but pushed BAR-ATH to 420 possessions and 43% dead (DFL semantics). Saved as
  `cache/tuned_round0_idsse_heavy.json`, not used.
- **Source-balanced objective** (0.5 × IDSSE mean + 0.5 × Metrica), started from those values,
  one round of coordinate descent. The second round was stopped: objective +0.007 while the
  validation score fell from 0.594 to 0.590 (overfitting at the noise level). Values are in
  `possession/params.py` = `possession/validation/tuned.json`.
- **External results with these values** (out-of-fold models): mean spellF1 0.562 over the 9
  tuning matches (IDSSE 0.540, Metrica 1–2 0.636). **Untouched held-out Metrica Game 3:
  spellF1 0.665, macroF1 0.845.** These are scores on degraded external tracking, not a
  BAR-ATH estimate.
- **BAR-ATH build** (`evaluation/results/solution.csv`, sha256 b9553932…): 349 possessions, 125
  dead stretches, mean possession 10.7 s, home share 0.55. All in range except **dead share
  0.415** (range 0.25–0.34). The same bias shows on Metrica: predicted 0.39–0.40 against labelled
  0.26–0.28, with the right number of stretches, so dead stretches are too *long*. Most likely the
  DFL-trained dead-ball model carries DFL's long-dead convention. Open issue, next step.
- **Determinism:** two builds produce byte-identical files.
