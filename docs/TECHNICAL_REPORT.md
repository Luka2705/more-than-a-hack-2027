# Possession labelling for BAR-ATH: technical report

More Than A Hack 2027, Phase 1. Code: `possession/`. Decision log and hyperparameters with
provenance: `notes/decisions.md`. Reproduction: `possession/README.md`.

<!-- RESULTS -->

## 1. Approach

The task asks for one label per frame, but it is scored on **possessions as objects**
(spellF1: maximal H or A runs matched by temporal IoU). So the system is built as a
segmenter, not a per-frame classifier:

1. **Per-frame evidence** from independent detectors, each a log-odds:
   - *team evidence* `e` (positive = home in possession):
     a hand-built proximity term (one team within control range of the ball and the other
     not), a last-touch bias while the ball is free or missing, and two learned classifiers:
     `teamgeom` (player positions only) and `teamnet` (player positions plus ball proximity);
   - *dead-ball evidence* `z`: a learned classifier `deadnet` on ball and collective-kinematics
     features, plus hand-built terms (ball off the pitch, ball at rest, team lull, ball missing,
     ball moving).
2. **Global decoding** by a three-state (H, A, D) Viterbi over each half, with fixed penalties
   for H↔A, alive→D and D→alive. Emissions are H = +e/2, A = −e/2, D = z.

There is no duration model and no minimum possession length. A possession survives if the
evidence accumulated over it beats the switch penalty, so short spells with real control
evidence are kept, while a single deflection or duel gives only a few frames of opposing
evidence, which the penalty absorbs. The last-touch bias makes the Viterbi place a change of
possession on the first frame of the new team rather than mid-flight, which matches the
ground-truth convention (transition at the new owner's first frame).

Why this rather than smoothing: a majority-vote window places boundaries at window midpoints
and deletes short possessions; a 1 s boundary error alone drops a perfect file to 0.65 spellF1.
Why learned evidence: the ball is missing on 41% of graded frames, in long gaps (median 10.7 s),
so ball interpolation recovers almost nothing; collective player geometry is the only signal
there, and a classifier on it reaches about 0.9 frame accuracy out-of-match.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | everything: ball, 46 slots with team membership |
| organiser-published half blocks and kickoff frames | half boundaries; second half pre-kickoff forced D |
| homography | not used: the ATD is already in pitch coordinates |
| Smart Tagging | **not used** (see below) |
| broadcast video / ATD | not used (unsynchronised; alignment would cost more than it returns here) |
| tactical video | not used (no GPU; a ball detector is not trainable on this machine) |
| ATD speed field | ignored (populated on 8.7% of visible frames); speeds derived from positions |

**Smart Tagging, tried on paper and dropped.** Set-piece windows are 35 s long and cover about
half of the restarts; phase `Team` tags have code-dependent meaning (some name the defending
team, some are coin flips). Neither can be validated on the external matches, which carry no
tags, and the rules forbid fitting their weights on the public window. With no honest way to
set a weight, the weight is 0.

## 3. External training and validation data

No label of any kind on BAR-ATH is used for training, tuning or model selection. The public
5-minute window was used only to check the wiring.

| dataset | source | licence | labels and their origin | volume |
|---|---|---|---|---|
| IDSSE / DFL | Bassek et al., Sci. Data 2025, figshare 10.6084/m9.figshare.28196177 | CC BY 4.0 | per-frame `BallStatus` and `BallPossession` from DFL's positional data (official league data) | 7 Bundesliga matches, 25 fps, ≈ 1.0 M frames |
| Metrica Sports sample data | github.com/metrica-sports/sample-data | public, "acknowledge the source" | derived by rule from Metrica's event data (below) | 3 matches, 25 fps, ≈ 0.43 M frames |

Metrica rule: the owner is the team of the last PASS / RECOVERY / SHOT / SET PIECE / BALL LOST /
CARRY / BALL OUT event started at or before the frame; challenges and fouls never change the
owner. Dead from a stoppage (ball out, shot out or goal, foul, offside) to the next restart
(set piece or goal kick), cancelled if a live event comes first; dead before each kickoff.

**Label semantics differ between sources.** IDSSE possessions are much shorter (median 3–4.7 s,
over half under 4 s) and the ball is dead 42–52% of the time. Metrica-derived labels give a
median of about 5 s and 23–28% dead. The public BAR-ATH window shows a median of 7.6 s and about
20% dead. The tuning objective therefore averages IDSSE and Metrica matches, and BAR-ATH outputs
are checked against sanity ranges (section 6).

**Noise simulator.** Clean external tracking is degraded to look like the ATD, using only
statistics of BAR-ATH's *unlabelled* tracking (`possession/validation/atd_stats.json`):
- ball visibility as an alternating renewal process with visible-run and gap lengths resampled
  from BAR-ATH;
- 37% of gaps snapped to start near a pitch edge, as on BAR-ATH;
- off-pitch ball runs hidden to match BAR-ATH's 4.2% off-pitch share of visible frames;
- slow positional drift (σ 0.2 m, ρ 0.999). The ATD is smoother frame to frame than the clean
  sources, so no frame noise is added. A first version with ρ = 0.9 inflated player speeds by
  about 1 m/s and made the dead-ball model put 54% of BAR-ATH in D; the measured fix is logged;
- player dropouts down to BAR-ATH's mean of 20.5 tracked players.

Metrica's ball is already untracked about 40% of the time, almost exactly while dead, so for
Metrica its own mask is kept and Metrica is never used to train the dead-ball model.

**Protocol.** Two folds over the 9 tuning matches (IDSSE ×7, Metrica 1–2). Every learned model
used in validation scores a match it was not trained on (out-of-fold). Hyperparameters are set by
coordinate descent on the official `spell_f1_mean` over one subset (4 IDSSE + Metrica 1), with
the rest reported at each step to expose overfitting. Metrica Game 3 was added last and kept
**untouched**: never trained on, never used to choose anything. The submission's models are
retrained on all 9 tuning matches.

<!-- EXTERNAL_TABLE -->

## 4. Possession rules as implemented

- **Control.** A team has control evidence when one of its players is within about the
  control radius of the ball and no opponent is. Two players of opposite teams both close (a
  duel) cancel out.
- **Loose balls, passes in flight.** No change: the last team with exclusive control keeps the
  ball (a small last-touch bias), as the challenge definition requires.
- **Deflections, failed tackles, aerial duels.** A few frames of opposing evidence, below the
  H↔A penalty, so no change unless the opponent then shows sustained control. There is no
  duration floor: a reception followed by a pass is enough if the evidence is clear.
- **Missing ball.** Team evidence comes from player geometry alone (`teamgeom`, `teamnet`): who
  is stretched and pushing forward, who is compact and retreating.
- **Dead ball.** `D` when the dead-ball evidence (ball off the pitch, ball at rest, players
  slowing and standing, missing ball as a weak prior) beats the alive→D penalty. A false D in
  the middle of a possession splits it in two (about two missed matches), a missed D merges two
  possessions (about one), and the penalties are tuned on spellF1 directly, which prices both.
- **Restarts and throw-in team.** The restart ends D where the evidence turns alive (ball moving,
  players re-accelerating); the restarting team is whichever team the team evidence names right
  after it, i.e. the team at the ball when it starts moving. No tag is needed.
- **Second half.** Frames 110,808 to 113,741 are D (published kickoff 113,742); the kickoff frame is
  alive. Each half is decoded independently.
- **No `X` on graded frames.** spellF1 forward-fills `X`, and macroF1 charges it.

<!-- LIMITATIONS -->

## 7. Reproduction

See `possession/README.md`. In short:

```bash
pip install -r requirements-solution.txt
python -m possession.build --out evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```
