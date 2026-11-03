# Import-Export — Local evidence fusion for ball possession

## Objective and approach

We produce one label per tactical-video frame: `H` (home possession), `A` (away
possession), `D` (dead ball), or `X` (outside the published half intervals). The
method combines geometric ball-control evidence, automatic tactical phase tags,
and restart/stoppage evidence in a three-state sequence decoder. It is a
deterministic, training-free engineering solution; it is not a fitted model.

A nearest-player decision alone confuses a passing ball or a brief challenge with
controlled possession. Independently labelling each frame also fragments
possessions. We instead integrate evidence over time and charge for changing
state. Unlike the reference baseline, missing ball tracking is not itself evidence
for a dead ball.

## Inputs and frame alignment

We use only three locally stored, automatically generated tactical assets:

- FIFA/EPTS metadata: team order, team membership of the anonymous track slots.
- Raw tactical ATD: player coordinates and speeds, and ball coordinates.
- SportsCode Smart Tagging: automatic phase and set-piece intervals with team tags.

ATD frame `i` maps to submission frame `i - 1`. Tag seconds map using the supplied
25 fps loader. Coordinates are converted to metres with the pitch dimensions in
the supplied configuration. The parser rejects duplicate, missing, extra, or
out-of-order frame records and malformed slot counts. No roster identity is
inferred from slot numbers.

We use the published half boundaries only to define the output's evaluated time
domain and reset temporal inference. We do not use the documented second-half
kickoff timestamp, goals, match reports, manually recorded events, or public
possession labels as inference inputs. Kickoff/restart estimates come from the
automatic tags and tracking.

Neither video is opened. Homography and broadcast tracking are not used: tactical
tracking is already in pitch coordinates, while the broadcast feed has a different
timeline and replays. These inputs were not used for visual validation or model
selection either.

## Evidence construction

**Ball quality.** An isolated ball point is rejected when both adjacent velocities
are implausible but the velocity skipping that point is plausible. Interior gaps
of at most 0.4 s are linearly interpolated only if the endpoint displacement is
physically plausible. Longer gaps and leading/trailing gaps remain missing. The
two halves are processed separately. Interpolation does not establish possession.

**Control.** For each team, we calculate the minimum player-ball distance in metres
and transform it with a Gaussian proximity kernel of radius 2.5 m. The difference
between the two team signals reduces confidence when opponents are similarly close.
Fast ball movement attenuates control evidence, helping preserve possession while
a pass travels past opponents. A short centred filter reduces observation noise.

**Automatic phases.** Build up, progression, attacking transitions, long balls and
ball-location attacking phases support the tagged team. Defensive phases support
the opposing team. Edges taper over one second because tag windows are approximate.
Overlapping correlated tags are max-pooled per team, not repeatedly counted.
Unknown teams and `N/A` provide no team vote. `PLAYERS IN THE BOX` and `GOALS` do not
define possession and are ignored. Team strings are case/diacritic normalised,
including the documented Athletic/Atletic spelling variation.

**Dead-ball evidence.** A tracked ball outside the pitch, beyond a 0.6 m tolerance,
supports `D`. Sustained stationary-ball evidence contributes only with collective
low player movement; standing still alone can be a live goalkeeper possession.
Missing tracking alone does not contribute dead-ball evidence.

**Restarts.** A set-piece interval supplies a search window. Stationary-to-moving
ball evidence estimates the restart; if it is absent, the interval midpoint is an
explicit uncertain fallback. A bounded backward search uses the last visible
moving in-pitch ball to propose the preceding stoppage, requiring corroborating
stationary/out-of-pitch evidence or a lack of active phase tags. This is a
heuristic, not an inferred referee decision. Automatic start-of-half kickoff tags
mark the preceding portion of that half as dead. A known restarting team supplies
a brief post-restart possession cue; `N/A` does not invent a team.

## Temporal decision rule

The states are `H`, `A`, `D`. Each frame receives a cost for each state. Strong dead
evidence suppresses team evidence; bounded team confidence prevents duplicate tags
from overriding clear stoppages. When every observation is missing, costs are
neutral rather than forcing a switch.

Dynamic programming finds the minimum of:

`sum(frame_cost / fps) + sum(state_change_cost)`.

A team-to-team switch costs 1.0 evidence-seconds; a switch to/from dead costs 0.65.
Ties prefer staying in the existing state. This suppresses short isolated touches
without a hard minimum possession length. A sustained new team's evidence can
still create a short possession. A dead interval splits two possessions even when
the same team restarts. The calculation uses future observations (offline
inference); it is not intended as a live broadcast estimator.

## Parameters, training data, and rule clarification

**No training data, pretrained model, training, fitting, calibration, or parameter
search was used.** Parameters were specified as engineering defaults before the
first full-match run. They were not selected from target-match output statistics,
public-ground-truth performance, or video inspection. No external dataset was
downloaded or used. The synthetic tests are software invariant tests and are not a
training or parameter selection dataset.

Important defaults are in `solution/config.py`: control radius 2.5 m, ball-speed
ceiling 45 m/s, interpolation limit 0.4 s, velocity/evidence filter radius 0.2 s,
moving threshold 2 m/s, stationary threshold 0.7 m/s, stationary window 2 s,
boundary tolerance 0.6 m, walking threshold 1.2 m/s, phase/control weights 0.8/1.2,
restart window margin 2 s, restart lookback 45 s and team seed 1 s. Additional fixed
confidence thresholds and formulas are explicit in `solution/infer.py`. These
values are design assumptions, not claims of empirically optimal settings.

The organisers' subsequent clarification takes precedence over older repository
wording. Match records, clips and extracts were not transmitted to AI services by
this workflow. AI assistance was used to implement code from the public task and
format documentation; restricted data was processed by local Python only. No
predictions were checked against video, and no manual corrections or human-created
match labels were used. All future training or tuning must use external data with
appropriate rights and separately documented provenance.

## Validation and reproducibility

The implementation includes synthetic tests for frame alignment, missing data,
impossible ball jumps, defensive-tag inversion, duplicate-tag handling, team-swap
symmetry, brief touches, genuine control changes, stoppage splitting, and exact
agreement of the dynamic-programming solver with exhaustive search on a small
fixture. These establish software behavior, not match accuracy.

`solution.verify` checks the official file contract, published half boundaries and
source/output SHA-256 hashes. With `--reproduce`, it regenerates the file and
requires byte identity and identical input hashes while blocking network and
ground-truth/video file access. The generation manifest records input/source
hashes, configuration and runtime versions. Intermediate evidence arrays and
restart estimates can be saved locally with `--intermediates`.

All 26 synthetic tests passed. Full-match output passed the official format
contract and byte-identical reproduction check. One frozen submission was then
processed successfully by the official public five-minute evaluator as a separate
pipeline check. Its report remains local; it was not sent to the AI assistant for
analysis, and no parameters or predictions were changed based on that report.
No score-based variant selection was performed. No measured improvement over the
full-match baseline or qualification to the next round is claimed.

Tested reproduction environment: Python **3.14.3** with exactly pinned packages
in `requirements-solution.txt`. From the repository root:

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-solution.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m solution.infer --data-dir /path/to/data --intermediates
.venv/bin/python -m solution.verify --data-dir /path/to/data --reproduce
```

Output: `artifacts/Import-Export.csv`, exactly 199,319 rows plus `frame,label`.
The match data is supplied separately by the organisers. No download service,
credentials, GPU, video decoder, or network is needed for inference.

## Limitations

- Long ball occlusions hide turnovers. Conflicting tags may preserve the wrong team.
- Tracking slots can switch or duplicate, and fast/airborne balls can be mislocated.
- Restarts are approximate windows, and fallback midpoints may be temporally wrong.
- Stationary-ball heuristics cannot reliably distinguish every foul, goalkeeper
  hold, slow buildup, or medical stoppage. Ball movement during a stoppage can make
  backward search underestimate its duration.
- Missing/incorrect automatic set-piece tags can merge or split possession spells.
- Fixed defaults have not been externally calibrated or demonstrated to beat the
  baseline. A next development phase would evaluate and tune on separately licensed
  external tracking matches with compatible possession and dead-ball labels.
