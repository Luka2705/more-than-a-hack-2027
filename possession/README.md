# `possession/` — frame-level possession labelling for BAR-ATH

Evidence detectors (ball proximity, learned team and dead-ball classifiers) fused per frame
and decoded globally by a three-state (H, A, D) Viterbi. Method, data and validation:
[`docs/TECHNICAL_REPORT.md`](../docs/TECHNICAL_REPORT.md). Decision log and hyperparameters
with provenance: [`notes/decisions.md`](../notes/decisions.md).

## Reproduce

From a fresh clone, Python 3.10, CPU only.

1. Install the pinned dependencies:

   ```bash
   pip install -r requirements-solution.txt
   ```

2. Unpack the match bundle into `data/` as described in `DATA.md` (or set `MTAH_DATA_DIR`).

3. Build the submission. The first run parses the 196 MB ATD into `cache/` (about 4 minutes);
   later runs take seconds. The model weights it needs are committed under `possession/models/`.

   ```bash
   python -m possession.build --out evaluation/results/<team_name>.csv
   ```

4. Validate and inspect it:

   ```bash
   python evaluation/validate_submission.py evaluation/results/<team_name>.csv
   ```

   ```bash
   python -m possession.report evaluation/results/<team_name>.csv
   ```

   ```bash
   python evaluation/run_evaluation.py
   ```

The build is deterministic: no randomness at inference, and the file is byte-identical across
runs (the build prints its SHA-256).

### Regenerating the models and the tuned parameters (optional)

Only needed to re-derive what is committed. Everything here uses external matches; nothing
uses BAR-ATH labels.

1. Download the external data into `external/` (gitignored):

   ```bash
   git clone --depth 1 https://github.com/metrica-sports/sample-data.git external/metrica
   ```

   ```bash
   python -m possession.validation.download_idsse
   ```

2. Parse IDSSE (about 1 minute with 6 processes), and measure the ATD statistics that drive
   the noise simulator:

   ```bash
   python -m possession.validation.loaders parse
   ```

   ```bash
   python -m possession.validation.measure_atd
   ```

3. Train the three models on all external matches:

   ```bash
   python -m possession.teamgeom train
   ```

   ```bash
   python -m possession.deadnet train
   ```

   ```bash
   python -m possession.teamnet train
   ```

4. Score on the degraded external matches (out-of-fold models), and re-tune:

   ```bash
   python -m possession.validation.harness
   ```

   ```bash
   python -m possession.validation.tune --rounds 3
   ```

Torch training is seeded and single-threaded. Bit-identical weights across different CPUs or
torch builds are not guaranteed, which is why the weights the submission uses are committed.

## Layout

| module | role |
|---|---|
| `data.py` | ATD + Smart Tagging parse, ball outlier gate, `.npz` cache |
| `features.py` | ball velocity, nearest-player distances, player speeds, team shape |
| `touches.py` | hand-built proximity / last-touch team evidence |
| `teamgeom.py` | player-only team classifier (for missing-ball frames) |
| `teamnet.py` | geometry + ball team classifier |
| `deadball.py` | hand-built dead-ball evidence |
| `deadnet.py` | learned dead-ball classifier |
| `decode.py` | three-state Viterbi, per half |
| `pipeline.py` | match → labels, shared by build and validation |
| `build.py` | the entry point |
| `report.py` | diagnostics and sanity ranges |
| `validation/` | external loaders, noise simulator, harness, tuner |
