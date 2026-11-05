"""Single reproducible entry point.

    python -m possession.build --out evaluation/results/<team_name>.csv

Deterministic: no randomness at inference, fixed seeds wherever a model is trained, no
wall-clock dependence. Regenerates the identical file from the data bundle.
"""

import argparse
import hashlib
import os

import numpy as np

from . import pipeline, report
from .data import load_match


def write(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(0)
    match = load_match()
    from . import deadnet, teamgeom, teamnet
    prep = pipeline.Prepared(match)
    prep.geom = teamgeom.evidence_for(match, prep.feat)
    prep.dead = deadnet.evidence_for(match, prep.feat)
    prep.team = teamnet.evidence_for(match, prep.feat)
    if prep.geom is None or prep.dead is None or prep.team is None:
        raise SystemExit("model weights missing under possession/models/")
    labels = pipeline.label(prep)
    write(args.out, labels)
    with open(args.out, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    print("wrote %s  sha256 %s" % (args.out, digest))
    d = report.diagnose(labels)
    print("possessions %d  dead stretches %d  dead share %.3f  mean poss %.1fs" % (
        d["possessions"], d["dead_stretches"], d["dead_share"], d["mean_poss_s"]))


if __name__ == "__main__":
    main()
