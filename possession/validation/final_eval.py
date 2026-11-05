"""Final external report with the committed parameters (params.DEFAULT).

  * the 9 tuning matches, out-of-fold models (the models never saw the match they score);
  * metrica_3, an untouched held-out match: never used to train a model or choose a value.
    Its evidence averages the two fold models, neither of which saw it.

    python -m possession.validation.final_eval
"""

import os

import numpy as np

from .. import deadnet, pipeline, report, teamgeom, teamnet
from ..params import DEFAULT
from . import harness

FINAL_HELDOUT = ["metrica_3"]


def prepare_final(name):
    prep, gt = harness.prepare(name)
    folds = range(len(harness.FOLDS))
    X = teamgeom.geom_features(prep.match, prep.feat)
    prep.geom = np.mean([teamgeom.predict(X, os.path.join(harness.CACHE, "geom_fold%d.pt" % k))
                         for k in folds], axis=0)
    X = deadnet.dead_features(prep.match, prep.feat)
    prep.dead = np.mean([teamgeom.predict(X, os.path.join(harness.CACHE, "dead_fold%d.pt" % k))
                         for k in folds], axis=0)
    X = teamnet.team_features(prep.match, prep.feat)
    prep.team = np.mean([teamgeom.predict(X, os.path.join(harness.CACHE, "team_fold%d.pt" % k))
                         for k in folds], axis=0)
    return prep, gt


def row(name, prep, gt, p):
    lab = pipeline.label(prep, p)
    r = pipeline.score(lab, gt)
    blocks = [(a, b) for _, a, b in prep.match.halves]
    d, dg = report.diagnose(lab, blocks), report.diagnose(gt, blocks)
    print("%-22s spellF1 %.3f  macroF1 %.3f  poss %3d/%3d  dead %3d/%3d  dead share %.2f/%.2f" % (
        name, r["spell_f1_mean"], r["macro_f1"], d["possessions"], dg["possessions"],
        d["dead_stretches"], dg["dead_stretches"], d["dead_share"], dg["dead_share"]))
    return r["spell_f1_mean"], r["macro_f1"]


def main(p=None):
    p = p or DEFAULT
    names = harness.TRAIN + harness.VALID + harness.HELDOUT
    preps = harness.load_all(names)
    print("tuning matches (out-of-fold models):")
    res = {n: row(n, *preps[n], p) for n in names}
    ids = [res[n][0] for n in names if n.startswith("idsse")]
    met = [res[n][0] for n in names if n.startswith("metrica")]
    print("mean spellF1: all %.3f  idsse %.3f  metrica %.3f" % (
        np.mean([v[0] for v in res.values()]), np.mean(ids), np.mean(met)))
    print("untouched held-out:")
    for n in FINAL_HELDOUT:
        row(n, *prepare_final(n), p)


if __name__ == "__main__":
    main()
