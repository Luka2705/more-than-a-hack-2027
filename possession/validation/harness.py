"""Score the pipeline on degraded external matches with the official spellF1.

    python -m possession.validation.harness            # prepare (cached) and score defaults

Split: IDSSE matches are assigned to train/validation by a fixed list; Metrica games are held
out entirely (different labelling route, used as a second opinion).
"""

import os
import pickle
import sys

import numpy as np

from .. import pipeline, report
from . import loaders, simulate

CACHE = os.path.join(loaders.CACHE, "prepared")
TRAIN = ["idsse_DFL-MAT-J03WMX", "idsse_DFL-MAT-J03WN1", "idsse_DFL-MAT-J03WOH",
         "idsse_DFL-MAT-J03WOY"]
VALID = ["idsse_DFL-MAT-J03WPY", "idsse_DFL-MAT-J03WQQ", "idsse_DFL-MAT-J03WR9"]
HELDOUT = ["metrica_1", "metrica_2"]
SEEDS = (0,)


def prepare(name, seed=0, geom_fn=None):
    """(Prepared, gt) for one external match, degraded with `seed`. Cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "%s_s%d.pkl" % (name, seed))
    if os.path.isfile(path):
        prep, gt = pickle.load(open(path, "rb"))
    else:
        if name.startswith("idsse_"):
            m, gt = loaders.load_idsse(name[len("idsse_"):])
            sim = simulate.degrade(m, seed)
        else:
            m, gt = loaders.load_metrica(int(name.split("_")[1]))
            sim = simulate.degrade(m, seed)
            # Metrica's ball is already genuinely missing ~40% of frames: keep its own mask
            # rather than stacking simulated gaps on top.
            keep = m.ball_valid
            noise = np.random.default_rng(seed).normal(0, simulate.JITTER_SIGMA,
                                                       (int(keep.sum()), 2))
            sim.ball[:] = np.nan
            sim.ball[keep] = m.ball[keep] + noise.astype(np.float32)
        prep = pipeline.Prepared(sim)
        pickle.dump((prep, gt), open(path, "wb"), protocol=4)
    if geom_fn is not None:
        prep.geom = geom_fn(prep)
    return prep, gt


def evaluate(names, params=None, preps=None, verbose=False):
    rows = []
    for name in names:
        prep, gt = preps[name] if preps else prepare(name)
        lab = pipeline.label(prep, params)
        r = pipeline.score(lab, gt)
        blocks = [(a, b) for _, a, b in prep.match.halves]
        d = report.diagnose(lab, blocks)
        rows.append({"name": name, "spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                     "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
                     "dead_pred": d["dead_stretches"],
                     "dead_gt": report.diagnose(gt, blocks)["dead_stretches"]})
        if verbose:
            print("  %-24s spell %.3f macro %.3f  poss %d/%d  dead %d/%d" % (
                name, rows[-1]["spell"], rows[-1]["macro"], rows[-1]["n_pred"],
                rows[-1]["n_gt"], rows[-1]["dead_pred"], rows[-1]["dead_gt"]))
    mean = float(np.mean([r["spell"] for r in rows]))
    return mean, rows


FOLDS = [TRAIN[:2] + VALID[:1] + HELDOUT[:1], TRAIN[2:] + VALID[1:] + HELDOUT[1:]]


def geom_oof(names, seed=0):
    """Out-of-fold player-only logits for every external match: each match is scored by a
    model trained on the other fold only, so tuning w_geom never sees in-sample predictions."""
    from .. import teamgeom
    path = os.path.join(CACHE, "geom_oof_s%d.pkl" % seed)
    if os.path.isfile(path):
        return pickle.load(open(path, "rb"))
    out = {}
    for k, fold in enumerate(FOLDS):
        others = [n for f in FOLDS if f is not fold for n in f]
        Xs, ys = [], []
        for n in others:
            prep, gt = prepare(n, seed)
            X = teamgeom.geom_features(prep.match, prep.feat)
            alive, y = teamgeom.labels_from_gt(gt)
            Xs.append(X[alive])
            ys.append(y[alive])
        mp = os.path.join(CACHE, "geom_fold%d.pt" % k)
        teamgeom.train(Xs, ys, out=mp)
        for n in fold:
            prep, _ = prepare(n, seed)
            out[n] = teamgeom.predict(teamgeom.geom_features(prep.match, prep.feat), mp)
    pickle.dump(out, open(path, "wb"))
    return out


def dead_oof(names, seed=0):
    """Out-of-fold dead-ball logits; trained on the IDSSE matches of the other fold only."""
    from .. import deadnet, teamgeom
    path = os.path.join(CACHE, "dead_oof_s%d.pkl" % seed)
    if os.path.isfile(path):
        return pickle.load(open(path, "rb"))
    out = {}
    for k, fold in enumerate(FOLDS):
        others = [n for f in FOLDS if f is not fold for n in f if n.startswith("idsse")]
        Xs, ys = [], []
        for n in others:
            prep, gt = prepare(n, seed)
            g, y = deadnet.labels_from_gt(gt)
            Xs.append(deadnet.dead_features(prep.match, prep.feat)[g])
            ys.append(y[g])
        mp = os.path.join(CACHE, "dead_fold%d.pt" % k)
        teamgeom.train(Xs, ys, out=mp)
        for n in fold:
            prep, _ = prepare(n, seed)
            out[n] = teamgeom.predict(deadnet.dead_features(prep.match, prep.feat), mp)
    pickle.dump(out, open(path, "wb"))
    return out


def team_oof(names, seed=0):
    """Out-of-fold geometry+ball team logits."""
    from .. import teamgeom, teamnet
    path = os.path.join(CACHE, "team_oof_s%d.pkl" % seed)
    if os.path.isfile(path):
        return pickle.load(open(path, "rb"))
    out = {}
    for k, fold in enumerate(FOLDS):
        others = [n for f in FOLDS if f is not fold for n in f]
        Xs, ys = [], []
        for n in others:
            prep, gt = prepare(n, seed)
            alive, y = teamgeom.labels_from_gt(gt)
            Xs.append(teamnet.team_features(prep.match, prep.feat)[alive])
            ys.append(y[alive])
        mp = os.path.join(CACHE, "team_fold%d.pt" % k)
        teamgeom.train(Xs, ys, out=mp)
        for n in fold:
            prep, _ = prepare(n, seed)
            out[n] = teamgeom.predict(teamnet.team_features(prep.match, prep.feat), mp)
    pickle.dump(out, open(path, "wb"))
    return out


def load_all(names, seed=0, geom=True, dead=True, team=True):
    preps = {n: prepare(n, seed) for n in names}
    if geom:
        g = geom_oof(list(preps), seed)
        for n, (prep, _) in preps.items():
            prep.geom = g.get(n)
    if dead:
        d = dead_oof(list(preps), seed)
        for n, (prep, _) in preps.items():
            prep.dead = d.get(n)
    if team:
        t = team_oof(list(preps), seed)
        for n, (prep, _) in preps.items():
            prep.team = t.get(n)
    return preps


if __name__ == "__main__":
    names = TRAIN + VALID + HELDOUT
    preps = load_all(names)
    for group, ns in (("train", TRAIN), ("valid", VALID), ("heldout", HELDOUT)):
        mean, _ = evaluate(ns, preps=preps, verbose=True)
        print("%s mean spellF1 %.3f" % (group, mean))
    sys.exit(0)
