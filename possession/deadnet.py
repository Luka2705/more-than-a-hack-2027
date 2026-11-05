"""Learned dead-ball evidence: an MLP on ball + collective-kinematics features -> log-odds dead.

Features (all computed from the degraded tracking the pipeline sees):
  ball visible, visibility fraction over ±1 s / ±4 s, ball out of the pitch, ball distance to the
  nearest line, ball speed and at-rest fraction over windows, ball-to-nearest-player distance,
  mean player speed and fraction standing over 1 / 3 / 6 s windows, the change of mean speed
  over the next and previous 2 s (re-acceleration at a restart, deceleration at a stoppage),
  team spreads, tracked-player count.
Trained on the external matches' dead/alive labels (IDSSE BallStatus; Metrica excluded from
training because its ball is never tracked while dead, which would teach "missing = dead").
Weights committed at possession/models/deadnet.pt; `python -m possession.deadnet train`.
"""

import os
import sys

import numpy as np

from .deadball import _centred_mean
from . import teamgeom

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "deadnet.pt")


def dead_features(match, feat):
    n = match.n
    L, W = match.pitch
    vis = feat.vis.astype(float)
    b = feat.ball
    with np.errstate(invalid="ignore"):
        line = np.minimum.reduce([b[:, 0], L - b[:, 0], b[:, 1], W - b[:, 1]])
        out = (line < 0).astype(float)
        bs = np.nan_to_num(np.clip(feat.ball_speed, 0, 40), nan=0.0)
        rest = np.where(feat.vis, (np.nan_to_num(feat.ball_speed, nan=0) < 0.7), np.nan)
        near = np.minimum(np.where(np.isfinite(feat.dist[:, 0]), feat.dist[:, 0], 50),
                          np.where(np.isfinite(feat.dist[:, 1]), feat.dist[:, 1], 50))
    line = np.where(feat.vis, np.clip(np.nan_to_num(line, nan=0), -5, 30), 0.0)
    near = np.where(feat.vis, np.clip(near, 0, 30), 0.0)
    ms = np.nan_to_num(feat.mean_speed, nan=0.0)
    sf = np.nan_to_num(feat.still_frac, nan=0.0)
    cols = [vis, _centred_mean(vis, 51), _centred_mean(vis, 201), out, _centred_mean(out, 51),
            line, bs, _centred_mean(bs, 25), _centred_mean(rest, 37), _centred_mean(rest, 101),
            near, _centred_mean(near, 25)]
    for w in (25, 75, 151):
        cols += [_centred_mean(ms, w), _centred_mean(sf, w)]
    m1 = _centred_mean(ms, 25)
    fwd = np.full(n, np.nan)
    bwd = np.full(n, np.nan)
    fwd[:-50] = m1[50:] - m1[:-50]
    bwd[50:] = m1[50:] - m1[:-50]
    cols += [fwd, bwd]
    for k in (0, 1):
        cols += [_centred_mean(np.nan_to_num(feat.spread[:, k, 0]), 25),
                 _centred_mean(np.nan_to_num(feat.spread[:, k, 1]), 25)]
    cols.append(feat.n_players.astype(float))
    X = np.stack(cols, axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def labels_from_gt(gt):
    graded = gt != "X"
    return graded, (gt == "D").astype(np.float32)


def evidence_for(match, feat, path=MODEL):
    if not os.path.isfile(path):
        return None
    return teamgeom.predict(dead_features(match, feat), path)


if __name__ == "__main__":
    from .validation import harness
    names = [n for n in harness.TRAIN + harness.VALID if n.startswith("idsse")]
    if sys.argv[1:2] == ["train"]:
        Xs, ys = [], []
        for nme in names:
            prep, gt = harness.prepare(nme)
            g, y = labels_from_gt(gt)
            Xs.append(dead_features(prep.match, prep.feat)[g])
            ys.append(y[g])
        teamgeom.train(Xs, ys, out=MODEL)
