"""Learned team-in-possession evidence from player geometry plus ball proximity.

Extends teamgeom's player-only features with ball terms: visibility, nearest home / away player
distance (and their difference) at the frame and smoothed over ±0.5 s / ±2 s, the last-touch team
(forward-filled exclusive control, signed) and the time since it. Trained on alive frames of the
external matches; weights at possession/models/teamnet.pt; `python -m possession.teamnet train`.
The hand-built proximity term in touches.py is kept alongside it for frame-sharp boundaries.
"""

import os
import sys

import numpy as np

from . import teamgeom, touches
from .deadball import _centred_mean
from .params import DEFAULT

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "teamnet.pt")


def ball_features(match, feat):
    vis = feat.vis.astype(float)
    d0 = np.where(feat.vis & np.isfinite(feat.dist[:, 0]), np.clip(feat.dist[:, 0], 0, 30), np.nan)
    d1 = np.where(feat.vis & np.isfinite(feat.dist[:, 1]), np.clip(feat.dist[:, 1], 0, 30), np.nan)
    dd = d1 - d0
    c = touches.control_scores(feat, DEFAULT)
    lt = np.zeros(match.n)
    since = np.zeros(match.n)
    for _, a, b in match.halves:
        l, own = touches.last_touch(c[a:b + 1])
        lt[a:b + 1] = np.where(l == 0, 1.0, np.where(l == 1, -1.0, 0.0))
        idx = np.where(own >= 0, np.arange(len(own)), -1)
        idx = np.maximum.accumulate(idx)
        since[a:b + 1] = np.where(idx >= 0, np.arange(len(own)) - idx, 1000)
    cols = [vis, np.nan_to_num(d0, nan=30), np.nan_to_num(d1, nan=30), np.nan_to_num(dd, nan=0)]
    for w in (25, 101):
        cols += [_centred_mean(dd, w), _centred_mean(lt, w)]
    cols += [lt, np.log1p(np.minimum(since, 1000)) / 7.0]
    X = np.stack(cols, axis=1)
    return np.nan_to_num(X, nan=0.0).astype(np.float32)


def team_features(match, feat):
    return np.concatenate([teamgeom.geom_features(match, feat), ball_features(match, feat)], 1)


def evidence_for(match, feat, path=MODEL):
    if not os.path.isfile(path):
        return None
    return teamgeom.predict(team_features(match, feat), path)


if __name__ == "__main__":
    from .validation import harness
    if sys.argv[1:2] == ["train"]:
        Xs, ys = [], []
        for nme in harness.TRAIN + harness.VALID + harness.HELDOUT:
            prep, gt = harness.prepare(nme)
            alive, y = teamgeom.labels_from_gt(gt)
            Xs.append(team_features(prep.match, prep.feat)[alive])
            ys.append(y[alive])
        teamgeom.train(Xs, ys, out=MODEL)
