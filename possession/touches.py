"""Team evidence from the ball and the nearest players.

Output is a per-frame log-odds `e` (positive = home in possession), built from:
  * exclusive proximity: one team within control range of the ball and the other not,
  * last-touch inheritance: while the ball is free (in flight) or missing, a small bias towards
    the team that last had it exclusively, so that the decoder places a change of possession on
    the first frame of the new team rather than somewhere inside the flight.
A single opponent touch gives only a few frames of evidence, which the decoder's switch penalty
absorbs; sustained control (a reception, a carry, consecutive touches) does not.
"""

import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def control_scores(feat, p):
    d = np.where(np.isfinite(feat.dist), feat.dist, 1e3)
    c = _sig((p["ctrl_r"] - d) / p["ctrl_s"])
    c[~feat.vis] = 0.0
    return c  # [N,2]


def last_touch(c, thresh=0.5):
    """Forward-filled team of the last frame with exclusive control (-1 before any)."""
    own = np.full(len(c), -1, dtype=np.int8)
    own[(c[:, 0] > thresh) & (c[:, 1] <= thresh)] = 0
    own[(c[:, 1] > thresh) & (c[:, 0] <= thresh)] = 1
    idx = np.where(own >= 0, np.arange(len(own)), -1)
    idx = np.maximum.accumulate(idx)
    lt = np.where(idx >= 0, own[np.maximum(idx, 0)], -1)
    return lt, own


def team_evidence(feat, p, geom=None, halves=None):
    c = control_scores(feat, p)
    e = p["w_near"] * (c[:, 0] - c[:, 1])
    lt, own = last_touch(c)
    if halves is not None:  # last touch never carries across a half boundary
        for _, a, b in halves:
            seg = slice(a, b + 1)
            lt_h, _ = last_touch(c[seg])
            lt[seg] = lt_h
    sign = np.where(lt == 0, 1.0, np.where(lt == 1, -1.0, 0.0))
    free = feat.vis & (own < 0)
    e = e + np.where(free, p["w_flight"] * sign, 0.0)
    e = e + np.where(~feat.vis, p["w_gap"] * sign, 0.0)
    if geom is not None:
        e = e + np.where(~feat.vis, p["w_geom"], p["w_geom_vis"]) * geom
    return e
