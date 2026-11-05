"""Dead-ball evidence: per-frame log-odds `z` (positive = ball not in play).

Independent soft signals, summed:
  * ball tracked outside the pitch (beyond a margin),
  * ball at rest across a window,
  * team kinematic lull (smoothed mean player speed below a level),
  * ball missing (weak prior only),
  * ball moving inside the pitch (evidence for alive),
  * optional set-piece tag window prior.
"""

import numpy as np


def _centred_mean(x, win):
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    h = win // 2
    lo = np.maximum(0, i - h)
    hi = np.minimum(len(x), i + h + 1)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def out_of_pitch(feat, pitch, margin):
    L, W = pitch
    b = feat.ball
    with np.errstate(invalid="ignore"):
        out = (b[:, 0] < -margin) | (b[:, 0] > L + margin) | (b[:, 1] < -margin) | \
              (b[:, 1] > W + margin)
    return out & feat.vis


def dead_evidence(feat, match, p, tag_prior=None):
    z = np.full(match.n, p["d_bias"], dtype=np.float64)
    out = out_of_pitch(feat, match.pitch, p["out_margin"])
    z += p["w_out"] * out

    slow = np.where(feat.vis & ~np.isnan(feat.ball_speed),
                    (feat.ball_speed < p["rest_v"]).astype(float), np.nan)
    rest_frac = _centred_mean(slow, p["rest_win"])
    z += p["w_rest"] * np.nan_to_num(rest_frac > 0.9).astype(float) * feat.vis

    ms = _centred_mean(feat.mean_speed, p["lull_win"])
    lull = np.clip(p["lull_v"] - np.nan_to_num(ms, nan=p["lull_v"]), -2.0, 2.0)
    z += p["w_lull"] * lull

    z += p["w_miss"] * (~feat.vis)
    with np.errstate(invalid="ignore"):
        moving = feat.vis & (feat.ball_speed > p["move_v"]) & ~out
    z -= p["w_move"] * moving
    if tag_prior is not None:
        z += p["w_tag"] * tag_prior
    return z


def tag_prior(match):
    """1 inside set-piece tag windows (restart likely somewhere inside), 0 elsewhere."""
    t = np.zeros(match.n)
    for tag in match.tags:
        if tag["code"] == "SET PIECES":
            t[max(0, tag["start"]):min(match.n, tag["end"])] = 1.0
    return t
