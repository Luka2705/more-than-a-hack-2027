"""Degrade clean external tracking into ATD-like tracking.

Every statistic comes from the provided game's unlabelled tracking (atd_stats.json, made by
measure_atd.py), never from its ground truth:
  * ball visibility as an alternating renewal process, visible-run and gap lengths resampled
    from the BAR-ATH empirical distributions (median gap ≈ 10.5 s, median visible run ≈ 13 s);
  * a fraction of gaps (gap_edge_frac, 37%) snapped to start where the ball is near a pitch
    edge, as observed on BAR-ATH;
  * slow AR(1) positional drift on ball and players (sigma 0.2 m, rho 0.999): the ATD is
    smoother frame to frame than the clean sources (median second difference 1.4 mm for
    players vs 10 mm in IDSSE), so no high-frequency noise is added;
  * off-pitch ball runs hidden (after OUT_LAG frames) with the probability that brings the
    off-pitch share of visible ball frames down to BAR-ATH's (out_of_pitch_frac_visible);
  * player dropouts (renewal on/off per slot) to reach BAR-ATH's mean of 20.5 tracked players;
  * no speed field (the pipeline never reads one).
Seeded per match, so a given (match, seed) is always degraded identically.
"""

import json
import os
import zlib

import numpy as np

from ..data import Match

STATS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atd_stats.json")
JITTER_SIGMA = 0.2   # m, slow positional drift (calibration error), not frame noise
JITTER_RHO = 0.999   # per-frame AR(1) correlation: the ATD is smooth frame to frame
OUT_LAG = 12         # frames the ATD keeps the ball after it leaves the pitch
EDGE_FRAC = 0.05     # of pitch size, "near the edge"
SNAP_WINDOW = 250    # frames searched for an edge frame when snapping a gap start


def _ar1(rng, shape, sigma, rho):
    out = np.empty(shape, np.float32)
    noise = rng.normal(0, sigma * np.sqrt(1 - rho ** 2), shape).astype(np.float32)
    out[0] = rng.normal(0, sigma, shape[1:])
    for t in range(1, shape[0]):
        out[t] = rho * out[t - 1] + noise[t]
    return out


def _ball_mask(rng, ball, halves, stats, L, W):
    n = len(ball)
    vis = np.zeros(n, bool)
    vr = np.array(stats["vis_runs"])
    gr = np.array(stats["gap_runs"])
    with np.errstate(invalid="ignore"):
        edge = ((ball[:, 0] < EDGE_FRAC * L) | (ball[:, 0] > (1 - EDGE_FRAC) * L) |
                (ball[:, 1] < EDGE_FRAC * W) | (ball[:, 1] > (1 - EDGE_FRAC) * W))
    edge_idx = np.flatnonzero(edge)
    for _, a, b in halves:
        t = a
        visible = True
        while t <= b:
            if visible:
                length = int(rng.choice(vr))
                end = t + length
                if rng.random() < stats["gap_edge_frac"] and len(edge_idx):
                    j = np.searchsorted(edge_idx, end)
                    cands = edge_idx[max(0, j - 3):j + 3]
                    cands = cands[np.abs(cands - end) <= SNAP_WINDOW]
                    if len(cands):
                        end = int(cands[np.argmin(np.abs(cands - end))]) + 1
                end = max(end, t + 1)
                vis[t:min(end, b + 1)] = True
                t = end
            else:
                t += int(rng.choice(gr))
            visible = not visible
    return vis


def _hide_out(rng, ball, vis, target, L, W):
    with np.errstate(invalid="ignore"):
        out = (ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W)
    cur = (out & vis).sum() / max(1, vis.sum())
    keep = np.ones(len(ball), bool)
    if cur <= target:
        return keep
    p_hide = 1 - target / cur
    edges = np.diff(np.concatenate(([0], out.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.random() < p_hide:
            keep[min(a + OUT_LAG, b):b] = False
    return keep


def degrade(match, seed=0):
    stats = json.load(open(STATS))
    rng = np.random.default_rng([seed, zlib.crc32(match.name.encode())])
    L, W = match.pitch
    n, P = match.n, match.players.shape[1]

    ball = match.ball.copy()
    ball += _ar1(rng, ball.shape, JITTER_SIGMA, JITTER_RHO)
    vis = _ball_mask(rng, match.ball, match.halves, stats, L, W)
    vis &= _hide_out(rng, match.ball, vis, stats["out_of_pitch_frac_visible"], L, W)
    ball[~vis] = np.nan

    pl = match.players.copy()
    pl += _ar1(rng, pl.shape, JITTER_SIGMA, JITTER_RHO)
    present = ~np.isnan(pl[:, :, 0])
    on_field = present.sum(1)
    target = stats["players_mean"]
    mean_on = on_field[on_field > 15].mean() if (on_field > 15).any() else 22.0
    keep = min(1.0, target / mean_on)
    # renewal dropouts: mean off-duration 3 s, on-duration chosen to give the keep fraction
    off_mean = 75.0
    on_mean = off_mean * keep / max(1e-6, 1 - keep)
    drop = np.zeros((n, P), bool)
    for j in range(P):
        t = int(rng.integers(0, int(on_mean) + 1))
        while t < n:
            d = int(rng.exponential(off_mean)) + 1
            drop[t:t + d, j] = True
            t += d + int(rng.exponential(on_mean)) + 1
    pl[drop] = np.nan
    return Match(name=match.name + "_sim%d" % seed, ball=ball, players=pl, team=match.team,
                 halves=match.halves, kickoffs=match.kickoffs, pitch=match.pitch, fps=match.fps,
                 tags=[])
