"""Per-frame kinematic features shared by every detector.

All quantities are in metres and m/s. Works on any `Match` (the provided game or an external one),
so the detectors see the same inputs in validation and in the build.
"""

from dataclasses import dataclass

import numpy as np

# Params
BALL_SMOOTH = 5        # frames, moving average of ball positions inside each visible run
DIFF_HALF = 2          # frames, half-width of the centred difference for velocities
PLAYER_VMAX = 12.0     # m/s, slot speeds above this are identity swaps, not running
STILL_SPEED = 0.6      # m/s, a player slower than this counts as standing


@dataclass
class Features:
    ball: np.ndarray          # [N,2] smoothed ball position, NaN when missing
    vis: np.ndarray           # [N] bool
    ball_v: np.ndarray        # [N,2] m/s, NaN when not computable
    ball_speed: np.ndarray    # [N]
    dist: np.ndarray          # [N,2] nearest home / away player distance to ball (inf if none)
    near_slot: np.ndarray     # [N,2] slot index of that player (-1 if none)
    pspeed: np.ndarray        # [N,P] per-slot speed, NaN when not computable
    team_speed: np.ndarray    # [N,2] mean speed per team
    mean_speed: np.ndarray    # [N] mean speed all players
    still_frac: np.ndarray    # [N] fraction of players standing
    n_players: np.ndarray     # [N] tracked players
    centroid: np.ndarray      # [N,2,2] team centroid (x,y)
    spread: np.ndarray        # [N,2,2] team std (x,y)
    attack_dir: dict          # {half: +1/-1} direction home attacks along x
    half_of: np.ndarray       # [N] half index or 0 if outside


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def smooth_runs(x, vis, k):
    """Moving average of width k inside each contiguous visible run (shrinking at the edges)."""
    out = np.full_like(x, np.nan)
    starts, stops = _runs(vis)
    h = k // 2
    for a, b in zip(starts, stops):
        seg = x[a:b].astype(np.float64)
        c = np.vstack([np.zeros((1, seg.shape[1])), np.cumsum(seg, axis=0)])
        i = np.arange(b - a)
        lo = np.maximum(0, i - h)
        hi = np.minimum(b - a, i + h + 1)
        out[a:b] = (c[hi] - c[lo]) / (hi - lo)[:, None]
    return out


def centred_velocity(pos, fps, half=DIFF_HALF):
    """(p[t+h]-p[t-h])/(2h/fps); falls back to one-sided where a neighbour is missing."""
    n = len(pos)
    v = np.full_like(pos, np.nan)
    for h in range(half, 0, -1):
        fwd = np.full_like(pos, np.nan)
        bwd = np.full_like(pos, np.nan)
        fwd[:n - h] = pos[h:]
        bwd[h:] = pos[:n - h]
        cand = (fwd - bwd) * fps / (2 * h)
        fill = np.isnan(v[..., 0]) & ~np.isnan(cand[..., 0])
        v[fill] = cand[fill]
    one = np.full_like(pos, np.nan)
    one[1:] = (pos[1:] - pos[:-1]) * fps
    fill = np.isnan(v[..., 0]) & ~np.isnan(one[..., 0])
    v[fill] = one[fill]
    back = np.full_like(pos, np.nan)
    back[:-1] = (pos[1:] - pos[:-1]) * fps
    fill = np.isnan(v[..., 0]) & ~np.isnan(back[..., 0])
    v[fill] = back[fill]
    return v


def compute(match):
    fps = match.fps
    n = match.n
    vis = match.ball_valid
    ball = smooth_runs(match.ball, vis, BALL_SMOOTH).astype(np.float32)
    ball_v = centred_velocity(ball, fps)
    ball_speed = np.linalg.norm(ball_v, axis=1)

    pl = match.players
    team = match.team
    dist = np.full((n, 2), np.inf, dtype=np.float32)
    near = np.full((n, 2), -1, dtype=np.int16)
    chunk = 20000
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = np.linalg.norm(pl[a:b] - ball[a:b, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        for k in (0, 1):
            cols = np.flatnonzero(team == k)
            sub = d[:, cols]
            j = np.argmin(sub, axis=1)
            dk = sub[np.arange(b - a), j]
            dist[a:b, k] = dk
            near[a:b, k] = np.where(np.isfinite(dk), cols[j], -1)

    pv = centred_velocity(pl, fps)
    pspeed = np.linalg.norm(pv, axis=2)
    pspeed[pspeed > PLAYER_VMAX] = np.nan
    present = ~np.isnan(pl[:, :, 0])
    n_players = present.sum(axis=1)

    with np.errstate(invalid="ignore"):
        team_speed = np.stack([np.nanmean(np.where(team[None] == k, pspeed, np.nan), axis=1)
                               for k in (0, 1)], axis=1)
        mean_speed = np.nanmean(pspeed, axis=1)
        valid_sp = ~np.isnan(pspeed)
        still_frac = np.where(valid_sp.sum(1) > 0,
                              (np.nan_to_num(pspeed, nan=99) < STILL_SPEED).sum(1)
                              / np.maximum(valid_sp.sum(1), 1), np.nan)
        centroid = np.stack([np.nanmean(np.where((team[None] == k)[..., None], pl, np.nan), axis=1)
                             for k in (0, 1)], axis=1)
        spread = np.stack([np.nanstd(np.where((team[None] == k)[..., None], pl, np.nan), axis=1)
                           for k in (0, 1)], axis=1)

    half_of = np.zeros(n, dtype=np.int8)
    attack_dir = {}
    for h, a, b in match.halves:
        half_of[a:b + 1] = h
        cx = centroid[a:b + 1, :, 0]
        diff = np.nanmean(cx[:, 1] - cx[:, 0])  # away minus home
        attack_dir[h] = 1 if diff > 0 else -1   # home attacks towards the away half

    return Features(ball=ball, vis=vis, ball_v=ball_v, ball_speed=ball_speed, dist=dist,
                    near_slot=near, pspeed=pspeed, team_speed=team_speed, mean_speed=mean_speed,
                    still_frac=still_frac, n_players=n_players, centroid=centroid, spread=spread,
                    attack_dir=attack_dir, half_of=half_of)
