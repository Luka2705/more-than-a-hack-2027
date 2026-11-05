"""Measure statistics of the provided game's *unlabelled* automatic tracking, for the noise
simulator. Writes possession/validation/atd_stats.json (histograms and rates only; nothing
frame-level, no ground truth involved).

    python -m possession.validation.measure_atd
"""
import json
import os

import numpy as np

from ..data import load_match
from ..features import _runs

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atd_stats.json")


def main():
    m = load_match()
    L, W = m.pitch
    vis_runs, gap_runs, gap_edge = [], [], []
    present = []
    for _, a, b in m.halves:
        a = max(a, m.kickoffs.get(_, a))
        v = m.ball_valid[a:b + 1]
        s, e = _runs(v)
        vis_runs += (e - s).tolist()
        gs, ge = _runs(~v)
        for x, y in zip(gs, ge):
            if x == 0 or y == len(v):
                continue
            gap_runs.append(int(y - x))
            last = m.ball[a + x - 1]
            near = (last[0] < 0.05 * L or last[0] > 0.95 * L or last[1] < 0.05 * W
                    or last[1] > 0.95 * W)
            gap_edge.append(bool(near))
        present.append((~np.isnan(m.players[a:b + 1, :, 0])).sum(1))
    present = np.concatenate(present)
    vis_all = np.concatenate([m.ball_valid[a:b + 1] for _, a, b in m.halves])
    ball_all = np.concatenate([m.ball[a:b + 1] for _, a, b in m.halves])
    with np.errstate(invalid="ignore"):
        outp = ((ball_all[:, 0] < 0) | (ball_all[:, 0] > L) | (ball_all[:, 1] < 0) |
                (ball_all[:, 1] > W))[vis_all]
    stats = {
        "vis_runs": sorted(int(x) for x in vis_runs),
        "gap_runs": sorted(gap_runs),
        "gap_edge_frac": float(np.mean(gap_edge)),
        "players_mean": float(present.mean()),
        "out_of_pitch_frac_visible": float(outp.mean()),
        "players_hist": np.bincount(present, minlength=30).tolist(),
        "ball_missing_frac": float(1 - np.mean([m.ball_valid[a:b + 1].mean()
                                                for _, a, b in m.halves])),
    }
    json.dump(stats, open(OUT, "w"))
    print({k: v for k, v in stats.items() if not isinstance(v, list)},
          "n_gaps", len(gap_runs), "median gap", np.median(gap_runs),
          "median vis", np.median(vis_runs))


if __name__ == "__main__":
    main()
