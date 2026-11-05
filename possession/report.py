"""Diagnostics for a submission file: counts, durations, dead-ball share, sanity ranges.

    python -m possession.report evaluation/results/<team>.csv
"""

import sys

import numpy as np
import pandas as pd

GRADED = ((11185, 90651), (110808, 190197))
FPS = 25
SANITY = {"possessions": (250, 400), "dead_stretches": (80, 130), "mean_poss_s": (10, 15),
          "dead_share": (0.25, 0.34)}


def runs(lab):
    br = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    b = np.r_[0, br, len(lab)]
    return [(int(b[i]), int(b[i + 1]), lab[b[i]]) for i in range(len(b) - 1)]


def diagnose(labels, blocks=GRADED, fps=FPS):
    labels = np.asarray(labels)
    out = {"per_half": []}
    poss, dead, n_trans, frames = [], [], 0, 0
    home = away = 0
    for a, b in blocks:
        lab = labels[a:b + 1].copy()
        # spellF1 view: forward-fill X
        s = pd.Series(np.where(lab == "X", None, lab)).ffill().bfill().values
        rs = runs(s)
        p = [(e - s0) / fps for s0, e, k in rs if k in ("H", "A")]
        d = [(e - s0) / fps for s0, e, k in rs if k == "D"]
        out["per_half"].append({"possessions": len(p), "dead_stretches": len(d)})
        poss += p
        dead += d
        n_trans += len(rs) - 1
        frames += len(s)
        home += int((s == "H").sum())
        away += int((s == "A").sum())
    out["possessions"] = len(poss)
    out["dead_stretches"] = len(dead)
    out["mean_poss_s"] = float(np.mean(poss)) if poss else 0.0
    out["median_poss_s"] = float(np.median(poss)) if poss else 0.0
    out["poss_under_4s"] = int(np.sum(np.array(poss) < 4))
    out["mean_dead_s"] = float(np.mean(dead)) if dead else 0.0
    out["dead_share"] = float(np.sum(dead) * fps / frames)
    out["transitions_per_min"] = n_trans / (frames / fps / 60)
    out["home_share"] = home / max(home + away, 1)
    out["poss_hist_s"] = np.histogram(poss, bins=[0, 2, 4, 8, 15, 30, 60, 1e9])[0].tolist()
    return out


def sanity_flags(d):
    return {k: (lo <= d[k] <= hi) for k, (lo, hi) in SANITY.items()}


def main(path):
    lab = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values
    d = diagnose(lab)
    for k, v in d.items():
        print("%-20s %s" % (k, v))
    flags = sanity_flags(d)
    print("sanity:", ", ".join("%s %s" % (k, "ok" if ok else "OUT OF RANGE")
                               for k, ok in flags.items()))


if __name__ == "__main__":
    main(sys.argv[1])
