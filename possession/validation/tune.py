"""Coordinate-descent tuning of params.py values on external matches (never on BAR-ATH).

    python -m possession.validation.tune [--rounds 2] [--keys k1,k2]

Objective: spellF1 averaged with equal weight per *source*: 0.5 x mean(IDSSE train matches)
+ 0.5 x metrica_1. IDSSE labels are choppier and more often dead than BAR-ATH's (see
notes/decisions.md); a per-match mean (4:1) pulled the decoder towards DFL semantics. The
VALID set (remaining IDSSE + metrica_2, same weighting) is reported at every accepted step.
metrica_3 is never used here.
Deterministic: the candidate grid is fixed and evaluation has no randomness.
"""

import argparse
import json
import os

import numpy as np

from ..params import DEFAULT
from . import harness

OBJ = harness.TRAIN + ["metrica_1"]
VAL = harness.VALID + ["metrica_2"]
FACTORS = (0.0, 0.25, 0.5, 0.75, 1.33, 2.0, 4.0)
INT_KEYS = {"rest_win", "lull_win"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned.json")


ADDITIVE = {"dnet_bias", "d_bias"}


def candidates(key, value):
    if key in ADDITIVE:
        return [value + d for d in (-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0)]
    if value == 0:
        base = {"w_geom": 0.1, "w_geom_vis": 0.05, "w_tag": 0.2, "w_team": 0.2}.get(key, 0.1)
        vals = [0.0, base * 0.5, base, base * 2, base * 4]
    else:
        vals = [value * f for f in FACTORS]
    if key in INT_KEYS:
        vals = sorted(set(max(1, int(round(v))) for v in vals))
    return vals


def balanced(names, p, preps):
    _, rows = harness.evaluate(names, p, preps)
    ids = [r["spell"] for r in rows if r["name"].startswith("idsse")]
    met = [r["spell"] for r in rows if r["name"].startswith("metrica")]
    return 0.5 * float(np.mean(ids)) + 0.5 * float(np.mean(met)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--keys", default=None)
    ap.add_argument("--start", default=None, help="json of starting params")
    args = ap.parse_args()
    preps = harness.load_all(OBJ + VAL)
    p = dict(DEFAULT)
    p.update({"w_geom": 0.1, "w_dnet": 0.4, "w_hand": 0.5, "w_team": 0.2})
    if args.start:
        p.update(json.load(open(args.start)))
    keys = args.keys.split(",") if args.keys else list(DEFAULT)
    best, _ = balanced(OBJ, p, preps)
    print("start obj %.4f valid %.4f" % (best, balanced(VAL, p, preps)[0]))
    for rnd in range(args.rounds):
        for k in keys:
            for v in candidates(k, p[k]):
                if v == p[k]:
                    continue
                q = dict(p)
                q[k] = v
                s, _ = balanced(OBJ, q, preps)
                if s > best + 1e-4:
                    best, p = s, q
                    val = balanced(VAL, p, preps)[0]
                    print("r%d %-12s = %-8.4g obj %.4f valid %.4f" % (rnd, k, v, best, val),
                          flush=True)
        json.dump(p, open(OUT, "w"), indent=1, sort_keys=True)
    print("final", json.dumps(p, sort_keys=True))


if __name__ == "__main__":
    main()
