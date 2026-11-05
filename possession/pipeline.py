"""Match -> labels. The same function runs on BAR-ATH and on every external validation match."""

import os
import sys

import numpy as np

from . import deadball, decode, features, touches
from .params import merged

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402


class Prepared:
    """Features plus anything expensive that does not depend on tunable parameters."""

    def __init__(self, match, geom=None, dead=None):
        self.match = match
        self.feat = features.compute(match)
        self.tag = deadball.tag_prior(match) if match.tags else None
        self.geom = geom
        self.dead = dead
        self.team = None


def label(prep, params=None):
    p = merged(params)
    m = prep.match
    e = touches.team_evidence(prep.feat, p, geom=prep.geom, halves=m.halves)
    team = getattr(prep, "team", None)
    if team is not None:
        e = e + p["w_team"] * team
    z = p["w_hand"] * deadball.dead_evidence(prep.feat, m, p, tag_prior=prep.tag)
    dead = getattr(prep, "dead", None)
    if dead is not None:
        z = z + p["w_dnet"] * (dead + p["dnet_bias"])
    return decode.decode_match(m, e, z, p)


def score(labels, gt_labels):
    """Official metrics (possession_metrics) for label arrays of letters."""
    gt = pm.label_codes(np.asarray(gt_labels))
    pred = pm.label_codes(np.asarray(labels))
    r = pm.evaluate(gt, pred)
    return r


def summary(r):
    keys = ("spell_f1_mean", "macro_f1")
    return {k: r.get(k) for k in keys}
