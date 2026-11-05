"""External test of pseudo-label adaptation (spec 3.4): does training the team classifier on a
match's own pipeline output (visible-ball alive frames) improve spellF1 on that match?

    python -m possession.validation.pseudo_test
"""
import os

import numpy as np

from .. import pipeline, teamgeom, teamnet
from . import harness

P = {"w_team": 0.2, "w_dnet": 0.4, "w_hand": 0.0, "w_geom": 0.1}


def main():
    names = harness.TRAIN + harness.VALID + harness.HELDOUT
    preps = harness.load_all(names)
    feats = {n: teamnet.team_features(preps[n][0].match, preps[n][0].feat) for n in names}
    rows = []
    for k, fold in enumerate(harness.FOLDS):
        others = [n for f in harness.FOLDS if f is not fold for n in f]
        Xo, yo = [], []
        for n in others:
            alive, y = teamgeom.labels_from_gt(preps[n][1])
            Xo.append(feats[n][alive])
            yo.append(y[alive])
        for n in fold:
            prep, gt = preps[n]
            base_lab = pipeline.label(prep, P)
            base = pipeline.score(base_lab, gt)["spell_f1_mean"]
            sel = np.isin(base_lab, ["H", "A"]) & prep.feat.vis
            mp = os.path.join(harness.CACHE, "pseudo_%s.pt" % n)
            teamgeom.train(Xo + [feats[n][sel]], yo + [(base_lab[sel] == "H").astype(np.float32)],
                           out=mp)
            old = prep.team
            prep.team = teamgeom.predict(feats[n], mp)
            adapt = pipeline.score(pipeline.label(prep, P), gt)["spell_f1_mean"]
            prep.team = old
            rows.append((n, base, adapt))
            print("%-24s base %.3f  adapted %.3f" % (n, base, adapt), flush=True)
    b = np.mean([r[1] for r in rows])
    a = np.mean([r[2] for r in rows])
    print("mean base %.4f adapted %.4f" % (b, a))


if __name__ == "__main__":
    main()
