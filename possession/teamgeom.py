"""Player-only team-in-possession evidence, for the 41% of frames without a ball.

Hand-built collective features (no ball), expressed in the home team's attacking direction:
centroids, spreads, depth of the deepest / highest player, collective velocity, speeds, and the
same quantities smoothed over ±1 s and differenced over 2 s. A small MLP maps them to the
log-odds that home is in possession.

Training labels: alive frames of the external matches (IDSSE + Metrica ground truth) and/or
pseudo-labels from this pipeline's own touch-based labeller on BAR-ATH visible-ball frames
(algorithmic, no human involved). Which mix is used is decided on external validation, see
notes/decisions.md. Weights are committed (possession/models/teamgeom.pt) so the build is
reproducible without retraining; `python -m possession.teamgeom train` regenerates them.
"""

import os
import sys

import numpy as np

from . import features as F

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "teamgeom.pt")
SMOOTH = 25
DELTA = 50
SEED = 0


def _cm(x, win):
    from .deadball import _centred_mean
    return np.stack([_centred_mean(x[:, j], win) for j in range(x.shape[1])], axis=1)


def geom_features(match, feat):
    n = match.n
    pl = match.players
    team = match.team
    sgn = np.zeros(n, np.float32)
    for h, a, b in match.halves:
        sgn[a:b + 1] = feat.attack_dir[h]
    L = match.pitch[0]
    x = (pl[:, :, 0] - L / 2) * sgn[:, None]  # home attacks +x
    y = pl[:, :, 1] - match.pitch[1] / 2
    vx = F.centred_velocity(pl, match.fps)[:, :, 0] * sgn[:, None]
    cols = []
    with np.errstate(invalid="ignore", all="ignore"):
        for k in (0, 1):
            msk = (team == k)[None, :]
            xk = np.where(msk, x, np.nan)
            yk = np.where(msk, y, np.nan)
            vk = np.where(msk & (np.abs(vx) < 12), vx, np.nan)
            cols += [np.nanmean(xk, 1), np.nanstd(xk, 1), np.nanstd(np.abs(yk), 1),
                     np.nanstd(yk, 1), np.nanmin(xk, 1), np.nanmax(xk, 1),
                     np.nanmean(vk, 1), np.nanmean(np.abs(vk), 1),
                     np.nanmean(xk > 0, 1)]
            # second-deepest / second-highest (deepest is usually the keeper)
            srt = np.sort(np.where(np.isnan(xk), np.inf, xk), axis=1)
            cols.append(np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan))
            srt2 = np.sort(np.where(np.isnan(xk), -np.inf, xk), axis=1)
            cols.append(np.where(np.isfinite(srt2[:, -2]), srt2[:, -2], np.nan))
    base = np.stack(cols, axis=1).astype(np.float64)
    base = np.concatenate([base, base[:, :11] - base[:, 11:]], axis=1)  # home minus away
    sm = _cm(base, SMOOTH)
    d = np.full_like(sm, np.nan)
    d[DELTA // 2:n - DELTA // 2] = sm[DELTA:] - sm[:-DELTA]
    X = np.concatenate([sm, d], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X


def _net(nf):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(nf, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, 1))


def train(Xs, ys, epochs=6, out=MODEL, stride=5):
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    X = np.concatenate([x[::stride] for x in Xs])
    y = np.concatenate([v[::stride] for v in ys]).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd)
    yt = torch.tensor(y)
    net = _net(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print("  epoch %d loss %.4f" % (ep, tot / len(Xt)), flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "nf": X.shape[1]}, out)
    return out


def predict(X, path=MODEL):
    import torch
    torch.set_num_threads(1)
    ck = torch.load(path, weights_only=False)
    net = _net(ck["nf"])
    net.load_state_dict(ck["state"])
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor((X - ck["mu"]) / ck["sd"])).squeeze(1).numpy()
    return out.astype(np.float64)


def evidence_for(match, feat=None, path=MODEL):
    if not os.path.isfile(path):
        return None
    feat = feat or F.compute(match)
    return predict(geom_features(match, feat), path)


def labels_from_gt(gt):
    alive = np.isin(gt, ["H", "A"])
    return alive, (gt == "H").astype(np.float32)


if __name__ == "__main__":
    from .validation import harness
    if sys.argv[1:2] == ["train"]:
        names = harness.TRAIN + harness.VALID + harness.HELDOUT  # final model: all external
        Xs, ys = [], []
        for nme in names:
            prep, gt = harness.prepare(nme)
            X = geom_features(prep.match, prep.feat)
            alive, y = labels_from_gt(gt)
            Xs.append(X[alive])
            ys.append(y[alive])
            print(nme, alive.sum(), flush=True)
        train(Xs, ys)
    # evaluate frame accuracy on validation matches
    for nme in harness.VALID + ["metrica_2"]:
        prep, gt = harness.prepare(nme)
        lo = predict(geom_features(prep.match, prep.feat))
        alive, y = labels_from_gt(gt)
        acc = ((lo > 0) == (y > 0.5))[alive].mean()
        miss = alive & ~prep.feat.vis
        print("%-24s acc alive %.3f  acc on missing-ball %.3f" % (
            nme, acc, ((lo > 0) == (y > 0.5))[miss].mean() if miss.any() else float("nan")))
