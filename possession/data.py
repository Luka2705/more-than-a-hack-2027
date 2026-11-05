"""Data layer: parse the tactical ATD and Smart Tagging once, cache as arrays.

Everything downstream works on a `Match` object holding metre-scaled numpy arrays indexed by
video frame (ATD frame_id - 1). The same structure is produced by the external-match loaders in
possession/validation/, so every detector runs unchanged on both.
"""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
FPS = 25
MAX_BALL_SPEED = 40.0  # m/s; anything faster between two visible frames is a tracking jump

# Organiser-published constants (DATA.md, FAQ.md, starter/load_smart_tagging.py).
KICKOFF_FRAMES = {1: 11185, 2: 113742}


@dataclass
class Match:
    """Per-frame arrays in metres, indexed by frame 0..N-1.

    ball      [N,2] float32, NaN where not visible (after the outlier gate)
    players   [N,P,2] float32, NaN where a slot is empty
    team      [P] int8, 0 home, 1 away
    halves    [(half, first, last)] inclusive graded frame blocks
    kickoffs  {half: frame} first live frame of each half
    tags      list of dict (code, start, end, team code 0/1/-1, type, half)
    pitch     (length, width) metres
    """
    name: str
    ball: np.ndarray
    players: np.ndarray
    team: np.ndarray
    halves: list
    kickoffs: dict
    pitch: tuple
    fps: int = FPS
    tags: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.ball)

    @property
    def ball_valid(self):
        return ~np.isnan(self.ball[:, 0])


# -- ATD -----------------------------------------------------------------------------------------

def team_of_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]  # FIFA/EPTS: first team listed is home (Barcelona here, DATA.md)
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse_atd(raw_path, n_slots):
    """Whole-file vectorised parse. Line: frame:x,y,s;x,y,s;...:bx,by,bs"""
    with open(raw_path, "r") as fh:
        text = fh.read()
    text = text.strip().replace(":", ",").replace(";", ",").replace("\r", "").replace("\n", ",")
    width = 1 + 3 * n_slots + 3
    flat = np.fromstring(text, sep=",", dtype=np.float64)
    del text
    if flat.size % width:
        raise ValueError("ATD line width mismatch: %d values, width %d" % (flat.size, width))
    arr = flat.reshape(-1, width)
    frame_id = arr[:, 0].astype(np.int64)
    if not np.array_equal(frame_id, np.arange(1, len(arr) + 1)):
        raise ValueError("ATD frame ids are not 1..N contiguous")
    players = arr[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = arr[:, 1 + 3 * n_slots:]
    return players[:, :, :2].astype(np.float32), ball[:, :2].astype(np.float32), \
        players[:, :, 2].astype(np.float32), ball[:, 2].astype(np.float32)


def gate_ball(ball_m, fps=FPS, vmax=MAX_BALL_SPEED):
    """NaN out isolated ball positions implying an impossible jump from both neighbours."""
    ball = ball_m.copy()
    vis = ~np.isnan(ball[:, 0])
    idx = np.flatnonzero(vis)
    if len(idx) < 3:
        return ball
    p = ball[idx]
    dt = np.diff(idx) / fps
    sp = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    bad_prev = np.concatenate(([False], sp > vmax))
    bad_next = np.concatenate((sp > vmax, [False]))
    # A point is an outlier when both its links are impossible (spike), or it is an isolated
    # single-frame detection with an impossible link on either side.
    single = np.concatenate(([True], np.diff(idx) > 1)) & np.concatenate((np.diff(idx) > 1, [True]))
    bad = (bad_prev & bad_next) | (single & (bad_prev | bad_next))
    ball[idx[bad]] = np.nan
    return ball


# -- Smart Tagging -------------------------------------------------------------------------------

TEAM_CODES = {"barcelona": 0, "atletic club": 1, "athletic club": 1}


def parse_tags(path, fps=FPS):
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        team = TEAM_CODES.get((tags.get("Team") or "").strip().lower(), -1)
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * fps)),
            "end": int(round(float(inst.findtext("end")) * fps)),
            "team": team,
            "type": (tags.get("Type") or "").strip().lower(),
            "half": tags.get("Half"),
        })
    out.sort(key=lambda t: (t["start"], t["end"], t["code"]))
    return out


# -- loading with cache ----------------------------------------------------------------------------

def load_match(game=GAME, data_dir=None, use_cache=True):
    meta_path, raw_path, tag_path = require(["atd_meta", "atd_raw", "smart_tagging"], game,
                                            data_dir)
    length, width = GAMES[game]["pitch"]
    team = team_of_slots(meta_path)
    cache = os.path.join(CACHE_DIR, "%s_tactical.npz" % game)
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        players_n, ball_n = z["players"], z["ball"]
    else:
        players_n, ball_n, _, _ = parse_atd(raw_path, len(team))
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cache, players=players_n, ball=ball_n)
    scale = np.array([length, width], dtype=np.float32)
    players = players_n * scale
    ball = gate_ball(ball_n * scale)
    halves = [(h, a, b) for h, a, b in half_video_blocks(game)]
    return Match(name=game, ball=ball, players=players, team=team, halves=halves,
                 kickoffs=dict(KICKOFF_FRAMES), pitch=(length, width), tags=parse_tags(tag_path))


if __name__ == "__main__":
    import time
    t = time.time()
    m = load_match()
    print("loaded in %.1fs" % (time.time() - t))
    print("frames", m.n, "slots", m.players.shape[1], "home slots", int((m.team == 0).sum()))
    print("ball visible %.3f" % m.ball_valid.mean())
    print("tags", len(m.tags))
