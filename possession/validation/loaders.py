"""External labelled matches turned into the same `Match` + per-frame ground truth as BAR-ATH.

Sources (licences and volumes in docs/TECHNICAL_REPORT.md):
  * IDSSE / DFL (Bassek et al. 2025, CC BY 4.0): 7 Bundesliga matches, 25 fps TRACAB positions.
    Labels: the per-frame BallStatus (0 dead / 1 alive) and BallPossession (1 home / 2 away)
    attributes of the ball frames. Same structure as the challenge GT (corrected tracking:
    dead where marked dead, otherwise the team named as owning the ball).
  * Metrica Sports sample data (public, "acknowledge the source"): 3 matches, 25 fps.
    Labels derived from the event data by rule (see metrica_labels).

Each match is laid out as half 1, a 500-frame empty separator, half 2, so the per-half decoder
and every centred filter see the same structure as the provided game.

Coordinates: metres with origin at the corner, x along the length, as for BAR-ATH.
"""

import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

from ..data import Match

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "external")
CACHE = os.path.join(ROOT, "cache", "external")
SEP = 500  # empty frames between halves

# -- IDSSE ---------------------------------------------------------------------------------------

_FS = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FR = re.compile(r'<Frame N="(\d+)"[^>]*? X="(-?[\d.]+)" Y="(-?[\d.]+)"(.*?)/>')
_BP = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def idsse_ids():
    files = sorted(glob.glob(os.path.join(EXT, "idsse", "DFL_04_03_positions_raw_observed_*.xml")))
    return [re.search(r"(DFL-MAT-\w+)\.xml", f).group(1) for f in files]


def _idsse_parse(match_id):
    out = os.path.join(CACHE, "idsse_%s.npz" % match_id)
    if os.path.isfile(out):
        return out
    pos = glob.glob(os.path.join(EXT, "idsse", "DFL_04_03_positions_raw_observed_*%s.xml"
                                 % match_id))[0]
    info = glob.glob(os.path.join(EXT, "idsse", "DFL_02_01_matchinformation_*%s.xml"
                                  % match_id))[0]
    txt = open(info, encoding="utf-8").read()
    home_id = re.search(r'HomeTeamId="([^"]+)"', txt).group(1)
    sections = {}  # section -> {"ball": (...), persons: {pid: (team, N, X, Y)}}
    cur = None
    with open(pos, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("<FrameSet"):
                m = _FS.search(line)
                sec, team, pid = m.groups()
                cur = {"sec": sec, "team": team, "pid": pid, "N": [], "X": [], "Y": [], "P": [],
                       "S": []}
                continue
            if line.startswith("</FrameSet"):
                s = sections.setdefault(cur["sec"], {"persons": [], "ball": None})
                if cur["team"] == "BALL":
                    s["ball"] = cur
                elif cur["team"] != "referee":
                    s["persons"].append(cur)
                cur = None
                continue
            if cur is not None and line.startswith("<Frame "):
                m = _FR.search(line)
                cur["N"].append(int(m.group(1)))
                cur["X"].append(float(m.group(2)))
                cur["Y"].append(float(m.group(3)))
                if cur["team"] == "BALL":
                    b = _BP.search(m.group(4))
                    cur["P"].append(int(b.group(1)))
                    cur["S"].append(int(b.group(2)))
    arrays = {}
    for k, sec in enumerate(("firstHalf", "secondHalf")):
        s = sections[sec]
        ball = s["ball"]
        n0 = min(ball["N"])
        T = max(ball["N"]) - n0 + 1
        bxy = np.full((T, 2), np.nan, np.float32)
        lab = np.full(T, ord("X"), np.uint8)
        idx = np.array(ball["N"]) - n0
        bxy[idx, 0] = ball["X"]
        bxy[idx, 1] = ball["Y"]
        P = np.array(ball["P"])
        S = np.array(ball["S"])
        lab[idx] = np.where(S == 0, ord("D"), np.where(P == 1, ord("H"), ord("A")))
        persons = s["persons"]
        pxy = np.full((T, len(persons), 2), np.nan, np.float32)
        team = np.array([0 if p["team"] == home_id else 1 for p in persons], np.int8)
        for j, p in enumerate(persons):
            i = np.array(p["N"]) - n0
            ok = (i >= 0) & (i < T)
            pxy[i[ok], j, 0] = np.array(p["X"])[ok]
            pxy[i[ok], j, 1] = np.array(p["Y"])[ok]
        arrays["ball%d" % k] = bxy
        arrays["lab%d" % k] = lab
        arrays["pl%d" % k] = pxy
        arrays["team%d" % k] = team
        arrays["pid%d" % k] = np.array([p["pid"] for p in persons])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return out


def parse_all_idsse(processes=6):
    ids = idsse_ids()
    with Pool(processes) as pool:
        return pool.map(_idsse_parse, ids)


def _assemble(name, halves_data, pitch, centred):
    """halves_data: list of (ball[T,2], players[T,P,2], team[P], pid[P], labels[T] uint8 chars).
    Player slots are unified across halves by person id. Coordinates moved to corner origin."""
    pids = []
    for _, _, team, pid, _ in halves_data:
        for t, p in zip(team, pid):
            if p not in pids:
                pids.append(p)
    team_of = {}
    for _, _, team, pid, _ in halves_data:
        for t, p in zip(team, pid):
            team_of[p] = t
    P = len(pids)
    total = sum(len(h[0]) for h in halves_data) + SEP * (len(halves_data) - 1)
    ball = np.full((total, 2), np.nan, np.float32)
    players = np.full((total, P, 2), np.nan, np.float32)
    gt = np.full(total, "X", dtype="<U1")
    halves, kickoffs = [], {}
    off = 0
    L, W = pitch
    shift = np.array([L / 2, W / 2], np.float32) if centred else np.zeros(2, np.float32)
    for h, (b, pl, team, pid, lab) in enumerate(halves_data, start=1):
        T = len(b)
        ball[off:off + T] = b + shift
        cols = [pids.index(p) for p in pid]
        players[off:off + T, cols] = pl + shift
        gt[off:off + T] = np.array([chr(c) for c in lab])
        graded = np.flatnonzero(gt[off:off + T] != "X")
        a, e = off + graded[0], off + graded[-1]
        halves.append((h, int(a), int(e)))
        alive = np.flatnonzero(np.isin(gt[a:e + 1], ["H", "A"]))
        kickoffs[h] = int(a + alive[0]) if len(alive) else int(a)
        off += T + SEP
    team = np.array([team_of[p] for p in pids], np.int8)
    m = Match(name=name, ball=ball, players=players, team=team, halves=halves,
              kickoffs=kickoffs, pitch=pitch, tags=[])
    return m, gt


def load_idsse(match_id):
    z = np.load(_idsse_parse(match_id))
    hd = [(z["ball%d" % k], z["pl%d" % k], z["team%d" % k], z["pid%d" % k], z["lab%d" % k])
          for k in (0, 1)]
    # DFL y axis points up; flipping is irrelevant for distances, kept as is.
    return _assemble("idsse_" + match_id, hd, (105.0, 68.0), centred=True)


# -- Metrica sample data -------------------------------------------------------------------------

MET = os.path.join(EXT, "metrica", "data")
POSS_TYPES = {"PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST", "CARRY", "BALL OUT"}


def _metrica_events(game):
    if game in (1, 2):
        import csv
        path = os.path.join(MET, "Sample_Game_%d" % game, "Sample_Game_%d_RawEventsData.csv" % game)
        rows = list(csv.DictReader(open(path)))
        ev = [{"team": r["Team"], "type": r["Type"], "sub": r["Subtype"] or "",
               "period": int(r["Period"]), "start": int(r["Start Frame"]),
               "end": int(r["End Frame"])} for r in rows]
        home = "Home"
    else:
        d = json.load(open(os.path.join(MET, "Sample_Game_3", "Sample_Game_3_events.json")))
        ev = []
        for r in d["data"]:
            sub = r.get("subtypes")
            if isinstance(sub, list):
                sub = "-".join(s["name"] for s in sub)
            elif isinstance(sub, dict):
                sub = sub["name"]
            ev.append({"team": r["team"]["name"], "type": r["type"]["name"], "sub": sub or "",
                       "period": int(r["period"]), "start": int(r["start"]["frame"]),
                       "end": int(r["end"]["frame"] or r["start"]["frame"])})
        home = "Team A"
    ev.sort(key=lambda e: (e["start"], e["end"]))
    return ev, home


def _is_restart(e):
    return e["type"] == "SET PIECE" or (e["type"] == "PASS" and "GOAL KICK" in e["sub"])


def _dead_start(e):
    """Frame at which play stops because of this event, or None."""
    t, s = e["type"], e["sub"]
    if t == "BALL OUT":
        return e["end"]
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return e["end"]
    if t == "FAULT RECEIVED":
        return e["start"]
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return e["start"]
    if t == "BALL LOST" and ("OFFSIDE" in s or "END HALF" in s):
        return e["end"]
    return None


def metrica_labels(ev, home, frames_by_period):
    """Per-frame labels from events.

    team    : team of the most recent possession-type event (PASS, RECOVERY, SHOT, SET PIECE,
              BALL LOST, CARRY, BALL OUT) started at or before the frame. Challenges and fouls do
              not change the owner (no control).
    dead    : from a stoppage event (ball out, shot out or goal, foul, offside) to the start of the
              next restart (set piece or goal kick); cancelled if a non-restart possession event
              happens first (advantage / missed stoppage). Before the kickoff of each period: dead.
    """
    out = {}
    for period, frames in frames_by_period.items():
        f0, f1 = frames[0], frames[-1]
        T = f1 - f0 + 1
        team = np.full(T, -1, np.int8)
        dead = np.zeros(T, bool)
        pe = [e for e in ev if e["period"] == period]
        for e in pe:
            if e["type"] in POSS_TYPES:
                i = e["start"] - f0
                if 0 <= i < T:
                    team[i] = 0 if e["team"] == home else 1
        idx = np.maximum.accumulate(np.where(team >= 0, np.arange(T), -1))
        team = np.where(idx >= 0, team[np.maximum(idx, 0)], -1)
        ko = [e["start"] for e in pe if e["type"] == "SET PIECE" and "KICK OFF" in e["sub"]]
        ko = ko[0] if ko else f0
        dead[:max(0, ko - f0)] = True
        for k, e in enumerate(pe):
            ds = _dead_start(e)
            if ds is None:
                continue
            stop = None
            for e2 in pe[k + 1:]:
                if e2["start"] < ds:
                    continue
                if _is_restart(e2):
                    stop = e2["start"]
                    break
                if e2["type"] in ("PASS", "RECOVERY", "CARRY", "SHOT") and e2["start"] > ds:
                    stop = None
                    break
            if stop is not None and stop - ds < 25 * 120:
                dead[max(0, ds - f0):max(0, stop - f0)] = True
        lab = np.where(dead, ord("D"), np.where(team == 0, ord("H"),
                                                np.where(team == 1, ord("A"), ord("D"))))
        out[period] = (f0, lab.astype(np.uint8), ko)
    return out


def _metrica_tracking_csv(game):
    import pandas as pd
    halves = {}
    teams = []
    for side in ("Home", "Away"):
        path = os.path.join(MET, "Sample_Game_%d" % game,
                            "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (game, side))
        df = pd.read_csv(path, skiprows=2)
        cols = df.columns.tolist()
        teams.append((side, df, cols))
    home_df = teams[0][1]
    period = home_df["Period"].values
    frame = home_df["Frame"].values
    # ball columns: last two
    bcols = home_df.columns[-2:]
    ball = home_df[bcols].values.astype(np.float32)
    plist, team, pid = [], [], []
    for k, (side, df, cols) in enumerate(teams):
        names = [c for c in cols if c.startswith("Player")]
        for nme in names:
            j = cols.index(nme)
            plist.append(df.iloc[:, [j, j + 1]].values.astype(np.float32))
            team.append(k)
            pid.append("%s_%s" % (side, nme))
    pl = np.stack(plist, axis=1)
    return period, frame, ball, pl, np.array(team, np.int8), np.array(pid)


def _metrica_tracking_epts():
    """Game 3 EPTS: the player channel layout changes at each DataFormatSpecification (one per
    substitution segment), so each line is decoded with the layout of its segment."""
    d = os.path.join(MET, "Sample_Game_3")
    meta = open(os.path.join(d, "Sample_Game_3_metadata.xml"), encoding="utf-8").read()
    chan_pid = dict(re.findall(r'<PlayerChannel channelId="x" id="(player\d+)_x" playerId="(\w+)"',
                               meta))
    team_of = dict(re.findall(r'<Player id="(\w+)" teamId="(\w+)"', meta))
    specs = []
    for m in re.finditer(r'<DataFormatSpecification endFrame="(\d+)" separator=":" '
                         r'startFrame="(\d+)">(.*?)</DataFormatSpecification>', meta, re.S):
        chans = re.findall(r'playerChannelId="(player\d+)_x"', m.group(3))
        specs.append((int(m.group(2)), int(m.group(1)), [chan_pid[c] for c in chans]))
    pids = []
    for _, _, ps in specs:
        for q in ps:
            if q not in pids:
                pids.append(q)
    col = {q: i for i, q in enumerate(pids)}
    lines = open(os.path.join(d, "Sample_Game_3_tracking.txt")).read().strip().split("\n")
    n = len(lines)
    frame = np.zeros(n, int)
    pl = np.full((n, len(pids), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    si = 0
    for i, line in enumerate(lines):
        fr, players, b = line.split(":")
        f = int(fr)
        frame[i] = f
        while not (specs[si][0] <= f <= specs[si][1]):
            si += 1
        vals = np.array(players.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 2)
        pl[i, [col[q] for q in specs[si][2]]] = vals
        ball[i] = np.array(b.split(","), dtype=np.float32)
    team = np.array([0 if team_of[q] == "FIFATMA" else 1 for q in pids], np.int8)
    ev = json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]
    last1 = max(int(e["end"]["frame"] or e["start"]["frame"]) for e in ev if e["period"] == 1)
    split = min(a for a, _, _ in specs if a > last1) if any(a > last1 for a, _, _ in specs)         else last1 + 1
    period = np.where(frame < split, 1, 2)
    return period, frame, ball, pl, team, np.array(pids)


def load_metrica(game):
    pitch = (105.0, 68.0)
    if game in (1, 2):
        period, frame, ball, pl, team, pid = _metrica_tracking_csv(game)
    else:
        period, frame, ball, pl, team, pid = _metrica_tracking_epts()
    ev, home = _metrica_events(game)
    fbp = {p: frame[period == p] for p in (1, 2)}
    labs = metrica_labels(ev, home, fbp)
    scale = np.array(pitch, np.float32)
    hd = []
    for p in (1, 2):
        sel = period == p
        f0, lab, ko = labs[p]
        b = ball[sel] * scale
        x = pl[sel] * scale
        hd.append((b, x, team, pid, lab[:sel.sum()]))
    return _assemble("metrica_%d" % game, hd, pitch, centred=False)


def all_external(include=("idsse", "metrica")):
    out = []
    if "metrica" in include:
        out += [("metrica_%d" % g, lambda g=g: load_metrica(g)) for g in (1, 2, 3)]
    if "idsse" in include:
        out += [("idsse_" + i, lambda i=i: load_idsse(i)) for i in idsse_ids()]
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "parse":
        print(parse_all_idsse())
