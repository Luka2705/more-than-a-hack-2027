"""Local tracking + automatic tags -> temporally coherent H/A/D/X labels.

No video, ground truth, network requests, training, or match-specific overrides.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, half_video_blocks, require
from solution.config import DEFAULT
from starter.baseline import write_submission
from starter.load_smart_tagging import load as load_tags


def runs(mask):
    """Half-open runs of True, including runs touching either endpoint."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def mean_window(values, radius):
    """Centred moving mean; truncate the window at the ends, do not zero-pad."""
    x = np.asarray(values, dtype=np.float64)
    if radius <= 0:
        return x.copy()
    idx = np.arange(len(x))
    c = np.concatenate((np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)))
    lo, hi = np.maximum(0, idx - radius), np.minimum(len(x), idx + radius + 1)
    return (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def normalise_team(name):
    text = unicodedata.normalize("NFKD", name or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("athletic", "atletic"))
    return " ".join(t for t in tokens if t not in {"fc", "club", "cf"})


def read_metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2 or len({t.get("id") for t in teams}) != 2:
        raise ValueError("Tracking metadata must define two distinct teams.")
    side = {team.get("id"): i for i, team in enumerate(teams)}
    slots = []
    for player in root.iter("Player"):
        if player.get("teamId") not in side:
            raise ValueError("Tracking slot has an unknown team identifier.")
        slots.append(side[player.get("teamId")])
    if not slots or set(slots) != {0, 1}:
        raise ValueError("Tracking must have player slots for both teams.")
    names = {normalise_team(t.findtext("Name")): i for i, t in enumerate(teams)}
    if "" in names or len(names) != 2:
        raise ValueError("Tracking team names are missing or ambiguous.")
    return np.asarray(slots), names


def read_tracking(path, n_frames, n_slots, pitch):
    """Strict streaming parser. Error messages never include restricted records."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)
    count = 0
    with open(path, encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if row >= n_frames:
                raise ValueError("Tracking has more rows than the reference video.")
            try:
                frame, groups, b = line.strip().split(":")
                if int(frame) != row + 1:
                    raise ValueError("non-contiguous frame index")
                values = np.fromstring(groups.replace(";", ","), sep=",")
                bvalues = np.fromstring(b, sep=",")
                if len(values) != n_slots * 3 or len(bvalues) != 3:
                    raise ValueError("unexpected field count")
                if np.isinf(values).any() or np.isinf(bvalues).any():
                    raise ValueError("infinite coordinates")
                players[row] = values.reshape(n_slots, 3)
                ball[row] = bvalues[:2]
            except (ValueError, IndexError):
                raise ValueError("Invalid tracking format at record {}.".format(row + 1)) from None
            count += 1
    if count != n_frames:
        raise ValueError("Tracking row count differs from the reference video.")
    players[:, :, :2] *= np.asarray(pitch, dtype=np.float32)
    ball *= np.asarray(pitch)
    return players, ball


def clean_ball(ball, fps, cfg):
    """Reject isolated impossible jumps; interpolate only short, plausible gaps.

    Interpolation is per half. A long pass or occlusion is never filled by a
    straight line through the entire missing interval.
    """
    out = np.array(ball, dtype=np.float64, copy=True)
    out[~np.isfinite(out).all(axis=1)] = np.nan
    observed = np.flatnonzero(np.isfinite(out).all(axis=1))
    if len(observed) >= 3:
        left, centre, right = observed[:-2], observed[1:-1], observed[2:]
        vin = np.linalg.norm(out[centre] - out[left], axis=1) * fps / (centre - left)
        vout = np.linalg.norm(out[right] - out[centre], axis=1) * fps / (right - centre)
        vskip = np.linalg.norm(out[right] - out[left], axis=1) * fps / (right - left)
        spike = (vin > cfg.max_ball_speed_m_s) & (vout > cfg.max_ball_speed_m_s)
        spike &= vskip <= cfg.max_ball_speed_m_s
        out[centre[spike]] = np.nan
    original = np.isfinite(out).all(axis=1)
    for lo, hi in runs(~original):
        if lo == 0 or hi == len(out) or hi - lo > round(cfg.interpolate_gap_s * fps):
            continue
        speed = np.linalg.norm(out[hi] - out[lo - 1]) * fps / (hi - lo + 1)
        if speed <= cfg.max_ball_speed_m_s:
            fraction = np.arange(1, hi - lo + 1) / (hi - lo + 1)
            out[lo:hi] = out[lo - 1] + fraction[:, None] * (out[hi] - out[lo - 1])
    return out, original


def ball_speed(ball, fps, cfg):
    radius = max(1, round(cfg.velocity_radius_s * fps))
    idx = np.arange(len(ball))
    lo, hi = np.maximum(0, idx - radius), np.minimum(len(ball) - 1, idx + radius)
    speed = np.linalg.norm(ball[hi] - ball[lo], axis=1) * fps / np.maximum(1, hi - lo)
    speed[speed > cfg.max_ball_speed_m_s] = np.nan
    return speed


ATTACK = {
    "build up", "progression", "creating chances", "attacking transition",
    "long ball", "ball in final third", "ball in the box",
}
DEFENCE = {
    "defensive transition", "defending in defensive third",
    "defending in middle third", "defending in attacking third",
}


def phase_evidence(events, team_names, n, first, fps):
    """Max-pool correlated tags; defending team's opponent possesses the ball.

    Player counts in the box and goals are not possession intervals. Set pieces
    are processed separately. N/A and unrecognised teams cast no team vote.
    """
    votes = np.zeros((n, 2), dtype=np.float64)
    for event in events:
        code = (event["code"] or "").strip().lower()
        if code not in ATTACK | DEFENCE:
            continue
        team = team_names.get(normalise_team(event["tags"].get("Team")))
        if team is None:
            continue
        if code in DEFENCE:
            team = 1 - team
        start, end = event["start_frame"] - first, event["end_frame"] - first
        lo, hi = max(0, start), min(n, end)
        if hi <= lo:
            continue
        # Soft 1 s edges acknowledge the automatic tag's temporal uncertainty.
        t = np.arange(lo, hi)
        weight = np.minimum(1., np.minimum((t - start + 1) / fps, (end - t) / fps))
        votes[lo:hi, team] = np.maximum(votes[lo:hi, team], weight)
    return votes


def tracking_evidence(players, ball, team_slots, pitch, fps, cfg):
    n = len(ball)
    known = np.isfinite(ball).all(axis=1)
    delta = players[:, :, :2].astype(np.float64) - ball[:, None, :]
    dist = np.linalg.norm(delta, axis=2)
    dist[~np.isfinite(dist)] = np.inf
    nearest = np.column_stack([dist[:, team_slots == team].min(axis=1) for team in (0, 1)])
    control = np.exp(-0.5 * (nearest / cfg.control_radius_m) ** 2)
    # A hotly contested ball is weak evidence, regardless of the closest side.
    team = control[:, 0] - control[:, 1]
    speed = ball_speed(ball, fps, cfg)
    # Very fast balls are passes, not controlled possession by whoever they pass.
    team *= 1 / (1 + (np.nan_to_num(speed, nan=0.) / 12.) ** 2)
    margin = cfg.boundary_margin_m
    outside = known & ((ball[:, 0] < -margin) | (ball[:, 0] > pitch[0] + margin)
                       | (ball[:, 1] < -margin) | (ball[:, 1] > pitch[1] + margin))
    inside = known & ~outside
    moving = inside & np.isfinite(speed) & (speed >= cfg.moving_ball_m_s)
    stationary = inside & np.isfinite(speed) & (speed < cfg.stationary_ball_m_s)
    radius = max(1, round(cfg.stationary_window_s * fps / 2))
    still = mean_window(stationary.astype(float), radius)
    visible = np.isfinite(players[:, :, :2]).all(axis=2) & np.isfinite(players[:, :, 2])
    count = visible.sum(axis=1)
    walking = (visible & (players[:, :, 2] < cfg.walking_speed_m_s)).sum(axis=1)
    quiet = (count >= cfg.min_players_for_motion) & (walking / np.maximum(1, count) >= 0.8)
    quiet = mean_window(quiet.astype(float), radius)
    # Stationary alone is insufficient: a goalkeeper holding the ball is live.
    stopped = (still >= 0.75) & (quiet >= 0.75)
    dead = np.maximum(mean_window(outside.astype(float), max(1, round(0.2 * fps))),
                      stopped.astype(float) * 0.65)
    team[outside] = 0
    return {
        "team": team, "control": control.max(axis=1), "dead": dead,
        "moving": moving, "stationary": stationary, "outside": outside,
        "known": known, "speed": speed, "quiet": quiet,
    }


def restart_evidence(events, names, features, phases, first, fps, cfg):
    """Locate restart motion in automatic tag windows and backtrack stoppages.

    No hard-coded game timestamps. A tagged window supplies context, never an
    exact hand-labelled restart. Ambiguous boundaries fall back to its midpoint.
    """
    n = len(features["known"])
    dead = np.zeros(n)
    seed = np.zeros((n, 2))
    records = []
    moving = mean_window(features["moving"].astype(float), max(1, round(0.12 * fps)))
    stationary = features["stationary"].astype(float)
    c = np.r_[0., np.cumsum(stationary)]
    idx = np.arange(n)
    previous_still = (c[idx] - c[np.maximum(0, idx - fps)]) / np.minimum(fps, np.maximum(1, idx))
    lookback = round(cfg.restart_lookback_s * fps)
    margin = round(cfg.restart_margin_s * fps)
    for event in sorted(events, key=lambda e: e["start_frame"]):
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        start, end = event["start_frame"] - first, event["end_frame"] - first
        if end <= 0 or start >= n or end <= start:
            continue
        lo, hi = max(0, start - margin), min(n, end + margin)
        centre = int(np.clip((start + end) // 2, lo, hi - 1))
        candidates = np.arange(lo, hi)
        # Prior is subordinate to actual stationary -> moving evidence.
        proximity = np.exp(-np.abs(candidates - centre) / max(1, fps * 2))
        launch = moving[lo:hi] * (0.25 + previous_still[lo:hi])
        strong = (moving[lo:hi] >= 0.6) & (previous_still[lo:hi] >= 0.25)
        if strong.any():
            score = np.where(strong, launch + 0.15 * proximity, -np.inf)
            restart = int(candidates[np.argmax(score)])
            method = "motion"
        else:
            # No fabricated exact timing when the ball is occluded.
            restart, method = centre, "tag_midpoint"
        kind = (event["tags"].get("Type") or "").strip().lower()
        begin = restart
        if kind == "kick off (start)":
            begin = 0
        else:
            left = max(0, restart - lookback)
            # Last visible, moving ball inside the pitch is conservative live evidence.
            live = np.flatnonzero(features["moving"][left:restart])
            candidate = left + int(live[-1]) + 1 if len(live) else left
            gap = restart - candidate
            # Require physical evidence OR absence of live phase tags. Missing
            # ball alone never globally implies dead; this is restart-conditioned.
            stopped = features["outside"][candidate:restart].any()
            stopped |= features["stationary"][candidate:restart].sum() >= fps / 2
            no_phase = phases[candidate:restart].max(axis=1).mean() < 0.25 if gap else False
            if gap >= fps / 2 and (stopped or no_phase):
                begin = candidate
        if begin < restart:
            dead[begin:restart] = np.maximum(dead[begin:restart], 0.95)
        team = names.get(normalise_team(event["tags"].get("Team")))
        if team is not None:
            stop = min(n, restart + round(cfg.restart_team_seed_s * fps))
            seed[restart:stop, team] = 1.
        records.append({"restart_frame": first + restart, "dead_start_frame": first + begin,
                        "timing_method": method})
    return dead, seed, records


def decode(costs, fps, cfg):
    """Exact minimum-cost state sequence. Costs are integrated over seconds.

    Positive transition costs suppress brief flickers without imposing a blanket
    minimum possession duration. Ties prefer staying, preserving the last team
    through missing evidence instead of inventing handovers inside an occlusion.
    """
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[1] != 3 or not np.isfinite(costs).all():
        raise ValueError("Decoder requires finite costs with shape (frames, 3).")
    if not len(costs):
        return np.empty(0, dtype=np.int8)
    transition = np.full((3, 3), cfg.dead_switch_cost_s)
    transition[0, 1] = transition[1, 0] = cfg.team_switch_cost_s
    np.fill_diagonal(transition, 0.)
    previous = costs[0] / fps
    back = np.zeros((len(costs), 3), dtype=np.int8)
    states = np.arange(3)
    for t in range(1, len(costs)):
        candidate = previous[:, None] + transition
        best = candidate.argmin(axis=0)
        # With no evidence, an equal-cost switch should occur only once new
        # evidence arrives, not arbitrarily at the start of the gap.
        best = np.where(candidate[states, states] <= candidate[best, states] + 1e-12, states, best)
        previous = candidate[best, states] + costs[t] / fps
        previous -= previous.min()
        back[t] = best
    out = np.empty(len(costs), dtype=np.int8)
    out[-1] = previous.argmin()
    for t in range(len(costs) - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def infer_half(players, ball, slots, names, events, first, pitch, fps, cfg=DEFAULT):
    ball, observed = clean_ball(ball, fps, cfg)
    f = tracking_evidence(players, ball, slots, pitch, fps, cfg)
    phases = phase_evidence(events, names, len(ball), first, fps)
    restart_dead, seeds, restart_records = restart_evidence(events, names, f, phases, first, fps, cfg)
    radius = max(1, round(cfg.evidence_radius_s * fps))
    team = cfg.control_weight * mean_window(f["team"], radius)
    team += cfg.phase_weight * (phases[:, 0] - phases[:, 1])
    team += seeds[:, 0] - seeds[:, 1]
    dead = np.maximum(f["dead"], restart_dead)
    live = np.maximum(f["control"] * f["known"], phases.max(axis=1))
    live = np.maximum(live, f["moving"].astype(float))
    # Bound team confidence so duplicated tags cannot overpower a clear stoppage.
    signed = np.clip(team, -1.5, 1.5) * (1 - dead)
    costs = np.column_stack((2.5 * dead - signed, 2.5 * dead + signed,
                             (0.15 * f["known"] + 1.5 * live) * (1 - dead)))
    states = decode(costs, fps, cfg)
    diagnostics = {"observed_ball": observed, "team_evidence": team,
                   "dead_evidence": dead, "state": states}
    return np.asarray(["H", "A", "D"])[states], diagnostics, restart_records


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(data_dir=None, out="artifacts/Import-Export.csv", intermediate=False, cfg=DEFAULT):
    n, fps, pitch = (GAMES[GAME][k] for k in ("n_video_frames", "fps", "pitch"))
    meta, raw, tagging = require(["atd_meta", "atd_raw", "smart_tagging"], data_dir=data_dir)
    slots, names = read_metadata(meta)
    print("Reading local tactical tracking (no data records are printed).", flush=True)
    players, ball = read_tracking(raw, n, len(slots), pitch)
    events = load_tags(data_dir=data_dir)
    labels = np.full(n, "X", dtype="<U1")
    output = Path(out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    restarts = []
    for half, first, last in half_video_blocks(GAME):
        print("Inferring half {} locally.".format(half), flush=True)
        labels[first:last + 1], diag, records = infer_half(
            players[first:last + 1], ball[first:last + 1], slots, names,
            events, first, pitch, fps, cfg)
        restarts.extend(records)
        if intermediate:
            np.savez_compressed(output.with_suffix(".half{}.npz".format(half)),
                                first_frame=first, **diag)
    write_submission(str(output), labels)
    # Format checks do not open any ground truth.
    from evaluation.possession_metrics import read_label_csv, validate_submission
    reason = validate_submission(read_label_csv(str(output)), n)
    if reason:
        raise ValueError("Generated submission failed validation: " + reason)
    source_root = Path(__file__).resolve().parents[1]
    sources = ["solution/infer.py", "solution/config.py", "starter/baseline.py",
               "starter/load_smart_tagging.py", "games.py", "requirements-solution.txt"]
    manifest = {
        "method": "Import-Export deterministic tracking/tag sequence decoder v1",
        "config": asdict(cfg), "python": platform.python_version(), "numpy": np.__version__,
        "inputs_sha256": {Path(p).name: sha256(p) for p in (meta, raw, tagging)},
        "source_sha256": {p: sha256(source_root / p) for p in sources},
        "submission_sha256": sha256(output), "frames": n,
        "format_valid": True, "ground_truth_used": False, "video_used": False,
        "training_or_tuning_performed": False, "network_required_for_inference": False,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if intermediate:
        output.with_suffix(".restarts.json").write_text(json.dumps(restarts, indent=2) + "\n")
    print("PASS: submission format and frame alignment. Saved {}".format(output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="directory containing tactical/ (or MTAH_DATA_DIR)")
    parser.add_argument("--out", default="artifacts/Import-Export.csv")
    parser.add_argument("--intermediates", action="store_true", help="save local diagnostic arrays")
    args = parser.parse_args()
    try:
        build(args.data_dir, args.out, args.intermediates)
    except (ValueError, OSError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
