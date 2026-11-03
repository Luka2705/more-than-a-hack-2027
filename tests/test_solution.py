"""Synthetic software checks; no competition data or ground truth is read.

These fixtures test invariants, not predictive quality or parameter selection.
"""

import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from solution.config import DEFAULT
from solution.infer import (
    ball_speed, clean_ball, decode, infer_half, mean_window, normalise_team,
    phase_evidence, read_metadata, read_tracking, restart_evidence, runs,
    tracking_evidence,
)


FPS = 25


def event(code, team, start, end, kind=None):
    tags = {"Team": team}
    if kind:
        tags["Type"] = kind
    return {"code": code, "tags": tags, "start_frame": start, "end_frame": end}


def scene(n=500):
    ball = np.tile([50., 30.], (n, 1))
    players = np.zeros((n, 2, 3))
    players[:, 0, :2] = ball + [0.5, 0]
    players[:, 1, :2] = ball + [20, 10]
    players[:, :, 2] = 3.
    return players, ball, np.array([0, 1])


class ArrayTests(unittest.TestCase):
    def test_runs_at_edges(self):
        self.assertEqual(runs([1, 1, 0, 1]), [(0, 2), (3, 4)])
        self.assertEqual(runs([]), [])

    def test_window_preserves_constant_at_edges(self):
        np.testing.assert_allclose(mean_window(np.ones((7, 2)), 50), 1.)

    def test_short_gap_only(self):
        ball = np.column_stack((np.arange(100) * .1, np.ones(100)))
        expected = ball.copy()
        ball[10:15] = np.nan
        ball[30:65] = np.nan
        clean, observed = clean_ball(ball, FPS, DEFAULT)
        np.testing.assert_allclose(clean[10:15], expected[10:15])
        self.assertTrue(np.isnan(clean[30:65]).all())
        self.assertFalse(observed[10:15].any())

    def test_no_extrapolation_or_impossible_interpolation(self):
        ball = np.zeros((20, 2))
        ball[:2] = np.nan
        ball[-2:] = np.nan
        ball[8:10] = np.nan
        ball[10:] += 100
        clean, _ = clean_ball(ball, FPS, DEFAULT)
        self.assertTrue(np.isnan(clean[:2]).all())
        self.assertTrue(np.isnan(clean[8:10]).all())
        self.assertTrue(np.isnan(clean[-2:]).all())

    def test_isolated_teleport_rejected(self):
        ball = np.column_stack((np.arange(100) * .1, np.ones(100)))
        ball[50] = [1000, 1000]
        clean, observed = clean_ball(ball, FPS, DEFAULT)
        self.assertFalse(observed[50])
        np.testing.assert_allclose(clean[50], [5., 1.])

    def test_speed_uses_metres_and_seconds(self):
        ball = np.column_stack((np.arange(100) * .12, np.zeros(100)))
        np.testing.assert_allclose(ball_speed(ball, FPS, DEFAULT), 3.)


class InputTests(unittest.TestCase):
    def test_tracking_indices_and_units(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "synthetic.txt"
            file.write_text("1:0.5,0.5,2;NaN,NaN,NaN:0.25,0.5,1\n"
                            "2:0.6,0.5,2;0.1,0.2,3:NaN,NaN,NaN\n")
            players, ball = read_tracking(file, 2, 2, (100, 60))
            np.testing.assert_allclose(players[0, 0], [50, 30, 2])
            np.testing.assert_allclose(ball[0], [25, 30])
            self.assertTrue(np.isnan(ball[1]).all())
            with self.assertRaises(ValueError):
                read_tracking(file, 3, 2, (100, 60))

    def test_duplicate_frame_rejected_without_record_leak(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "synthetic.txt"
            file.write_text("1:0.98765,0.5,2:0.25,0.5,1\n" * 2)
            with self.assertRaises(ValueError) as error:
                read_tracking(file, 2, 1, (100, 60))
            self.assertNotIn("98765", str(error.exception))

    def test_metadata_team_order_and_bad_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "synthetic.xml"
            xml = ('<root><Team id="h"><Name>Home</Name></Team>'
                   '<Team id="a"><Name>Away</Name></Team>'
                   '<Player teamId="a"/><Player teamId="h"/></root>')
            file.write_text(xml)
            slots, names = read_metadata(file)
            self.assertEqual(slots.tolist(), [1, 0])
            self.assertEqual(names, {"home": 0, "away": 1})
            file.write_text(xml.replace('teamId="a"', 'teamId="other"'))
            with self.assertRaises(ValueError):
                read_metadata(file)


class TagTests(unittest.TestCase):
    def test_defending_team_is_inverted(self):
        votes = phase_evidence([event("DEFENDING IN MIDDLE THIRD", "Home", 0, 100)],
                               {"home": 0, "away": 1}, 100, 0, FPS)
        self.assertEqual(votes[50].tolist(), [0., 1.])

    def test_unknown_and_box_player_counts_do_not_claim_possession(self):
        events = [event("BUILD UP", "N/A", 0, 100),
                  event("PLAYERS IN THE BOX", "Home", 0, 100)]
        self.assertFalse(phase_evidence(events, {"home": 0}, 100, 0, FPS).any())

    def test_duplicate_tags_do_not_amplify(self):
        e = event("PROGRESSION", "Home", 1000, 1100)
        one = phase_evidence([e], {"home": 0}, 100, 1000, FPS)
        np.testing.assert_array_equal(one, phase_evidence([e] * 10, {"home": 0}, 100, 1000, FPS))

    def test_name_normalisation(self):
        self.assertEqual(normalise_team("Athletic Club"), normalise_team("Atletic Club"))
        self.assertEqual(normalise_team("FC Barcelona"), "barcelona")


class DecoderTests(unittest.TestCase):
    def test_dynamic_programming_matches_exhaustive_search(self):
        costs = np.random.default_rng(17).normal(size=(7, 3))
        fps = 2
        result = decode(costs, fps, DEFAULT)
        def energy(path):
            total = sum(costs[t, state] / fps for t, state in enumerate(path))
            for a, b in zip(path, path[1:]):
                if a != b:
                    total += DEFAULT.team_switch_cost_s if a < 2 and b < 2 else DEFAULT.dead_switch_cost_s
            return total
        optimum = min(energy(p) for p in itertools.product(range(3), repeat=len(costs)))
        self.assertAlmostEqual(energy(result), optimum)

    def test_brief_opponent_touch_does_not_change_control(self):
        costs = np.tile([-1., 1., 2.], (250, 1))
        costs[100:105, :2] = [1., -1.]
        self.assertTrue((decode(costs, FPS, DEFAULT) == 0).all())

    def test_sustained_opponent_control_changes_team(self):
        costs = np.tile([-1., 1., 2.], (250, 1))
        costs[125:, :2] = [1., -1.]
        expected = np.r_[np.zeros(125), np.ones(125)]
        np.testing.assert_array_equal(decode(costs, FPS, DEFAULT), expected)

    def test_gap_does_not_invent_a_new_team(self):
        costs = np.tile([-1., 1., 2.], (250, 1))
        costs[50:200] = 0
        self.assertTrue((decode(costs, FPS, DEFAULT) == 0).all())

    def test_gap_does_not_end_a_stoppage(self):
        costs = np.tile([2., 2., -1.], (250, 1))
        costs[50:200] = 0
        self.assertTrue((decode(costs, FPS, DEFAULT) == 2).all())

    def test_stoppage_splits_same_team_possessions(self):
        costs = np.tile([-1., 1., 2.], (300, 1))
        costs[100:200] = [2., 2., -1.]
        expected = np.r_[np.zeros(100), np.full(100, 2), np.zeros(100)]
        np.testing.assert_array_equal(decode(costs, FPS, DEFAULT), expected)

    def test_decoder_rejects_nan(self):
        with self.assertRaises(ValueError):
            decode([[0, np.nan, 0]], FPS, DEFAULT)


class EndToEndSyntheticTests(unittest.TestCase):
    def test_missing_ball_is_not_dead(self):
        players, ball, slots = scene()
        ball[150:350] = np.nan
        labels, _, _ = infer_half(players, ball, slots, {"home": 0, "away": 1}, [], 0,
                                  (105, 68), FPS)
        self.assertTrue((labels == "H").all())

    def test_team_swap_symmetry(self):
        players, ball, slots = scene()
        events = [event("BUILD UP", "Home", 0, 500)]
        home, _, _ = infer_half(players, ball, slots, {"home": 0, "away": 1}, events, 0,
                                (105, 68), FPS)
        away, _, _ = infer_half(players, ball, 1 - slots, {"home": 1, "away": 0}, events, 0,
                                (105, 68), FPS)
        self.assertTrue((home == "H").all())
        self.assertTrue((away == "A").all())

    def test_ball_outside_pitch_is_dead_even_with_attack_tag(self):
        players, ball, slots = scene()
        ball[:, 1] = 70
        labels, _, _ = infer_half(players, ball, slots, {"home": 0},
                                  [event("BUILD UP", "Home", 0, 500)], 0, (105, 68), FPS)
        self.assertTrue((labels == "D").all())

    def test_stationary_ball_with_moving_players_stays_live(self):
        players, ball, slots = scene()
        labels, _, _ = infer_half(players, ball, slots, {"home": 0}, [], 0, (105, 68), FPS)
        self.assertTrue((labels == "H").all())

    def test_kickoff_comes_from_tag_not_a_game_timestamp(self):
        n = 600
        features = {key: np.zeros(n, dtype=bool) for key in ("known", "moving", "stationary", "outside")}
        tags = [event("SET PIECES", "Away", 300, 400, "Kick Off (Start)")]
        dead, seed, records = restart_evidence(tags, {"away": 1}, features, np.zeros((n, 2)),
                                               0, FPS, DEFAULT)
        self.assertEqual(records[0]["restart_frame"], 350)
        self.assertTrue((dead[:350] > 0).all())
        self.assertTrue((dead[350:] == 0).all())
        self.assertGreater(seed[350, 1], 0)

    def test_inference_never_opens_files_or_network(self):
        players, ball, slots = scene(100)
        with patch("builtins.open", side_effect=AssertionError("unexpected file access")), \
                patch("socket.socket", side_effect=AssertionError("unexpected network access")):
            infer_half(players, ball, slots, {"home": 0}, [], 0, (105, 68), FPS)


if __name__ == "__main__":
    unittest.main()
