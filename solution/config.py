"""Fixed engineering defaults, specified before running the competition data.

These are NOT learned or tuned parameters. Do not select new values using the
competition match, its video, or its public possession labels. Any future tuning
must use a separately licensed external match and document its provenance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    control_radius_m: float = 2.5
    max_ball_speed_m_s: float = 45.0
    interpolate_gap_s: float = 0.4
    velocity_radius_s: float = 0.2
    evidence_radius_s: float = 0.2
    moving_ball_m_s: float = 2.0
    stationary_ball_m_s: float = 0.7
    stationary_window_s: float = 2.0
    boundary_margin_m: float = 0.6
    walking_speed_m_s: float = 1.2
    min_players_for_motion: int = 8
    phase_weight: float = 0.8
    control_weight: float = 1.2
    team_switch_cost_s: float = 1.0
    dead_switch_cost_s: float = 0.65
    restart_margin_s: float = 2.0
    restart_lookback_s: float = 45.0
    restart_team_seed_s: float = 1.0


DEFAULT = Config()
