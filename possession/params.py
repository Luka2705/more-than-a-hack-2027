"""Every tunable number in one place. Values here are the ones the build uses.

Values are the output of possession/validation/tune.py (source-balanced objective, round 1),
copied from possession/validation/tuned.json. Provenance in notes/decisions.md. All tuning is done on external
matches (possession/validation/); nothing here is fitted to the BAR-ATH public window.
"""

DEFAULT = {
    # -- team evidence (touches.py) --
    "ctrl_r": 0.5625,        # m, distance at which a player is considered in control range
    "ctrl_s": 0.0656,       # m, softness of that edge
    "w_near": 2.0,        # per-frame log-odds weight of exclusive proximity
    "w_flight": 0.2,     # per-frame bias towards the last-touch team while the ball is free
    "w_gap": 0.0,        # per-frame bias towards the last-touch team while the ball is missing
    "w_geom": 0.532,        # weight of the player-only classifier on missing-ball frames
    "w_geom_vis": 0.1,    # weight of the player-only classifier on visible-ball frames
    "w_team": 0.4,        # weight of the learned geometry+ball team logit (teamnet.py)
    # -- dead-ball evidence (deadball.py) --
    "out_margin": 0.5,    # m beyond the lines before the ball counts as out
    "w_out": 2.66,
    "rest_v": 0.7,        # m/s ball speed counted as at rest
    "rest_win": 37,       # frames; ball at rest across a centred window of this length
    "w_rest": 0.8,
    "lull_v": 1.6,        # m/s mean player speed at which the lull term is zero
    "lull_win": 75,       # frames, smoothing of mean player speed
    "w_lull": 0.225,        # per m/s below lull_v
    "w_miss": 0.025,       # ball missing, weak prior
    "w_move": 0.9,        # ball visible, moving and inside the pitch: evidence for alive
    "move_v": 3.0,
    "w_tag": 0.0,         # set-piece tag window prior
    "d_bias": 1.95,       # constant offset on D
    "w_hand": 0.665,        # weight of the hand-built dead-ball evidence above
    "w_dnet": 0.8,        # weight of the learned dead-ball logit (deadnet.py)
    "dnet_bias": 1.0,     # offset added to that logit before weighting
    # -- decoder (decode.py) --
    "p_switch": 7.98,     # H<->A
    "p_to_dead": 80.0,    # alive -> D
    "p_from_dead": 48.0,   # D -> alive
}


def merged(overrides=None):
    p = dict(DEFAULT)
    if overrides:
        p.update(overrides)
    return p
