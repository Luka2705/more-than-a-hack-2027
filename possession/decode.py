"""Global three-state (H, A, D) Viterbi decoding of fused per-frame evidence.

Emission scores per frame: H = +e/2, A = -e/2, D = z, where e is the team log-odds and z the
dead-ball log-odds. Transitions carry fixed penalties (H<->A, alive->D, D->alive). There is no
duration model: a change is accepted whenever the evidence accumulated over the new state beats
the penalty, so short possessions survive if their evidence is real. Each half is decoded
independently; nothing crosses the halftime hole.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])
NEG = -1e18


def viterbi(scores, p_switch, p_to_dead, p_from_dead, start=None):
    """scores [T,3] -> best path [T] int8. Pure Python inner loop on floats (fast enough)."""
    T = len(scores)
    cost = ((0.0, p_switch, p_to_dead),
            (p_switch, 0.0, p_to_dead),
            (p_from_dead, p_from_dead, 0.0))
    s = scores.tolist()
    v0, v1, v2 = s[0]
    if start is not None:
        v0, v1, v2 = [x if k == start else NEG for k, x in enumerate((v0, v1, v2))]
    back = np.zeros((T, 3), dtype=np.int8)
    bk = [None] * T
    c01, c02, c10, c12, c20, c21 = cost[0][1], cost[0][2], cost[1][0], cost[1][2], cost[2][0], \
        cost[2][1]
    for t in range(1, T):
        e0, e1, e2 = s[t]
        # to H
        a, b, c = v0, v1 - c10, v2 - c20
        if a >= b and a >= c:
            n0, b0 = a, 0
        elif b >= c:
            n0, b0 = b, 1
        else:
            n0, b0 = c, 2
        a, b, c = v0 - c01, v1, v2 - c21
        if b >= a and b >= c:
            n1, b1 = b, 1
        elif a >= c:
            n1, b1 = a, 0
        else:
            n1, b1 = c, 2
        a, b, c = v0 - c02, v1 - c12, v2
        if c >= a and c >= b:
            n2, b2 = c, 2
        elif a >= b:
            n2, b2 = a, 0
        else:
            n2, b2 = b, 1
        v0, v1, v2 = n0 + e0, n1 + e1, n2 + e2
        bk[t] = (b0, b1, b2)
    path = np.empty(T, dtype=np.int8)
    last = int(np.argmax([v0, v1, v2]))
    path[-1] = last
    for t in range(T - 1, 0, -1):
        last = bk[t][last]
        path[t - 1] = last
    return path


def emission_scores(e, z, forced_dead=None, forced_alive=None):
    sc = np.stack([0.5 * e, -0.5 * e, z], axis=1)
    if forced_dead is not None:
        sc[forced_dead, 0] = NEG / 1e6
        sc[forced_dead, 1] = NEG / 1e6
    if forced_alive is not None:
        sc[forced_alive, 2] = NEG / 1e6
    return sc


def decode_match(match, e, z, p):
    """Labels [N] of 'H','A','D','X'; X outside the graded halves."""
    labels = np.full(match.n, "X", dtype="<U1")
    for h, a, b in match.halves:
        ko = match.kickoffs.get(h, a)
        forced_dead = np.zeros(b - a + 1, dtype=bool)
        forced_dead[:max(0, ko - a)] = True
        forced_alive = np.zeros(b - a + 1, dtype=bool)
        if ko - a < len(forced_alive):
            forced_alive[ko - a] = True  # the kickoff frame is live by definition
        sc = emission_scores(e[a:b + 1], z[a:b + 1], forced_dead, forced_alive)
        path = viterbi(sc, p["p_switch"], p["p_to_dead"], p["p_from_dead"])
        labels[a:b + 1] = LETTERS[path]
    return labels
