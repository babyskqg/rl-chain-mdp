# core/mdp.py
# Finite-horizon tabular MDP utilities: value iteration, regret computation, RNG helpers.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple

def rand_argmax_1d(x: np.ndarray, rng: np.random.Generator) -> int:
    """Return a uniformly random index among ties of the maximum."""
    m = np.max(x)
    ties = np.flatnonzero(x == m)
    return int(rng.choice(ties))

def rand_argmax_rows(Q: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise randomized argmax for a 2D array (S x A)."""
    S = Q.shape[0]
    out = np.zeros(S, dtype=int)
    m = Q.max(axis=1, keepdims=True)
    tie_mask = (Q == m)
    for s in range(S):
        choices = np.flatnonzero(tie_mask[s])
        out[s] = int(rng.choice(choices))
    return out

@dataclass
class TabularMDP:
    S: int                    # number of states (1..S). We'll index states as 0..S-1 internally.
    A: int                    # number of actions (0..A-1)
    H: int                    # horizon (timesteps per episode)
    P: np.ndarray            # shape (H, S, A, S) transition probabilities
    r: np.ndarray            # shape (H, S, A) expected rewards in [0,1]

    def optimal_value_and_policy(self, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Backward DP to compute optimal value and greedy policy (random tie-breaking).
        Returns:
            V: (H+1, S) value-to-go with V[H] = 0
            Pi: (H, S) int actions
        """
        if rng is None:
            rng = np.random.default_rng()
        H, S, A = self.H, self.S, self.A
        V = np.zeros((H+1, S), dtype=float)
        Pi = np.zeros((H, S), dtype=int)
        for h in range(H-1, -1, -1):
            # Q_h(s,a) = r_h(s,a) + P_h(s,a)^T V_{h+1}
            Q = self.r[h] + self.P[h].dot(V[h+1])
            Pi[h] = rand_argmax_rows(Q, rng)                # <-- randomized tie-breaking
            V[h] = Q[np.arange(S), Pi[h]]
        return V, Pi

def regret_one_episode(mdp_true: TabularMDP, actions: np.ndarray, states: np.ndarray) -> float:
    """Episode regret: V^*_1(s1) - sum_{h=1..H} r(s_h,a_h).
    - states: (H+1,) visited states indices
    - actions: (H,) actions taken
    Uses the *true* MDP reward for the realized trajectory.
    """
    V_opt, _ = mdp_true.optimal_value_and_policy()
    s1 = states[0]
    optimal_value = V_opt[0, s1]
    # realized return under true rewards
    rew = 0.0
    for h in range(mdp_true.H):
        rew += mdp_true.r[h, states[h], actions[h]]
    return float(optimal_value - rew)

def sample_next_state(P_row: np.ndarray, rng: np.random.Generator) -> int:
    """Sample next state index given a probability vector over S."""
    return int(rng.choice(P_row.shape[0], p=P_row))

def clip_bonus(b: np.ndarray, H_remaining: int) -> np.ndarray:
    """Clip optimism bonuses to keep Q in [0, H_remaining]."""
    return np.minimum(b, H_remaining * np.ones_like(b))
