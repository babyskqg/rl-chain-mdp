
# envs/chain.py
# Deterministic chain MDP used in Osband & Van Roy (2017) empirical study (Fig. 7, 8, 9).
# S = H = N; two actions: LEFT=0, RIGHT=1. Start at state 0 each episode.
# Reward 1 only if the agent moves RIGHT in the last state and reaches the goal by the end of the episode.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple
from core.mdp import TabularMDP

LEFT = 0
RIGHT = 1

@dataclass
class ChainEnvSpec:
    N: int   # number of non-terminal states; equals horizon H
    def build_true_mdp(self) -> TabularMDP:
        S = self.N
        A = 2
        H = self.N
        # Transition kernel P[h,s,a,s']
        P = np.zeros((H, S, A, S), dtype=float)
        # Deterministic dynamics:
        # RIGHT: s -> min(s+1, S-1)
        # LEFT : s -> max(s-1, 0)
        for h in range(H):
            for s in range(S):
                # Right
                s_r = min(s+1, S-1)
                P[h, s, RIGHT, s_r] = 1.0
                # Left
                s_l = max(s-1, 0)
                P[h, s, LEFT, s_l] = 1.0
        # Reward only when taking RIGHT from the last state and it's the last step (i.e., h == H-1, s == S-1, a==RIGHT)
        r = np.zeros((H, S, A), dtype=float)
        r[H-1, S-1, RIGHT] = 1.0
        return TabularMDP(S=S, A=A, H=H, P=P, r=r)

class ChainEnvRunner:
    """Simulator that keeps track of state for one episode under the *true* MDP."""
    def __init__(self, spec: ChainEnvSpec, rng: np.random.Generator):
        self.spec = spec
        self.true_mdp = spec.build_true_mdp()
        self.rng = rng
        self.reset()

    def reset(self) -> int:
        self.t = 0
        self.s = 0  # start at leftmost state
        return self.s

    def step(self, a: int) -> Tuple[int, float, bool]:
        """Advance environment by one step, returning (next_state, reward, done)."""
        h = self.t
        s = self.s
        next_s_probs = self.true_mdp.P[h, s, a]
        s_next = int(np.argmax(next_s_probs))  # deterministic
        r = float(self.true_mdp.r[h, s, a])
        self.t += 1
        self.s = s_next
        done = (self.t >= self.true_mdp.H)
        return s_next, r, done

    def horizon(self) -> int:
        return self.true_mdp.H
