
# agents/psrl.py
# Posterior Sampling for Reinforcement Learning (PSRL) for finite-horizon tabular MDPs.
# We assume rewards are known (as in UCBVI in AJKS Chapter 7), and place a Dirichlet prior on transitions.
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from core.mdp import TabularMDP

@dataclass
class PSRLConfig:
    alpha_dirichlet: float = 1.0   # symmetric Dirichlet prior over transitions
    seed: int = 0

class PSRLAgent:
    def __init__(self, S: int, A: int, H: int, reward: np.ndarray, cfg: PSRLConfig):
        self.S, self.A, self.H = S, A, H
        self.r = reward  # (H,S,A)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # Dirichlet counts per (h,s,a, s')
        self.alpha = np.full((H, S, A, S), cfg.alpha_dirichlet, dtype=float)
        # Running state-action-next_state counts to update posterior
        self.counts = np.zeros_like(self.alpha)

    def sample_mdp(self) -> TabularMDP:
        P = np.zeros_like(self.alpha)
        for h in range(self.H):
            for s in range(self.S):
                for a in range(self.A):
                    params = self.alpha[h, s, a] + self.counts[h, s, a]
                    P[h, s, a] = self.rng.dirichlet(params)
        return TabularMDP(S=self.S, A=self.A, H=self.H, P=P, r=self.r)

    def begin_episode(self):
        self.mdp_sample = self.sample_mdp()
        #self.V, self.Pi = self.mdp_sample.optimal_value_and_policy()
        self.V, self.Pi = self.mdp_sample.optimal_value_and_policy(rng=self.rng) 

    def act(self, h: int, s: int) -> int:
        return int(self.Pi[h, s])

    def update(self, h: int, s: int, a: int, s_next: int):
        self.counts[h, s, a, s_next] += 1.0
