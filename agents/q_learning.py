
# agents/q_learning.py
# Episodic tabular Q-learning with epsilon-greedy exploration ("epsilon dithering").
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from core.mdp import rand_argmax_1d

@dataclass
class QLearningConfig:
    alpha: float = 0.5           # learning rate (constant or initial value for schedule)
    alpha_decay: float = 0.0     # optional decay per episode
    gamma: float = 1.0
    epsilon: float = 0.1
    min_epsilon: float = 0.0
    epsilon_decay: float = 0.0   # per-episode multiplicative decay
    seed: int = 0

class QLearningAgent:
    def __init__(self, S: int, A: int, H: int, cfg: QLearningConfig):
        self.S, self.A, self.H = S, A, H
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # time-dependent Q-table (H,S,A)
        self.Q = np.zeros((H, S, A), dtype=float)

    def begin_episode(self):
        pass
    
    def act(self, h: int, s: int) -> int:
        if self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(self.A))
        return rand_argmax_1d(self.Q[h, s], self.rng)

    def update(self, h: int, s: int, a: int, s_next: int, r: float, done: bool):
        # Q-learning target: r + max_a' Q[h+1, s_next, a']
        target = r + (0.0 if done else np.max(self.Q[h+1, s_next]))
        # Constant or decaying learning rate
        alpha = self.cfg.alpha
        self.Q[h, s, a] += alpha * (target - self.Q[h, s, a])

    def end_episode(self):
        # decay exploration and learning rate if configured
        if self.cfg.epsilon_decay > 0.0:
            self.cfg.epsilon = max(self.cfg.min_epsilon, self.cfg.epsilon * (1.0 - self.cfg.epsilon_decay))
        if self.cfg.alpha_decay > 0.0:
            self.cfg.alpha = max(1e-3, self.cfg.alpha * (1.0 - self.cfg.alpha_decay))
