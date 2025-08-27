# agents/ucbvi.py
# UCBVI variants for episodic finite-horizon tabular MDPs.
# - UCBVI-CH: Hoeffding-style bonuses (Chapter 7 of AJKS book)
# - UCBVI-BF: Bernstein/Freedman (variance-aware) bonuses (Azar et al., 2017)
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal

# clip_bonus is no longer used after the fix, but left here to keep the file identical elsewhere.
from core.mdp import TabularMDP, clip_bonus, rand_argmax_rows


@dataclass
class UCBVIConfig:
    variant: Literal["CH", "BF"] = "CH"
    delta: float = 1e-6             # global failure prob for the whole run
    K: int = 1                      # episode budget for this run (used in log term)
    seed: int = 0


class UCBVIAgent:
    def __init__(self, S: int, A: int, H: int, reward: np.ndarray, cfg: UCBVIConfig):
        self.S, self.A, self.H = S, A, H
        self.r = reward  # shape (H, S, A); assumed known and in [0,1]
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # counts and empirical transitions
        self.n = np.zeros((H, S, A), dtype=int)          # visit counts n_h(s,a)
        self.N = np.zeros((H, S, A, S), dtype=int)       # next-state counts N_h(s,a,s')
        self.Phat = np.zeros((H, S, A, S), dtype=float)  # empirical transitions

        # policy/value placeholders (filled each episode)
        self.Pi = np.zeros((H, S), dtype=int)
        self.V = np.zeros((H + 1, S), dtype=float)

    # --- helpers ----------------------------------------------------------------

    def _log_term(self) -> float:
        """Return the log factor L used in the confidence bonuses.

        AJKS Ch.7 (CH):  L = ln( S*A*H*K / δ )
        Azar et al. 2017 (BF): L = ln( 2*S*A*H*K / δ )
        A small floor is applied for numerical stability.
        """
        S, A, H, K = self.S, self.A, self.H, max(1, int(self.cfg.K))
        delta = max(min(self.cfg.delta, 1.0), 1e-12)
        if self.cfg.variant == "CH":
            L = np.log((S * A * H * K) / delta)
        else:
            L = np.log((2 * S * A * H * K) / delta)
        return float(max(L, 1.0))

    # --- required public API ----------------------------------------------------

    def begin_episode(self):
        """Compute an optimistic policy via backward DP with UCB bonuses."""
        H, S, A = self.H, self.S, self.A
        V = np.zeros((H + 1, S), dtype=float)
        self.Pi = np.zeros((H, S), dtype=int)
        log_term = self._log_term()

        for h in range(H - 1, -1, -1):
            Q = np.zeros((S, A), dtype=float)
            Hrem = self.H - h
            for s in range(S):
                for a in range(A):
                    n_sa = self.n[h, s, a]
                    n_eff = max(1, n_sa)
                    # Use empirical transition if we have samples; otherwise uniform.
                    p = self.Phat[h, s, a] if n_sa > 0 else (np.ones(S) / S)
                    mean = self.r[h, s, a] + p.dot(V[h + 1])  # expected return without bonus

                    if self.cfg.variant == "CH":
                        # Hoeffding-style bonus (AJKS Ch. 7)
                        bonus = Hrem * np.sqrt((2.0 * log_term) / n_eff)
                    else:
                        # Bernstein/Freedman bonus (Azar et al., 2017)
                        EV = p.dot(V[h + 1])
                        EV2 = p.dot(V[h + 1] ** 2)
                        var = max(EV2 - EV ** 2, 0.0)
                        bonus = np.sqrt((2.0 * var * log_term) / n_eff) + (7.0 / 3.0) * Hrem * log_term / max(n_sa - 1, 1)

                    # --------- FIX: CLIP THE Q-VALUE, NOT THE BONUS -----------------
                    # Ensure the optimistic Q-value remains in [0, H-h].
                    Q[s, a] = float(np.minimum(mean + bonus, Hrem))
                    Q[s, a] = max(0.0, Q[s, a])
                    # ----------------------------------------------------------------

            # randomized tie-breaking for greedy policy
            self.Pi[h] = rand_argmax_rows(Q, self.rng)
            V[h] = Q[np.arange(S), self.Pi[h]]

        self.V = V

    def act(self, h: int, s: int) -> int:
        return int(self.Pi[h, s])

    def update(self, h: int, s: int, a: int, s_next: int):
        """Model update from a transition (h,s,a,s'). Rewards are known."""
        self.n[h, s, a] += 1
        self.N[h, s, a, s_next] += 1
        self.Phat[h, s, a] = self.N[h, s, a] / max(1, self.n[h, s, a])
