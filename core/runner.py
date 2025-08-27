
# core/runner.py
# High-level experiment driver: runs episodes, logs regret, aggregates results.
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from core.mdp import TabularMDP, regret_one_episode, sample_next_state
from envs.chain import ChainEnvSpec, ChainEnvRunner, LEFT, RIGHT
from agents.psrl import PSRLAgent, PSRLConfig
from agents.ucbvi import UCBVIAgent, UCBVIConfig
from agents.q_learning import QLearningAgent, QLearningConfig

@dataclass
class ExperimentConfig:
    algo_name: str
    seed: int
    N: int
    episodes: int
    ucbvi_variant: str = "CH"   # "CH" or "BF"
    ucbvi_delta: float = 1e-6
    q_alpha: float = 0.5
    q_alpha_decay: float = 0.0
    q_epsilon: float = 0.1
    q_epsilon_decay: float = 0.0
    q_min_epsilon: float = 0.0
    
    
def make_agent(algo_name: str, true_mdp: TabularMDP, cfg: ExperimentConfig):
    S, A, H = true_mdp.S, true_mdp.A, true_mdp.H
    if algo_name == "PSRL":
        return PSRLAgent(S, A, H, true_mdp.r.copy(), PSRLConfig(seed=cfg.seed))
    elif algo_name == "UCBVI-CH":
        return UCBVIAgent(S, A, H, true_mdp.r.copy(),
                          UCBVIConfig(variant="CH", delta=cfg.ucbvi_delta, K=cfg.episodes, seed=cfg.seed))
    elif algo_name == "UCBVI-BF":
        return UCBVIAgent(S, A, H, true_mdp.r.copy(),
                          UCBVIConfig(variant="BF", delta=cfg.ucbvi_delta, K=cfg.episodes, seed=cfg.seed))
    elif algo_name == "Q-Learning":
        return QLearningAgent(S, A, H, QLearningConfig(alpha=cfg.q_alpha, alpha_decay=cfg.q_alpha_decay,
                                                       epsilon=cfg.q_epsilon, epsilon_decay=cfg.q_epsilon_decay,
                                                       min_epsilon=cfg.q_min_epsilon, seed=cfg.seed))
    else:
        raise ValueError(f"Unknown algo {algo_name}")


def run_single(cfg: ExperimentConfig) -> Dict[str, Any]:
    # Build environment and true MDP
    rng = np.random.default_rng(cfg.seed)
    spec = ChainEnvSpec(N=cfg.N)
    env = ChainEnvRunner(spec, rng)
    true_mdp = env.true_mdp

    # Agent
    agent = make_agent(cfg.algo_name, true_mdp, cfg)

    regrets = []
    actions_all = []
    states_all = []

    for ep in range(cfg.episodes):
        states = np.zeros(true_mdp.H+1, dtype=int)
        actions = np.zeros(true_mdp.H, dtype=int)

        s = env.reset()
        agent.begin_episode()
        states[0] = s

        for h in range(true_mdp.H):
            a = agent.act(h, s)
            s_next, r, done = env.step(a)
            if cfg.algo_name.startswith("Q-Learning"):
                agent.update(h, s, a, s_next, r, done)
            else:
                agent.update(h, s, a, s_next)
            states[h+1] = s_next
            actions[h] = a
            s = s_next
            if done: break

        if cfg.algo_name.startswith("Q-Learning"):
            agent.end_episode()

        reg = regret_one_episode(true_mdp, actions, states)
        regrets.append(reg)
        actions_all.append(actions.copy())
        states_all.append(states.copy())

    return {
        "cfg": cfg,
        "regrets": np.array(regrets),
        "actions": np.array(actions_all),
        "states": np.array(states_all),
    }

def compute_learning_time(regrets: np.ndarray, threshold: float = 0.1) -> int:
    """Return minimal K with (1/K) sum_{k<=K} Δ_k <= threshold; returns len(regrets)+1 if not reached."""
    cum = np.cumsum(regrets)
    K = np.arange(1, len(regrets)+1)
    avg = cum / K
    mask = np.where(avg <= threshold)[0]
    if mask.size == 0:
        return len(regrets) + 1
    return int(mask[0] + 1)
