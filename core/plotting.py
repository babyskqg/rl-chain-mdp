# core/plotting.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ---------- Kaplan–Meier median for right-censored samples ----------

def km_median(times: List[float], censored: List[bool]) -> Tuple[float, bool]:
    """
    Kaplan–Meier median for right-censored data.
    times[i] = observed time or censoring time
    censored[i] = True if the i-th observation is censored at times[i]

    Returns (median, is_censored_median). If the survival never drops below 0.5
    within the observation window, we return (max time, True) to indicate
    a right-censored median.
    """
    if len(times) == 0:
        return (np.nan, True)
    t = np.asarray(times, dtype=float)
    c = np.asarray(censored, dtype=bool)

    # sort by time
    order = np.argsort(t)
    t, c = t[order], c[order]

    uniq = np.unique(t)
    n = len(t)
    at_risk = n
    S = 1.0

    for ti in uniq:
        # events (learned) at ti are those with time==ti & not censored
        d_i = int(np.sum((t == ti) & (~c)))
        # censorings at ti
        c_i = int(np.sum((t == ti) & (c)))

        if at_risk > 0 and d_i > 0:
            S *= (1.0 - d_i / at_risk)
        if S <= 0.5:
            return float(ti), False

        at_risk -= (d_i + c_i)

    # never crossed 0.5 within window => right-censored median
    return float(np.max(t)), True

# ---------- Fig. 8 panels: cumulative regret vs episode ----------

def plot_figure8_panels(curves_by_N: Dict[int, Dict[str, np.ndarray]],
                        episodes: int,
                        save_path: str) -> None:
    """
    One panel per N; each line is mean cumulative regret across seeds.
    curves_by_N[N][algo] = (num_seeds, episodes) of per-episode regrets.
    """
    Ns = sorted(curves_by_N.keys())
    ncols = 2
    nrows = int(np.ceil(len(Ns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows), squeeze=False)
    algo_order = ["PSRL", "UCBVI-BF", "UCBVI-CH", "Q-Learning"]

    for idx, N in enumerate(Ns):
        ax = axes[idx // ncols][idx % ncols]
        data = curves_by_N[N]
        for algo in algo_order:
            if algo in data:
                reg = data[algo].mean(axis=0)  # mean across seeds
                ax.plot(np.arange(1, len(reg) + 1), np.cumsum(reg), label=algo)
        ax.set_title(f"N = {N}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative regret")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    # hide any unused axes
    total = nrows * ncols
    for j in range(len(Ns), total):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# ---------- Fig. 9: learning time scaling (log–log) ----------

def _plot_scaling_on_ax(ax: plt.Axes,
                        learning_times: Dict[str, Dict[int, Dict[str, List[float]]]]
                        ) -> None:
    """
    Draw log(learning time) vs log(N) with:
      • points for each seed at each N (censored shown as open triangles)
      • dashed slope line fit to **KM medians** over seeds at each N
    learning_times[algo][N] = {"times": [...], "censored": [...]}
    """
    eps = 1e-12
    algo_order = ["PSRL", "UCBVI-BF", "UCBVI-CH", "Q-Learning"]

    # get a color for each algorithm from Matplotlib's cycle, but keep stable across scatters
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = (prop_cycle.by_key().get("color", []) if prop_cycle is not None else [])
    color_map = {}

    for i, algo in enumerate(algo_order):
        if algo not in learning_times:
            continue
        d = learning_times[algo]

        # assign a stable color for this algo
        c = colors[i % len(colors)] if colors else None
        color_map[algo] = c

        Ns_sorted = sorted(d.keys())
        xs_obs, ys_obs, xs_cen, ys_cen = [], [], [], []
        Ns_med, T_med = [], []

        for N in Ns_sorted:
            entry = d[N]
            times = np.asarray(entry["times"], float)
            cens = np.asarray(entry["censored"], bool)

            # KM median at this N
            med, med_cens = km_median(times.tolist(), cens.tolist())
            if med_cens:
                xs_cen.append(np.log(N)); ys_cen.append(np.log(max(med, eps)))
            else:
                xs_obs.append(np.log(N)); ys_obs.append(np.log(max(med, eps)))
                Ns_med.append(N); T_med.append(med)

        # scatter the points (observed and censored)
        if xs_obs:
            ax.scatter(xs_obs, ys_obs, label=algo, s=28, alpha=0.9, edgecolors="none", c=c)
        if xs_cen:
            ax.scatter(xs_cen, ys_cen, s=42, facecolors="none", edgecolors=c, marker="v", alpha=0.9)

        # fit a dashed slope line through KM medians if we have >= 2 medians
        if len(Ns_med) >= 2:
            Ns_med = np.array(Ns_med, float)
            T_med = np.array(T_med, float)
            X = np.vstack([np.log(Ns_med), np.ones_like(Ns_med)]).T
            m, b = np.linalg.lstsq(X, np.log(np.maximum(T_med, eps)), rcond=None)[0]
            xline = np.linspace(np.log(Ns_med.min()), np.log(Ns_med.max()), 100)
            ax.plot(xline, m * xline + b, linestyle="--", linewidth=1.5, c=c)
            ax.text(xline[-1], m * xline[-1] + b, f"slope≈{m:.2f}",
                    ha="right", va="bottom", color=c)

def plot_figure9_scaling(learning_times: Dict[str, Dict[int, Dict[str, List[float]]]],
                         save_path: str) -> None:
    """Wrapper that owns the Figure for the Fig. 9-style plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_scaling_on_ax(ax, learning_times)
    ax.set_xlabel("log N")
    ax.set_ylabel("log learning time")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

# Backward-compatible alias so either name works from scripts
def plot_scaling(learning_times: Dict[str, Dict[int, Dict[str, List[float]]]],
                 save_path: str) -> None:
    """Alias kept for compatibility with older scripts."""
    plot_figure9_scaling(learning_times, save_path)
