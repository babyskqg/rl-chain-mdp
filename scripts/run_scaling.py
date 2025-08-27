# --- Allow running this file directly in VS Code (no terminal needed) ---
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------------------------------------------

# scripts/run_scaling.py
# Run the Fig. 9-style scaling experiment from a YAML config.
import os, sys, platform
import numpy as np

if __name__ == "__main__":
    if platform.system() == "Windows":
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

import yaml
from core.runner import ExperimentConfig, run_single, compute_learning_time
from core.plotting import plot_figure9_scaling, km_median




# --- Keep the machine awake for long runs (Windows/macOS/Linux) ---
def _keep_awake():
    import atexit, os, platform, subprocess
    sysname = platform.system()

    if sysname == "Windows":
        import ctypes
        ES_CONTINUOUS       = 0x80000000
        ES_SYSTEM_REQUIRED  = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        ES_AWAYMODE_REQUIRED = 0x00000040  # may be ignored on some systems

        # Try to keep system awake without forcing the display on; fall back if needed.
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
        except Exception:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
        atexit.register(lambda: ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS))

    elif sysname == "Darwin":  # macOS
        # Tie caffeinate to this Python process; exits automatically when we exit.
        try:
            p = subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
            atexit.register(lambda: p.terminate())
        except FileNotFoundError:
            pass  # caffeinate should exist on macOS; if not, we do nothing.

    elif sysname == "Linux":
        # Requires systemd-inhibit (present on most systemd-based distros).
        try:
            p = subprocess.Popen([
                "systemd-inhibit", "--what=sleep", "--why=long-python-run",
                "bash", "-lc", f"while kill -0 {os.getpid()}; do sleep 60; done"
            ])
            atexit.register(lambda: p.terminate())
        except FileNotFoundError:
            pass  # No inhibitor available; OS power settings will govern.

_keep_awake()
# -------------------------------------------------------------------



def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    episodes = int(cfg["episodes"])
    seeds = list(cfg["seeds"])
    Ns = list(cfg["Ns"])
    algos = list(cfg["algorithms"])
    qcfg = cfg.get("q_learning", {})
    thresh = float(cfg.get("threshold", 0.1))

    # -------------------- ONLY CHANGE: make csv_path optional --------------------
    # Read plot/csv targets from YAML, but fall back to a default CSV path if missing.
    plot_path = cfg["plot_path"]
    csv_path = cfg.get("csv_path", "outputs/csv/fig9_learning_times.csv")

    # Timestamped output subfolders to avoid overwriting previous runs
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    _plot_dir = os.path.join(os.path.dirname(plot_path), ts)
    _csv_dir = os.path.join(os.path.dirname(csv_path), ts)
    os.makedirs(_plot_dir, exist_ok=True)
    os.makedirs(_csv_dir, exist_ok=True)
    plot_path_ts = os.path.join(_plot_dir, os.path.basename(plot_path))
    csv_path_ts = os.path.join(_csv_dir, os.path.basename(csv_path))
    # -----------------------------------------------------------------------------


    learning_times = {algo: {} for algo in algos}
    for algo in algos:
        for N in Ns:
            times = []
            cens = []
            for seed in seeds:
                ecfg = ExperimentConfig(
                    algo_name=algo, seed=seed, N=N, episodes=episodes,
                    ucbvi_variant="BF" if algo=="UCBVI-BF" else "CH",
                    ucbvi_delta=float(cfg.get("ucbvi", {}).get("delta", 1e-6)),  # pass δ from YAML if present
                    q_alpha=qcfg.get("alpha", 0.5),
                    q_alpha_decay=qcfg.get("alpha_decay", 0.0),
                    q_epsilon=qcfg.get("epsilon", 0.1),
                    q_epsilon_decay=qcfg.get("epsilon_decay", 0.0),
                    q_min_epsilon=qcfg.get("min_epsilon", 0.0),
                )
                out = run_single(ecfg)
                K_star = compute_learning_time(out["regrets"], threshold=thresh)   # > episodes → not learned
                t = min(K_star, episodes)
                times.append(float(t))
                cens.append(bool(K_star > episodes))
            learning_times[algo][N] = {"times": times, "censored": cens}

    # plot and save
    os.makedirs(os.path.dirname(plot_path_ts), exist_ok=True)
    plot_figure9_scaling(learning_times, plot_path_ts)

    os.makedirs(os.path.dirname(csv_path_ts), exist_ok=True)
    # Write CSV: rows algorithms, columns Ns (KM medians)
    Ns_sorted = sorted(Ns)
    header = "algorithm," + ",".join(str(n) for n in Ns_sorted)
    lines = [header]
    for algo in algos:
        row = [algo]
        for n in Ns_sorted:
            entry = learning_times[algo].get(n, {"times": [], "censored": []})
            med, med_cens = km_median(entry["times"], entry["censored"])
            row.append(">{}".format(med) if med_cens else str(med))
        lines.append(",".join(row))
    with open(csv_path_ts, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved figure to {plot_path_ts} and CSV to {csv_path_ts}")


if __name__ == "__main__":
    default_cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "scaling.yaml")
    main(os.path.abspath(default_cfg))
