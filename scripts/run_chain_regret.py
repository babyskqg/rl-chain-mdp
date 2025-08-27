# --- Allow running this file directly in VS Code (no terminal needed) ---
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ------------------------------------------------------------------------

# scripts/run_chain_regret.py
# Run the Fig. 8-style experiment from a YAML config.
import os, sys, platform, json
import numpy as np

# OS/multiprocessing niceties for "Run" from VS Code (no terminal needed)
if __name__ == "__main__":
    if platform.system() == "Windows":
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

import yaml
from core.runner import ExperimentConfig, run_single
from core.plotting import plot_figure8_panels



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

    
    # Timestamped output subfolders to avoid overwriting previous runs
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # Figure path
    _plot_dir = os.path.join(os.path.dirname(cfg['plot_path']), ts)
    os.makedirs(_plot_dir, exist_ok=True)
    plot_path_ts = os.path.join(_plot_dir, os.path.basename(cfg['plot_path']))
    # CSV prefix
    _prefix = cfg.get('csv_prefix', 'outputs/csv/fig8')
    _csv_dir = os.path.join(os.path.dirname(_prefix), ts)
    os.makedirs(_csv_dir, exist_ok=True)
    csv_prefix_ts = os.path.join(_csv_dir, os.path.basename(_prefix))

    episodes = int(cfg["episodes"])
    seeds = list(cfg["seeds"])
    Ns = list(cfg["Ns"])
    algos = list(cfg["algorithms"])
    qcfg = cfg.get("q_learning", {})
    curves_by_N = {}

    for N in Ns:
        algo_to_curves = {}
        for algo in algos:
            reg_all_seeds = []
            for seed in seeds:
                ecfg = ExperimentConfig(
                    algo_name=algo, seed=seed, N=N, episodes=episodes,
                    ucbvi_variant="BF" if algo=="UCBVI-BF" else "CH",
                    q_alpha=qcfg.get("alpha", 0.5),
                    q_alpha_decay=qcfg.get("alpha_decay", 0.0),
                    q_epsilon=qcfg.get("epsilon", 0.1),
                    q_epsilon_decay=qcfg.get("epsilon_decay", 0.0),
                    q_min_epsilon=qcfg.get("min_epsilon", 0.0),
                )
                out = run_single(ecfg)
                reg_all_seeds.append(out["regrets"])
            algo_to_curves[algo] = np.array(reg_all_seeds)
        curves_by_N[N] = algo_to_curves

    # plotting
    os.makedirs(os.path.dirname(plot_path_ts), exist_ok=True)
    plot_figure8_panels(curves_by_N, episodes, plot_path_ts)

    # save CSVs
    prefix = csv_prefix_ts
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    for N, d in curves_by_N.items():
        for algo, arr in d.items():
            outpath = f"{prefix}_N{N}_{algo.replace('-', '')}.csv"
            # each row a seed, columns per episode (regret per-episode)
            #np.savetxt(outpath, arr.mean(axis=0)[None, :], delimiter=",")
            np.savetxt(outpath, arr, delimiter=",")
    print(f"Saved figure to {cfg['plot_path']}")

if __name__ == "__main__":
    # Default to project-relative path when launched in VS Code
    default_cfg = os.path.join(os.path.dirname(__file__), "..", "configs", "chain_regret.yaml")
    main(os.path.abspath(default_cfg))
