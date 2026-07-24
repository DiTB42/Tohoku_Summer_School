"""Benchmark saved SAC checkpoints by how MANY STEPS they need to reach the goal.

For a set of training checkpoints this rolls each model out on the SAME battery of
N reproducible test mazes and records how many **RL steps** the snake needs to reach
the goal. It plots the mean steps-to-goal (averaged over the N mazes) as a function
of training steps. Fewer steps = a better, more direct policy.

An episode that never reaches the goal within --max-steps is a failure; its step
count is capped at --max-steps (so a never-finishing model reads as "took the maximum
number of steps" rather than being dropped from the average). The per-model success
count is also recorded in the CSV so you can see how much of the mean is real
finishing vs. capped failures.

Speed
-----
The whole point of this rewrite is to be fast:
  * By default it evaluates a small fixed set of ~11 roughly log-spaced checkpoints
    (1k, 5k, 10k, 20k, 50k, 75k, 100k, 200k, 500k, 750k, 1M), snapped to the nearest
    checkpoints on disk (there can be 500+ — running them all is what made the old
    version unusably slow). Use the override flags below for a dense sweep.
  * Models are evaluated **in parallel across CPU cores** (one MuJoCo env per worker
    process). Physics is the bottleneck and the installed torch is CPU-only, so this
    scales close to linearly with cores. `--jobs 1` forces the old serial behavior.

Reproducibility: every model is evaluated on the identical maze battery. Maze i is
regenerated from a fixed seed (--seed + i) by re-seeding BOTH Python's global `random`
(which drives the maze layout in mazes/make_maze.create_maze_layout) and the env's
Gymnasium RNG (which drives goal-cell selection) right before each reset. Each worker
process has its own `random` state, so the battery is byte-identical for every model
regardless of --jobs.

Examples
--------
    # default: fixed ~11-checkpoint set (1k..1M), parallel across all cores, 30 mazes
    uv run python benchmark_models.py

    # specific checkpoints only
    uv run python benchmark_models.py --pick 1000,10000,100000,1000000 --n-mazes 20

    # dense linear sweep (old behavior): every 10k steps up to 500k
    uv run python benchmark_models.py --first-steps 10000 --step-interval 10000 \
        --last-steps 500000

    uv run python benchmark_models.py --jobs 8 --out plots/steps.png
"""
import argparse
import atexit
import glob as globmod
import multiprocessing as mp
import os
import random
import re
import warnings

import numpy as np


# --- observation layout (mirrors env_snake.SnakeEnv._get_obs) --------------
# The env always emits the full 42-D observation, but checkpoints come from
# five eras with different widths (see CLAUDE.md / swarm_core.py). The full
# 42-D layout, in order, is:
#   [ joint_pos(12) | joint_vel(12) | head->target xy(2) | orient(2)
#     | head angular velocity(3) | heading error(2) | next-turn(1)
#     | terrain friction head+tail(2) | cave width cur+next(2) | last action(4) ]
# Older policies omit the trailing-but-not-last blocks:
#   40-D: drop the 2-D cave-width block
#   38-D: also drop the 2-D terrain-friction block
#   37-D: also drop the 1-D next-turn signal
#   34-D: also drop the 3-D head angular velocity block
# We build the model-appropriate obs by slicing the full 42-D vector.
_FULL_OBS_DIM = 42
_ANGVEL_SLICE = slice(28, 31)   # head angular velocity (3)
_NEXT_TURN_IDX = 33             # next-turn signal (1)
_TERRAIN_SLICE = slice(34, 36)  # terrain friction under head, tail (2)
_WIDTH_SLICE = slice(36, 38)    # cave width current, next-path cell (2)

# Checkpoint filename -> training step count (same pattern as viz_checkpoints.py).
_STEPS_RE = re.compile(r"sac_snake_(\d+)_steps")


def adapt_obs(full_obs, obs_dim):
    """Slice the env's full 42-D observation down to what `obs_dim` expects."""
    if obs_dim == _FULL_OBS_DIM:
        return full_obs
    width_idx = list(range(_WIDTH_SLICE.start, _WIDTH_SLICE.stop))
    terrain_idx = list(range(_TERRAIN_SLICE.start, _TERRAIN_SLICE.stop))
    if obs_dim == 40:  # drop cave width only
        return np.delete(full_obs, width_idx)
    if obs_dim == 38:  # drop cave width AND terrain friction
        return np.delete(full_obs, terrain_idx + width_idx)
    if obs_dim == 37:  # drop cave width, terrain AND next-turn signal
        return np.delete(full_obs, [_NEXT_TURN_IDX] + terrain_idx + width_idx)
    if obs_dim == 34:  # drop cave width, terrain, next-turn AND head angular velocity
        drop = (list(range(_ANGVEL_SLICE.start, _ANGVEL_SLICE.stop))
                + [_NEXT_TURN_IDX] + terrain_idx + width_idx)
        return np.delete(full_obs, drop)
    raise ValueError(
        f"Unsupported policy observation dim {obs_dim}; expected 34, 37, 38, 40 or 42. "
        "Update adapt_obs() if the obs layout changed."
    )


# --- checkpoint discovery / selection -------------------------------------
def available_checkpoints(models_dir):
    """Return {training_steps: path} for every sac_snake_<steps>_steps.zip on disk."""
    found = {}
    for p in globmod.glob(os.path.join(models_dir, "sac_snake_*_steps.zip")):
        m = _STEPS_RE.search(os.path.basename(p))
        if m:
            found[int(m.group(1))] = p
    return found


# Default checkpoints when no --pick / range flags are given. Roughly log-spaced
# so the plot shows early learning (1k..) and the long tail (..1M) without
# evaluating all ~500 checkpoints. Each is snapped to the nearest checkpoint that
# actually exists on disk (see default_pick).
_DEFAULT_PICKS = [1000, 5000, 10000, 20000, 50000, 75000,
                  100000, 200000, 500000, 750000, 1000000]


def default_pick(available_steps, wanted=_DEFAULT_PICKS):
    """Snap each preferred step count in `wanted` to the nearest available one.

    Returns a sorted, de-duplicated list, so if two targets snap to the same
    checkpoint (e.g. when only coarse checkpoints exist) it appears once.
    """
    avail = sorted(set(available_steps))
    picked = []
    for target in wanted:
        nearest = min(avail, key=lambda s: abs(s - target))
        if nearest not in picked:
            picked.append(nearest)
    return sorted(picked)


def select_models(args):
    """Resolve the CLI flags to a sorted [(steps, path), ...] list to evaluate.

    Precedence: --pick  >  --first/--interval/--last (dense sweep)  >  default set.
    """
    avail = available_checkpoints(args.models_dir)
    if not avail:
        raise SystemExit(f"No sac_snake_<steps>_steps.zip files found in {args.models_dir}.")

    if args.pick:
        wanted = [int(x) for x in args.pick.split(",")]
        missing = [w for w in wanted if w not in avail]
        if missing:
            raise SystemExit(f"No checkpoint for step counts {missing} in {args.models_dir}.")
        chosen = wanted
    elif any(x is not None for x in (args.first_steps, args.step_interval, args.last_steps)):
        first = args.first_steps if args.first_steps is not None else min(avail)
        last = args.last_steps if args.last_steps is not None else max(avail)
        interval = args.step_interval if args.step_interval is not None else 2000
        chosen = [s for s in range(first, last + 1, interval) if s in avail]
        wanted_ct = len(range(first, last + 1, interval))
        if len(chosen) < wanted_ct:
            print(f"[warn] {wanted_ct - len(chosen)} step counts in "
                  f"{first}:{last}:{interval} had no checkpoint and were skipped.")
        if not chosen:
            raise SystemExit(f"No checkpoints in range {first}:{last}:{interval}.")
    else:
        chosen = default_pick(list(avail))

    return sorted((s, avail[s]) for s in chosen)


# --- rollout ---------------------------------------------------------------
def rollout_steps(model, env, obs_dim, max_steps):
    """Roll one episode (env already reset). Return (steps_to_goal, reached_goal).

    A failure (never reached the goal before truncation) is capped at `max_steps`.
    """
    obs = env._get_obs()  # env was just reset; this is the initial obs
    steps = 0
    reached = False
    while True:
        action, _ = model.predict(adapt_obs(obs, obs_dim), deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
        if terminated:
            reached = True
            break
        if truncated:
            break
    effective_steps = steps if reached else max_steps
    return effective_steps, reached


# --- worker process state --------------------------------------------------
# Each worker builds ONE headless env and reuses it across every model it is
# assigned. The env gets a UNIQUE env_index so its maze XML is written to a
# per-worker file (scenes/scene_maze_env{i}_{pid}.xml) instead of the shared
# scenes/scene_maze_trial.xml. Without this, N workers would clobber the same
# file mid-write -> NaN / corrupt-load crashes. close() (registered atexit)
# deletes the per-worker file.
_ENV = None
_MAZE_SEEDS = None
_DEVICE = None
_MAX_STEPS = None


def _init_worker(counter, config_path, no_terrain, max_steps, maze_seeds, device):
    global _ENV, _MAZE_SEEDS, _DEVICE, _MAX_STEPS
    import signal
    from env_snake import SnakeEnv
    from config_utils import load_config

    # Ignore Ctrl-C in workers so the interrupt reaches ONLY the main process,
    # which then tears the pool down. Otherwise every worker also gets the
    # KeyboardInterrupt while deep in a C-level physics loop (250x250 mj_step)
    # where it can't respond, and the run appears to hang on Ctrl-C.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    with counter.get_lock():
        env_index = counter.value
        counter.value += 1

    config = load_config(config_path)
    if no_terrain:
        config.env.terrain_enabled = False
    env = SnakeEnv(render_mode=None, config=config, env_index=env_index)
    env.max_episode_steps = max_steps
    atexit.register(env.close)

    _ENV = env
    _MAZE_SEEDS = maze_seeds
    _DEVICE = device
    _MAX_STEPS = max_steps


def _eval_model(item):
    """Worker task: evaluate one model over the full maze battery.

    Returns (training_steps, mean_steps, n_success, obs_dim).
    """
    from stable_baselines3 import SAC

    training_steps, path = item
    model = SAC.load(path, device=_DEVICE)
    obs_dim = int(model.observation_space.shape[0])

    n = len(_MAZE_SEEDS)
    print(f"  [{training_steps:>8d}] start (obs{obs_dim}, {n} mazes)", flush=True)
    step_counts = np.empty(n, dtype=np.float64)
    n_success = 0
    for i, seed in enumerate(_MAZE_SEEDS):
        # Reproducible maze battery: seed BOTH RNGs the env draws from.
        random.seed(seed)
        _ENV.reset(seed=seed)
        steps, reached = rollout_steps(model, _ENV, obs_dim, _MAX_STEPS)
        step_counts[i] = steps
        n_success += int(reached)
        # Per-maze tick so a long parallel run shows continuous progress. Each
        # line is prefixed with the model's step count since workers interleave.
        # flush=True: worker stdout is block-buffered on spawn, so without it the
        # ticks wouldn't appear until the process exits.
        print(f"  [{training_steps:>8d}] maze {i + 1}/{n}: "
              f"{steps:>4.0f} steps {'ok ' if reached else 'CAP'} "
              f"({n_success}/{i + 1} solved)", flush=True)

    return (training_steps, float(step_counts.mean()), n_success, obs_dim)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SAC checkpoints by RL-steps-to-goal over a fixed maze battery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pick", type=str, default=None,
                        help="Comma-separated step counts to evaluate, e.g. "
                             "1000,10000,100000. Overrides the default checkpoint set.")
    parser.add_argument("--first-steps", type=int, default=None,
                        help="Dense sweep: first step count (enables linear range mode).")
    parser.add_argument("--step-interval", type=int, default=None,
                        help="Dense sweep: spacing between successive step counts "
                             "(default 2000 when a range is requested).")
    parser.add_argument("--last-steps", type=int, default=None,
                        help="Dense sweep: last step count (enables linear range mode).")
    parser.add_argument("--n-mazes", type=int, default=30,
                        help="Number of reproducible test mazes each model is run on.")
    parser.add_argument("--max-steps", type=int, default=250,
                        help="Max RL steps per episode before truncation. Failures "
                             "are counted at this cap.")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Base seed; maze i uses seed+i (identical for every model).")
    parser.add_argument("--jobs", type=int, default=None,
                        help="Parallel worker processes (default: min(cpu_count, n_models)). "
                             "Use 1 for serial.")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="YAML config the models were trained with.")
    parser.add_argument("--models-dir", type=str, default="models/checkpoints",
                        help="Directory holding sac_snake_<steps>_steps.zip files.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for SAC.load ('cpu', 'cuda', 'auto').")
    parser.add_argument("--out", type=str, default="benchmark_steps_vs_training.png",
                        help="Output path for the plot (PNG).")
    parser.add_argument("--csv", type=str, default="benchmark_steps_vs_training.csv",
                        help="Output path for the raw per-model data (CSV).")
    parser.add_argument("--no-terrain", action="store_true",
                        help="Benchmark without per-cell terrain (grass/ice/dirt tiles) "
                             "or variable-width caves: bare 1-wide maze. Obs stays 42-D "
                             "(terrain dims read the constant grass mu, width dims read 1.0).")
    args = parser.parse_args()

    models = select_models(args)
    n_models = len(models)
    n_jobs = args.jobs if args.jobs is not None else min(os.cpu_count() or 1, n_models)
    n_jobs = max(1, min(n_jobs, n_models))

    print(f"Benchmarking {n_models} models on {args.n_mazes} mazes "
          f"(max {args.max_steps} RL steps/episode, {n_jobs} worker(s)).", flush=True)
    print("  models: " + ", ".join(str(s) for s, _ in models), flush=True)

    maze_seeds = [args.seed + i for i in range(args.n_mazes)]
    counter = mp.Value("i", 0)
    initargs = (counter, args.config, args.no_terrain, args.max_steps,
                maze_seeds, args.device)

    # Snapshot per-worker maze XML files so we can guarantee this run leaves none
    # behind, regardless of how the workers were torn down (see _sweep_env_xml).
    env_xml_before = _env_xml_files()

    results = []
    if n_jobs == 1:
        # Serial: initialize the worker state in this process and loop.
        _init_worker(*initargs)
        try:
            for mi, item in enumerate(models):
                row = _eval_model(item)
                results.append(row)
                _print_row(mi + 1, n_models, row, args.n_mazes)
        finally:
            if _ENV is not None:
                _ENV.close()
    else:
        # spawn is the default on Windows; force it everywhere for consistency.
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        initargs = (counter, args.config, args.no_terrain, args.max_steps,
                    maze_seeds, args.device)
        # NOTE: don't use `with Pool(...)` — its __exit__ calls terminate(), which
        # kills workers so their atexit env.close() never runs (leaking XML). We
        # close()+join() for a graceful shutdown, then sweep as a backstop.
        pool = ctx.Pool(processes=n_jobs, initializer=_init_worker, initargs=initargs)
        try:
            # Poll with a short timeout instead of `for row in imap_unordered(...)`.
            # On Windows a blocking (timeout=None) Pool wait is NOT interruptible by
            # Ctrl-C — the main thread never gets the KeyboardInterrupt delivered.
            # Waking up every second lets the interpreter raise it so we can stop.
            it = pool.imap_unordered(_eval_model, models)
            done = 0
            while done < n_models:
                try:
                    row = it.next(timeout=1.0)
                except mp.TimeoutError:
                    continue
                done += 1
                results.append(row)
                _print_row(done, n_models, row, args.n_mazes)
            pool.close()   # graceful: workers exit normally so their atexit close() fires
            pool.join()
        except KeyboardInterrupt:
            print("\n[interrupted] stopping workers...", flush=True)
            pool.terminate()
            pool.join()
            _sweep_env_xml(env_xml_before)  # killed workers skip their own cleanup
            raise SystemExit(130)
        except BaseException:
            pool.terminate()
            pool.join()
            raise

    # Backstop: remove any per-worker maze XML this run created but didn't clean up.
    _sweep_env_xml(env_xml_before)

    results.sort(key=lambda r: r[0])
    steps_arr = np.array([r[0] for r in results])
    mean_arr = np.array([r[1] for r in results])
    succ_arr = np.array([r[2] for r in results])

    _ensure_parent_dir(args.csv)
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("training_steps,mean_steps,n_success,n_mazes\n")
        for s, mean_s, ns, _obs in results:
            f.write(f"{s},{mean_s:.4f},{ns},{args.n_mazes}\n")
    print(f"Wrote data -> {args.csv}")

    fail_pct_arr = 100.0 * (args.n_mazes - succ_arr) / args.n_mazes
    _plot(steps_arr, mean_arr, fail_pct_arr, args)

    if (succ_arr < args.n_mazes).any():
        worst = int(succ_arr.min())
        print(f"[note] Some models failed to reach the goal on some mazes "
              f"(min success {worst}/{args.n_mazes}); those episodes are counted at "
              f"the {args.max_steps}-step cap. See n_success in the CSV.")


def _env_xml_files():
    """Set of per-worker maze XML files currently in scenes/ (env_index era)."""
    return set(globmod.glob(os.path.join("scenes", "scene_maze_env*.xml")))


def _sweep_env_xml(before):
    """Delete per-worker maze XML files that appeared since `before`.

    Backstop for the workers' own atexit env.close() cleanup: only files created
    during this run (not pre-existing stale ones from e.g. parallel training) are
    removed, so a concurrent run's files are left alone.
    """
    leaked = sorted(_env_xml_files() - before)
    for path in leaked:
        try:
            os.remove(path)
        except OSError:
            pass
    if leaked:
        print(f"[cleanup] removed {len(leaked)} per-worker maze XML file(s).")


def _print_row(idx, total, row, n_mazes):
    steps, mean_s, ns, obs_dim = row
    print(f"[done {idx}/{total}] {steps:>8d} train-steps (obs{obs_dim}): "
          f"mean={mean_s:6.1f} steps  success={ns}/{n_mazes}", flush=True)


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _plot(steps_arr, mean_arr, fail_pct_arr, args):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    # dataviz: two series on twin axes -> two distinct hues. Blue for the primary
    # steps-to-goal (left), a warm amber for the secondary failure rate (right).
    # Each y-axis label/ticks/spine is tinted to match its series so it's obvious
    # which axis belongs to which line. Light recessive grid, thin 2px lines.
    ACCENT = "#3b6fb0"       # steps-to-goal (left axis)
    FAIL = "#d98a2b"         # failure % (right axis) — warm amber
    INK = "#1f2933"
    MUTED = "#6b7280"

    _ensure_parent_dir(args.out)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    line_steps, = ax.plot(steps_arr, mean_arr, color=ACCENT, linewidth=2.0,
                          marker="o", markersize=3.5, label="mean steps-to-goal")

    ax.set_xlabel("Training steps", color=INK)
    ax.set_ylabel("RL steps to reach goal", color=ACCENT)
    ax.set_xscale("log")  # checkpoints span 1k..1M; log spreads them evenly
    ax.set_title(f"Snake steps-to-goal vs. training  "
                 f"({args.n_mazes} fixed mazes, cap {args.max_steps} steps)",
                 color=INK, fontsize=12)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=ACCENT)
    ax.tick_params(axis="x", colors=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["left"].set_color(ACCENT)

    # Right-side twin axis: percentage of mazes the model failed to solve (0..100).
    ax2 = ax.twinx()
    line_fail, = ax2.plot(steps_arr, fail_pct_arr, color=FAIL, linewidth=2.0,
                          marker="s", markersize=3.5, linestyle="--",
                          label="failed mazes (%)")
    ax2.set_ylabel("Failed mazes (%)", color=FAIL)
    ax2.set_ylim(-2, 102)
    ax2.tick_params(axis="y", colors=FAIL)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(FAIL)
    ax2.spines["left"].set_visible(False)

    ax.legend(handles=[line_steps, line_fail], frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote plot -> {args.out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
