"""Benchmark saved SAC checkpoints on a fixed battery of test mazes.

For a range of training checkpoints (selectable by first / interval / last step
count) this rolls each model out on the SAME set of N reproducible test mazes and
records how long the snake takes to reach the goal ("simulation time"). It then
plots the mean time-to-goal as a function of training steps, with a min/max band
(fill_between) over the N mazes.

Time is reported in SECONDS OF SIMULATED PHYSICS TIME, computed as
    time = rl_steps_to_finish * sim_steps_per_rl_step * model.opt.timestep
An episode that never reaches the goal within --max-steps is counted as a failure
and its time is capped at the episode limit (--max-steps). The per-model success
count is also recorded in the CSV so you can see how much of the average is real
finishing vs. capped failures.

Reproducibility: every model is evaluated on the identical maze battery. Maze i is
regenerated from a fixed seed (--seed + i) by re-seeding BOTH Python's global
`random` (which drives the maze layout in mazes/make_maze.create_maze_layout) and
the env's Gymnasium RNG (which drives goal-cell selection) right before each reset.

Examples
--------
    # every 2k-step model from 2k to 1M, 30 mazes (defaults)
    uv run python benchmark_models.py

    # coarser sweep: every 10k steps up to 500k, 20 mazes
    uv run python benchmark_models.py --first-steps 10000 --step-interval 10000 \
        --last-steps 500000 --n-mazes 20

    uv run python benchmark_models.py --device cpu --out plots/timing.png
"""
import argparse
import os
import random
import warnings

import numpy as np


# --- observation layout (mirrors env_snake.SnakeEnv._get_obs) --------------
# The env always emits the full 38-D observation, but checkpoints come from
# three eras with different widths (see CLAUDE.md / swarm_core.py). The full
# 38-D layout, in order, is:
#   [ joint_pos(12) | joint_vel(12) | head->target xy(2) | orient(2)
#     | head angular velocity(3) | heading error(2) | next-turn(1)
#     | last action(4) ]
# Older policies omit the trailing-but-not-last blocks:
#   37-D: drop the 1-D next-turn signal
#   34-D: drop next-turn AND the 3-D head angular velocity block
# We build the model-appropriate obs by slicing the full 38-D vector.
_FULL_OBS_DIM = 38
_ANGVEL_SLICE = slice(28, 31)   # head angular velocity (3)
_NEXT_TURN_IDX = 33             # next-turn signal (1)


def adapt_obs(full_obs, obs_dim):
    """Slice the env's full 38-D observation down to what `obs_dim` expects."""
    if obs_dim == _FULL_OBS_DIM:
        return full_obs
    if obs_dim == 37:  # drop next-turn signal only
        return np.delete(full_obs, _NEXT_TURN_IDX)
    if obs_dim == 34:  # drop next-turn AND head angular velocity
        drop = list(range(_ANGVEL_SLICE.start, _ANGVEL_SLICE.stop)) + [_NEXT_TURN_IDX]
        return np.delete(full_obs, drop)
    raise ValueError(
        f"Unsupported policy observation dim {obs_dim}; expected 34, 37 or 38. "
        "Update adapt_obs() if the obs layout changed."
    )


def discover_models(models_dir, first, interval, last):
    """Return [(steps, path), ...] for existing sac_snake_<steps>_steps.zip in range."""
    found = []
    missing = []
    for steps in range(first, last + 1, interval):
        path = os.path.join(models_dir, f"sac_snake_{steps}_steps.zip")
        if os.path.isfile(path):
            found.append((steps, path))
        else:
            missing.append(steps)
    return found, missing


def rollout_time(model, env, obs_dim, time_per_rl_step, max_steps):
    """Roll one episode (env already reset). Return (time_seconds, reached_goal)."""
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
    # Failures are capped at the episode limit so a never-finishing model reads
    # as "took the maximum time" rather than being dropped from the average.
    effective_steps = steps if reached else max_steps
    return effective_steps * time_per_rl_step, reached


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SAC checkpoints' time-to-goal over a fixed maze battery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--first-steps", type=int, default=2000,
                        help="Training-step count of the FIRST model to include.")
    parser.add_argument("--step-interval", type=int, default=2000,
                        help="Step spacing between successive models to include "
                             "(e.g. 10000 to use a model every 10k steps).")
    parser.add_argument("--last-steps", type=int, default=1000000,
                        help="Training-step count of the LAST model to include.")
    parser.add_argument("--n-mazes", type=int, default=30,
                        help="Number of reproducible test mazes each model is run on.")
    parser.add_argument("--max-steps", type=int, default=250,
                        help="Max RL steps per episode before truncation. Failures "
                             "are counted at this cap.")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Base seed; maze i uses seed+i (identical for every model).")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="YAML config the models were trained with.")
    parser.add_argument("--models-dir", type=str, default="models/checkpoints",
                        help="Directory holding sac_snake_<steps>_steps.zip files.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device for SAC.load ('cpu', 'cuda', 'auto').")
    parser.add_argument("--out", type=str, default="benchmark_time_vs_steps.png",
                        help="Output path for the plot (PNG).")
    parser.add_argument("--csv", type=str, default="benchmark_time_vs_steps.csv",
                        help="Output path for the raw per-model data (CSV).")
    args = parser.parse_args()

    # Import heavy deps after arg parsing so --help is instant.
    from stable_baselines3 import SAC
    from env_snake import SnakeEnv
    from config_utils import load_config

    models, missing = discover_models(
        args.models_dir, args.first_steps, args.step_interval, args.last_steps)
    if not models:
        raise SystemExit(
            f"No models found in {args.models_dir} for steps "
            f"{args.first_steps}:{args.last_steps}:{args.step_interval}.")
    if missing:
        print(f"[warn] {len(missing)} requested step counts had no checkpoint file "
              f"and were skipped (e.g. {missing[:5]}{'...' if len(missing) > 5 else ''}).")
    print(f"Benchmarking {len(models)} models on {args.n_mazes} mazes "
          f"(max {args.max_steps} RL steps/episode).")

    # One env, reused across every model and maze. Headless (no render).
    config = load_config(args.config)
    env = SnakeEnv(render_mode=None, config=config)
    env.max_episode_steps = args.max_steps

    maze_seeds = [args.seed + i for i in range(args.n_mazes)]

    def reset_to_maze(seed):
        """Reset the env to the reproducible maze identified by `seed`.

        Seeds BOTH RNGs the env draws from so the maze layout (global `random`)
        and the goal-cell choice (env np_random) are identical for every model.
        """
        random.seed(seed)
        env.reset(seed=seed)

    # Physics time per RL step (constant once the model is built in reset()).
    reset_to_maze(maze_seeds[0])
    time_per_rl_step = env.sim_steps_per_rl_step * env.model.opt.timestep

    rows = []  # (steps, mean, min, max, n_success)
    for mi, (steps, path) in enumerate(models):
        model = SAC.load(path, device=args.device)
        obs_dim = int(model.observation_space.shape[0])

        times = np.empty(args.n_mazes, dtype=np.float64)
        n_success = 0
        for i, seed in enumerate(maze_seeds):
            reset_to_maze(seed)
            t, reached = rollout_time(model, env, obs_dim, time_per_rl_step,
                                      args.max_steps)
            times[i] = t
            n_success += int(reached)

        mean_t, min_t, max_t = float(times.mean()), float(times.min()), float(times.max())
        rows.append((steps, mean_t, min_t, max_t, n_success))
        print(f"[{mi + 1}/{len(models)}] {steps:>8d} steps (obs{obs_dim}): "
              f"mean={mean_t:6.2f}s  min={min_t:6.2f}s  max={max_t:6.2f}s  "
              f"success={n_success}/{args.n_mazes}")

    env.close()

    rows.sort(key=lambda r: r[0])
    steps_arr = np.array([r[0] for r in rows])
    mean_arr = np.array([r[1] for r in rows])
    min_arr = np.array([r[2] for r in rows])
    max_arr = np.array([r[3] for r in rows])
    succ_arr = np.array([r[4] for r in rows])

    # --- save raw data --------------------------------------------------
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("training_steps,mean_time_s,min_time_s,max_time_s,"
                f"n_success,n_mazes\n")
        for s, mean_t, min_t, max_t, ns in rows:
            f.write(f"{s},{mean_t:.6f},{min_t:.6f},{max_t:.6f},{ns},{args.n_mazes}\n")
    print(f"Wrote data -> {args.csv}")

    # --- plot -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    # dataviz: single series -> one hue (a calm blue), light→ recessive grid,
    # thin 2px line, a muted same-hue fill for the min/max band, direct title.
    ACCENT = "#3b6fb0"       # line
    BAND = "#8fb4dd"         # min/max fill (same hue, lighter)
    INK = "#1f2933"
    MUTED = "#6b7280"

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=130)
    ax.fill_between(steps_arr, min_arr, max_arr, color=BAND, alpha=0.35,
                    linewidth=0, label="min–max over mazes")
    ax.plot(steps_arr, mean_arr, color=ACCENT, linewidth=2.0, marker="o",
            markersize=3.5, label="mean time-to-goal")

    ax.set_xlabel("Training steps", color=INK)
    ax.set_ylabel("Simulation time to goal (s)", color=INK)
    ax.set_title(f"Snake time-to-goal vs. training  "
                 f"({args.n_mazes} fixed mazes, cap {args.max_steps} steps)",
                 color=INK, fontsize=12)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote plot -> {args.out}")

    if (succ_arr < args.n_mazes).any():
        worst = int(succ_arr.min())
        print(f"[note] Some models failed to reach the goal on some mazes "
              f"(min success {worst}/{args.n_mazes}); those episodes are counted "
              f"at the {args.max_steps}-step cap. See n_success in the CSV.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
