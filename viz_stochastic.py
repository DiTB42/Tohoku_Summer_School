"""Stochastic-spread swarm visualization.

Spawn N snakes that ALL use the SAME trained checkpoint but sample their actions
stochastically (SAC's Gaussian policy, deterministic=False), so they diverge and
reveal the behavior distribution of one policy in a single shared maze/view.

Usage:
    uv run python viz_stochastic.py models/checkpoints/sac_snake_1000_steps.zip --n 20
    uv run python viz_stochastic.py models/sac_snake_final.zip --n 12 --seed 0 \
        --render-slowdown 2 --max-steps 250

See swarm_core.py for the engine and the shared CLI flags (--config, --max-steps,
--render-slowdown, --seed, --jitter).
"""

import argparse
from stable_baselines3 import SAC

from config_utils import load_config
from swarm_core import SnakeSwarm, add_common_args, distinct_colors, run_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=str,
                        help="Path to the trained SAC .zip to sample from.")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of stochastic snake instances (default 20).")
    add_common_args(parser)
    # Stochastic spread: labels would just clutter 20 identical-policy snakes.
    parser.add_argument("--labels", action="store_true",
                        help="Show a floating index tag above each snake head.")
    args = parser.parse_args()

    config = load_config(args.config)
    model = SAC.load(args.model_path)  # one shared policy; predict() is batched

    colors = distinct_colors(args.n)
    specs = [
        {"policy": model, "deterministic": False,
         "label": f"#{i}", "color": colors[i]}
        for i in range(args.n)
    ]

    print(f"[viz_stochastic] {args.n} snakes from {args.model_path} "
          f"(stochastic sampling)")
    swarm = SnakeSwarm(
        specs, config, seed=args.seed,
        render_slowdown=args.render_slowdown, show_labels=args.labels,
        jitter=args.jitter,
    )
    run_from_args(swarm, args)


if __name__ == "__main__":
    main()
