"""Checkpoint-progression swarm visualization.

Spawn one snake per training checkpoint (e.g. 1k / 10k / 100k / 1M steps), all in
the SAME shared maze, each a distinct color with a floating label, running
deterministically. Shows how the policy improves over training: later checkpoints
should track the A* path to the goal noticeably better than early ones.

Usage (explicit checkpoints, in the order you want them colored/labeled):
    uv run python viz_checkpoints.py \
        models/checkpoints/sac_snake_1000_steps.zip \
        models/checkpoints/sac_snake_10000_steps.zip \
        models/checkpoints/sac_snake_100000_steps.zip \
        models/checkpoints/sac_snake_1000000_steps.zip

Or pick step counts from a glob of the checkpoints dir:
    uv run python viz_checkpoints.py \
        --glob "models/checkpoints/sac_snake_*_steps.zip" --pick 1000,10000,100000,1000000

See swarm_core.py for the engine and shared CLI flags.
"""

import os
import re
import glob as globmod
import argparse
from stable_baselines3 import SAC

from config_utils import load_config
from swarm_core import SnakeSwarm, add_common_args, distinct_colors, run_from_args

_STEPS_RE = re.compile(r"sac_snake_(\d+)_steps")


def _label_for(path):
    """Short label from a checkpoint filename: step count as 1k/10k/1M, else stem."""
    m = _STEPS_RE.search(os.path.basename(path))
    if not m:
        return os.path.splitext(os.path.basename(path))[0]
    n = int(m.group(1))
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def _resolve_paths(args):
    if args.glob:
        found = {}
        for p in globmod.glob(args.glob):
            m = _STEPS_RE.search(os.path.basename(p))
            if m:
                found[int(m.group(1))] = p
        if args.pick:
            wanted = [int(x) for x in args.pick.split(",")]
            missing = [w for w in wanted if w not in found]
            if missing:
                raise SystemExit(f"[viz_checkpoints] no checkpoint for steps {missing} "
                                 f"in glob {args.glob!r}")
            return [found[w] for w in wanted]
        return [found[k] for k in sorted(found)]
    # Drop blank/stray args (e.g. a lone "\" left over from bash-style line
    # continuation pasted into PowerShell, where the continuation char is a
    # backtick `, not a backslash).
    paths = [p.strip().strip("\\") for p in args.checkpoints]
    paths = [p for p in paths if p]
    if not paths:
        raise SystemExit("[viz_checkpoints] give checkpoint paths, or --glob/--pick.\n"
                         "  NOTE: in PowerShell, continue a line with a backtick `, "
                         "not a backslash \\ (or just put it all on one line).")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit("[viz_checkpoints] checkpoint file(s) not found:\n  "
                         + "\n  ".join(missing))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*",
                        help="Checkpoint .zip paths, one snake each (ordered).")
    parser.add_argument("--glob", type=str, default=None,
                        help="Glob of checkpoints to choose from (with --pick).")
    parser.add_argument("--pick", type=str, default=None,
                        help="Comma-separated step counts to pick from --glob, "
                             "e.g. 1000,10000,100000.")
    add_common_args(parser)
    args = parser.parse_args()

    paths = _resolve_paths(args)
    config = load_config(args.config)
    colors = distinct_colors(len(paths))

    specs = []
    for i, p in enumerate(paths):
        specs.append({
            "policy": SAC.load(p),          # one model per checkpoint
            "deterministic": True,
            "label": _label_for(p),
            "color": colors[i],
        })
        print(f"[viz_checkpoints] snake {i}: {_label_for(p):>6s}  <- {p}")

    swarm = SnakeSwarm(
        specs, config, seed=args.seed,
        render_slowdown=args.render_slowdown, show_labels=True,
        jitter=args.jitter,
    )
    run_from_args(swarm, args)


if __name__ == "__main__":
    main()
