# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Make sure to keep this file updated when doing or seeing code changes that could be important observations.

## What this is

A **teaching repository** (Tohoku University TESP summer school) that re-implements the paper
*Hierarchical RL-Guided Large-scale Navigation of a Snake Robot* (arXiv:2312.03223) in MuJoCo.
Some parts are **deliberately left inconsistent with the paper** as student exercises — see
`docs/STUDENT_TASKS_EN.md` / `docs/STUDENT_TASKS_JP.md` and the `# TODO(student task ...)` /
`# 学生課題` comments in the code. When editing, be aware you may be touching an intentional gap;
`docs/COBRA_implementation_diff_EN.md` lists every known divergence from the paper.

Note: the README describes the "shipped" reduced version (scalar-θ action space), but the working
tree has already been extended toward student task 1 — the action space is 4-D `(R, ω, θ, δ)`.
Read the actual code, not just the README, before assuming which state you're in.

## Commands

The project uses [uv](https://docs.astral.sh/uv/) with Python 3.11.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt

uv run python train_sac.py                              # train SAC (config/default.yaml, device auto)
uv run python train_sac.py --config config/best_training.yaml --device cpu --render
uv run python test_model.py                             # roll out models/sac_snake_final.zip in the viewer
uv run python test_model.py models/checkpoints/sac_snake_20000_steps.zip --episodes 3 --render-slowdown 4
uv run python cpg/snake_cpg.py                          # CPG-only sanity check (no RL) — snake should crawl forward
uv run tensorboard --logdir sac_snake_tensorboard/      # monitor training (localhost:6006)
```

There is no test suite, linter, or build step — "testing" means running the CPG sanity check,
a short training run, and a viewer rollout, then inspecting TensorBoard.

**Windows-only:** `train_sac.py` calls `ctypes.windll.kernel32.SetThreadExecutionState` (keeps the
machine awake during long training). This import-time call will fail on non-Windows platforms; guard
or remove it if running elsewhere.

## Architecture: hierarchical control loop

The system is a four-layer stack; a change at one layer usually ripples to the observation/reward:

```
A* (global planner)  →  SAC (local planner)  →  CPG (gait generator)  →  MuJoCo (physics)
   waypoints              CPG params (R,ω,θ,δ)     12 joint targets         locomotion
```

- **`env_snake.py` — `SnakeEnv(gym.Env)`** is the integration hub. One RL `step()` calls the SAC
  action once, then advances the CPG + physics `sim_steps_per_rl_step = int(cpg_frequency / rl_frequency)`
  times (currently 250). Observation is **34-D**: joint pos(12) + joint vel(12) + head→target vec(3) +
  head orientation axis-angle(4) + head angular velocity(3). Reward is the paper's 3-term Eq. 12:
  `w_progress·r1 + w_velocity·r2 − w_smoothness·r3` (proximity, closing speed, action-change penalty).
- **`cpg/snake_cpg.py` — `PaperCPG`** integrates the paper's Eq. 9 phase/amplitude dynamics with Euler
  steps and emits `x = r·sin(φ) + δ` per joint. **Critical invariant:** the CPG's `dt` must equal
  `model.opt.timestep`, because `cpg.update()` is called once per `mj_step`. The env recreates the CPG
  every `reset()` with `dt=self.model.opt.timestep` so phase/amplitude state never carries across episodes.
- **`mazes/make_maze.py`** — random maze generation (recursive backtracker), valid-spawn enumeration,
  and A* pathfinding. Comments are in Italian.
- **`config_utils.py`** — loads a YAML into nested dataclasses (`Config` → reward/cpg/training/env).
  Missing keys fall back to dataclass defaults, so `load_config(None)` reproduces baseline behavior.
  **`config/default.yaml` values override the dataclass defaults** and can differ from them — trust the YAML.

## MuJoCo scene generation (important gotcha)

The maze XML is **generated at runtime, not authored by hand**:

- `scenes/scene.xml` is the base scene; it `<include>`s `scenes/snake.xml` (the 12-joint snake body).
- On every `reset()`, `SnakeEnv` calls `make_maze_on_mujoco(...)` (in `mazes/mujoco_tools.py`), which
  parses `scene.xml`, injects wall/goal/waypoint bodies for the current maze, and writes
  **`scenes/scene_maze_trial.xml`** — then loads *that* file. So `scene_maze_trial.xml` is a
  regenerated artifact (it shows as modified in git after any run); don't hand-edit it expecting the
  change to survive. Edit `scene.xml` / `snake.xml` or the generator instead.
- The snake **head body is named `frame_0-1`** — used throughout `env_snake.py` for head position,
  orientation, angular velocity, and the tracking camera.

## Key conventions

- Maze→world coordinates: `x = (col − start_col)`, `y = −(row − start_row)`, `z = 0.15`
  (cell size 0.5, so a step of one cell = 1.0 world units). Note the **y-axis is flipped**.
- Config/reward tuning is the primary knob for behavior — prefer editing `config/*.yaml` over
  hardcoding. Create a new YAML and pass `--config` rather than mutating `default.yaml` for experiments.
- Known limitation: complex full mazes produce NaN errors; the config is tuned for corridor-like layouts.
