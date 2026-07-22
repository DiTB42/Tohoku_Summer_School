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
uv run python train_sac.py --n-envs 8                    # CPU-parallel training (SubprocVecEnv, 8 workers)
uv run python train_sac.py --no-terrain                  # bare floor, no grass/ice/dirt (also on test_model/viz_*/benchmark)
uv run python test_model.py                             # roll out models/sac_snake_final.zip in the viewer
uv run python test_model.py models/checkpoints/sac_snake_20000_steps.zip --episodes 3 --render-slowdown 4
uv run python cpg/snake_cpg.py                          # CPG-only sanity check (no RL) — snake should crawl forward
uv run tensorboard --logdir sac_snake_tensorboard/      # monitor training (localhost:6006)

# Multi-snake swarm visualization (presentation tool; live viewer, distinct color per snake)
uv run python viz_stochastic.py models/checkpoints/sac_snake_1000000_steps.zip --n 20 --seed 1
uv run python viz_checkpoints.py CKPT1.zip CKPT2.zip CKPT3.zip --seed 1   # one snake per checkpoint
uv run python viz_checkpoints.py --glob "models/checkpoints/sac_snake_*_steps.zip" --pick 1000,10000,100000,1000000
#   add --no-render for a headless sanity check; PowerShell line-continuation is ` (backtick), NOT \
# Record an MP4 instead of the live viewer (offscreen; for slides, no screen-recording needed):
uv run python viz_stochastic.py CKPT.zip --n 20 --record out.mp4 --video-seconds 20 --video-fps 30 --video-res 1280x720
```

There is no test suite, linter, or build step — "testing" means running the CPG sanity check,
a short training run, and a viewer rollout, then inspecting TensorBoard.

**Windows-only:** `train_sac.py` calls `ctypes.windll.kernel32.SetThreadExecutionState` (keeps the
machine awake during long training). This call runs inside the `if __name__ == "__main__"` guard so
it does not re-fire in spawned worker processes; it will fail on non-Windows platforms, so guard or
remove it if running elsewhere.

**Parallel training (`--n-envs N` / `training.n_envs`):** the bottleneck is CPU physics (250
`mj_step` per RL step), not the tiny `[64,64]` MLP, and the installed torch is CPU-only — so speedup
comes from `SubprocVecEnv`, not CUDA. `n_envs=1` uses `DummyVecEnv` (original single-env behavior);
`n_envs>1` runs one MuJoCo sim per subprocess. Key wiring in `train_sac.py`: the env is always
`VecMonitor`-wrapped (needed for `rollout/ep_rew_mean`), `gradient_steps` is multiplied by `n_envs`
to keep one update per collected transition, and `CheckpointCallback.save_freq` is `2000 // n_envs`
(SB3 counts vec-steps). **Gotcha:** each `SnakeEnv` must write a *unique* maze XML — `SnakeEnv(...,
env_index=i)` produces `scenes/scene_maze_env{i}_{pid}.xml` (cleaned up in `close()`); the shared
default `scenes/scene_maze_trial.xml` (env_index=None) is what single-env / `test_model.py` use. N
workers sharing one file would clobber each other mid-write → NaN/corrupt-load crashes. Parallel
training is headless (`--render` is ignored when `n_envs>1`).

## Architecture: hierarchical control loop

The system is a four-layer stack; a change at one layer usually ripples to the observation/reward:

```
A* (global planner)  →  SAC (local planner)  →  CPG (gait generator)  →  MuJoCo (physics)
   waypoints              CPG params (R,ω,θ,δ)     12 joint targets         locomotion
```

- **`env_snake.py` — `SnakeEnv(gym.Env)`** is the integration hub. One RL `step()` calls the SAC
  action once, then advances the CPG + physics `sim_steps_per_rl_step = int(cpg_frequency / rl_frequency)`
  times (currently 250). Observation is **42-D**: actuated joint pos(12) + actuated joint vel(12) +
  head→target vec **xy**(2) + head orientation (az, angle)(2) + head angular velocity(3) +
  heading error (cos, sin)(2) + next-turn signal(1) + terrain friction under head & tail(2) +
  cave width current & next-path cell(2) + last action(4). The terrain block is the raw tangential μ
  of the maze cell under the head (`frame_0-1`) and tail (`frame12-2`) bodies
  (`_get_terrain_friction` / `_world_to_maze`), falling back to grass over wall cells. The cave-width
  block (`_get_cave_widths`) is the open corridor span (world units, 1.0 normal .. ~0.12 tightest pinch)
  of the cell under the head and of the cell one step ahead in the A* path — see the variable-width
  caves section below. The next-turn signal
  (`_get_next_turn_signal`) is the turn the A* path makes AT the next waypoint — signed 2D cross of
  incoming (B−A) vs outgoing (B→C) directions: +1 left, −1 right, 0 straight (same +=left convention
  as heading error). Waypoints are dense (one per cell) so it is 0 along corridors and flips to ±1
  ~one cell before a corner. Reward is the paper's 3-term Eq. 12:
  `w_progress·r1 + w_velocity·r2 − w_smoothness·r3` (proximity, closing speed, action-change penalty),
  plus three deliberate divergences (all in config): `w_goal` one-shot terminal bonus, `time_penalty`
  per-step living cost, and `w_waypoint` one-shot bonus per newly reached waypoint (breadcrumb trail
  for terrain-hard maps; path is always 30 waypoints so total = 30·w_waypoint — keep below w_goal).
  **Gotcha:** the snake model interleaves each actuated joint (`Actuator1..12`) with a passive
  free-spinning wheel joint, so joint state is looked up **by name** (`_resolve_actuated_joints`) —
  never via `qpos[-12:]`, which mixes in wheels (unbounded angle) and drops the first actuators.
  The same trap applied to `jnt_range[-12:]` when clipping CPG targets (it zeroed half the actuators);
  both are now fixed. `last_action` in the obs makes the r3 smoothness penalty Markovian; heading error
  pre-computes head-forward-vs-target so the policy needn't infer it from world-frame vectors.
- **`cpg/snake_cpg.py` — `PaperCPG`** integrates the paper's Eq. 9 phase/amplitude dynamics with Euler
  steps and emits `x = r·sin(φ) + δ` per joint. **Critical invariant:** the CPG's `dt` must equal
  `model.opt.timestep`, because `cpg.update()` is called once per `mj_step`. The env recreates the CPG
  every `reset()` with `dt=self.model.opt.timestep` so phase/amplitude state never carries across episodes.
- **`mazes/make_maze.py`** — random maze generation (recursive backtracker), valid-spawn enumeration,
  A* pathfinding, per-cell terrain generation (`generate_terrain`), and variable-width cave generation
  (`generate_widths`). Comments are in Italian.
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
- **Per-cell terrain (grass/ice/dirt):** every open cell gets a terrain type from
  `generate_terrain()` (in `mazes/make_maze.py`) — a BFS flood-fill from the always-grass start
  cell where each new cell keeps its parent's terrain with per-type probability
  (`TERRAIN_KEEP_PROB`: grass 0.8, ice/dirt 0.65) and otherwise switches uniformly to one of the
  other two (spatially correlated patches; ice/dirt runs are shorter than grass). Drawn from the
  env's `self.np_random`, so it is seed-reproducible. **`--no-terrain`** (supported by train_sac,
  test_model, viz_stochastic, viz_checkpoints, benchmark_models; config key
  `env.terrain_enabled`) disables tiles + terrain physics AND variable-width caves (below) for
  presentations — the obs STAYS 42-D
  (terrain dims read the constant grass μ=3.0, width dims read the constant 1.0), so checkpoints work with or without it. `make_maze_on_mujoco(terrain=...)` injects one thin `terrain_{type}_{row}_{col}`
  box geom per open cell (grass tiles are green, μ = the baseline 3.0, mirroring the kb branch's
  corridor floor; ice bluish, dirt brown); tile tops sit 0.2 mm above the floor plane so
  contacts land on the tile. **`priority="1"` on the tiles is load-bearing:** snake geoms inherit
  μ=3 from snake.xml's global geom default and equal-priority contacts take the elementwise max,
  so without it ice (μ=1) would be silently ignored. Friction/color constants (`TERRAIN_FRICTION`:
  grass 3.0 / ice 0.5 / dirt 7.0, `TERRAIN_TILES`) live in `mazes/mujoco_tools.py` and are the
  single source of truth for both the physics and the obs values.
- **Variable-width caves:** `generate_widths()` (in `mazes/make_maze.py`, gated by the SAME
  `env.terrain_enabled` / `--no-terrain` flag) probabilistically **narrows** (pinches) straight
  segments of the A* solution path into "caves" — runs of 1–7 cells (single-cell pinches included),
  each with its OWN random peak tightness and a symmetric taper centered on the run (so the entrance is
  in the middle), like the kb branch's funnel. It returns
  `{(col,row): extension e}` (the open span becomes `1.0 - 2e` world units, peak `e` drawn in
  `[0.2, 0.44]` per cave, so spans range ~0.6 mild to ~0.12 tight), drawn from `self.np_random` right
  after `generate_terrain` (both are the only rng
  consumers after the goal draw, so seeds stay reproducible with caves on or off). Only cells whose two
  perpendicular neighbors are walls qualify (a plain straight corridor cell).
  `make_maze_on_mujoco(widths=...)` realizes a cave by **growing the two flanking wall boxes inward
  symmetrically** (`_cave_recessions`): each flank wall's inner face grows `e` into the corridor while
  its outer face stays fixed, so the pinch is symmetric about the corridor centerline (which stays on
  the integer grid, `_world_to_maze` still correct). Terrain floor tiles stay full-cell (the grown
  walls sit on top of the tile edges). The env stores `self.width_map` + `self.path_maze`; the obs
  reads the current cell (under the head) and the next path cell via `_get_cave_widths`.
- **Decorative dressing (cosmetic only, gated by the SAME `--no-terrain` flag):**
  `make_maze_on_mujoco(decorations=...)` injects purely visual geometry when `decorations=True`
  (callers pass `env.terrain_enabled`, so `--no-terrain` gives the bare clean maze). Helpers live
  in `mazes/mujoco_tools.py`, ported from the kb branch: `_add_mountain_wall_layers` /
  `_mountain_layers_for_cell` (asymmetric brown mountain stacks that grow OUT of a wall's exposed
  corridor face — 1-5 tapering tiers + optional peak — on only ~1/4-1/3 of walls, gated by
  `_should_add_mountain_wall`/`_exposed_wall_faces`), and `_add_base_stones`/`_base_stones_for_cell`
  (2-3 grey caps on ~half the walls). Cherry blossoms are now also injected on a subset of those
  decorated walls (`_add_cherry_blossom_tree`): a brown trunk + pink canopy made from many clustered
  small rectangular leaf blocks, plus tiny rectangular petals over nearby corridor-facing edges.
  Placement now mixes single trees and
  clusters, with short/medium/tall height variants and narrow/wide canopy variants; some trees
  intentionally overhang slightly into the route for visual drama. Tree/petal geoms are children of
  the `box_*` bodies and include
  decorative joints named `cherry_sway_*` (canopy sway) and `cherry_petal_drop_*` (looping falling
  petals). A decorative waterfall is also spawned from the tallest generated mountain peak as a
  straight vertical drop to the floor, built entirely from rectangular blue/white box strips and
  animated through
  `waterfall_flow_*` slide joints. All decoration joints are animated at runtime in both `SnakeEnv`
  and `SnakeSwarm` by directly writing their `qpos` each physics substep (no extra actuators; snake
  control remains unchanged). Plus
  `_add_goal_beacon` (a small translucent `beacon` pillar+orb over the goal) and `_recolor_scene`
  (greener groundplane + warmer skybox, done in-place on the parsed `<asset>` at gen time — NOT
  baked into `scene.xml`, so the recolor is properly gated). **All decoration geoms are
  non-colliding (`contype=0 conaffinity=0`) with no friction/priority, so physics and the 42-D obs
  are untouched → every checkpoint still loads.** Randomness comes from per-cell generators seeded
  by `_decor_salt(maze)` (a layout-derived salt XORed into kb's fixed per-cell seeds), never
  `self.np_random`, so a seeded maze/goal stays byte-identical with decorations on or off
  (reproducibility invariant preserved). Name prefixes (`mountain_`, `stone_`, `beacon`) are
  stripped in the idempotent cleanup at the top of `make_maze_on_mujoco`. The swarm viz tools get
  all of this for free (they call the same generator); `update_maze` (not used by `reset()`) has no
  decoration handling.
- The snake **head body is named `frame_0-1`** — used throughout `env_snake.py` for head position,
  orientation, angular velocity, and the tracking camera. The **tail body is `frame12-2`** (used
  for the tail terrain-friction obs).

## Multi-snake swarm visualization (`swarm_core.py` + `viz_*.py`)

A **standalone presentation tool**, independent of the training pipeline (does NOT touch
`env_snake.py` / `train_sac.py` / `test_model.py`). It runs many snakes of trained policies in
**one shared maze, one physics loop, one live viewer**, each a distinct color.

- **`swarm_core.py`** — the engine. `SnakeSwarm` assembles ONE `MjModel` containing N snakes by
  ElementTree surgery (same idiom as `make_maze_on_mujoco`): every snake gets an `s{k:02d}_` name
  prefix (snake.xml's names — `frame_0-1`, `Actuator1..12`, camera, light — are hard-coded and would
  otherwise collide), and the copies are merged into a maze scene built from **`scenes/scene_nosnake.xml`**.
  `SnakeState` holds per-snake handles (by prefixed name), a `PaperCPG`, and rollout state.
- **`viz_stochastic.py`** — N snakes from ONE checkpoint, `deterministic=False` (SAC samples), so they
  fan out and show the policy's behavior spread. Shared policy → obs are batched in one `predict()`.
- **`viz_checkpoints.py`** — one snake per checkpoint (labels like `1k`/`10k`/`1M`), `deterministic=True`,
  showing training progression. `--glob`/`--pick` picks step counts; paths are validated up front.

Key gotchas / invariants (mirror or diverge from `SnakeEnv` deliberately):
- **`scenes/scene_nosnake.xml`** is `scene.xml` WITHOUT the `<include file="snake.xml"/>` (swarm injects
  its own snakes). It duplicates the `compiler angle="radian"`, `option integrator="implicitfast"`, and
  `<default><geom friction=...>` that snake.xml normally contributes via the include — **without
  `angle="radian"` the CPG's radian targets are read as degrees and the gait breaks.**
- **Snake↔snake collisions are disabled** by setting every snake geom `contype="2" conaffinity="1"`
  (env/walls stay default `1/1`): bitmask `snake-snake (2&1)|(2&1)=0`, `snake-floor (2&1)|(1&1)=1`.
  Side effect: each snake's own non-adjacent self-collisions are also off (minor for a thin chain).
  This lets all N snakes share the start cell (world origin) without interpenetrating.
- **Observation width auto-adapts per policy.** `SnakeState` reads `policy.observation_space` and builds
  the **42-D** obs (adds cave width current/next-path cell — from `generate_widths` on the shared maze,
  seeded by `--seed`), the **40-D** obs (omits the width block, keeps terrain), the **38-D** obs (also
  omits the terrain block), the **37-D** obs (also omits
  the 1-D next-turn signal), or the oldest **34-D** obs (also omits the 3-D head angular-velocity
  block). Saved checkpoints come from all eras — e.g. `sac_snake_1000_steps.zip`/`best_model.zip` are
  34-D, `sac_snake_final.zip` is 37-D, next-turn-era checkpoints are 38-D, terrain-era ones are 40-D,
  cave-era ones are 42-D —
  and can be mixed in one `viz_checkpoints` run. Action is always 4-D `(R,ω,θ,δ)`; scalar-θ (1-D) policies are rejected. Actions are clipped to
  each policy's own `action_space` bounds (they differ per checkpoint).
- The obs math in `SnakeState.get_obs` is a **hand-copy of `env_snake._get_obs`** — keep them in sync if
  the obs layout changes.
- Generated XML (combined scene + intermediate maze) is written to a **scratch dir**, not `scenes/`, so
  it does not pollute git. Camera is a fixed top-down free camera framing the whole maze (not the env's
  per-head tracking). `--no-render` runs headless (no viewer, no real-time sleep) for smoke tests.
- **MP4 recording (`--record out.mp4`):** renders offscreen via `mujoco.Renderer` (no live viewer),
  compressing the episode's sim time into `--video-seconds` at `--video-fps` by capturing every Nth
  substep. Encoded with `imageio[ffmpeg]` (added to `requirements.txt`). `scenes/scene_nosnake.xml` sets
  `<global offwidth/offheight>` to size the offscreen framebuffer — `--video-res` is clamped to it and
  rounded to a multiple of 16 for H.264. The goal/label markers are drawn into `renderer.scene`
  (`_draw_markers(..., append=True)`) rather than the viewer's `user_scn`. `_apply_camera` is shared by
  the live viewer and the recorder so both frame the maze identically.

## Key conventions

- Maze→world coordinates: `x = (col − start_col)`, `y = −(row − start_row)`, `z = 0.15`
  (cell size 0.5, so a step of one cell = 1.0 world units). Note the **y-axis is flipped**.
- **Maze dims must be odd** — `create_maze_layout` normalizes even values UP to the next odd
  (10→11): with even dims the carver could open the bottom row / right column, leaving the maze
  with no outer wall on two sides (old bug, fixed). `config/default.yaml` is now 11×11, which has
  the same junction grid the buggy 10×10 effectively had.
- **Fixed goal distance:** every `SnakeEnv.reset()` produces a goal that is *exactly*
  `GOAL_PATH_LENGTH` (=30, hardcoded in `env_snake.py`) A* moves from the fixed start
  `(1,1)`, so episode difficulty is constant. `reset()` regenerates the maze (up to
  `MAX_MAZE_ATTEMPTS`=1000) until a cell at that exact BFS distance exists (~99% of 10x10
  mazes qualify on the first try). Candidate distances come from `cell_distances()`
  (single BFS) in `mazes/make_maze.py`. The old `min_goal_distance` config key is now
  **unused** (superseded). `swarm_core.py` (viz tool) still uses the old random goal and
  was intentionally left unchanged.
- Config/reward tuning is the primary knob for behavior — prefer editing `config/*.yaml` over
  hardcoding. Create a new YAML and pass `--config` rather than mutating `default.yaml` for experiments.
- Known limitation: complex full mazes produce NaN errors; the config is tuned for corridor-like layouts.
- **Checkpoint compatibility:** the env's obs is now 42-D (cave-width block added on top of the
  terrain block), so pre-cave checkpoints (40-D and older, incl. the shipped
  `models/sac_snake_final.zip`) fail SB3's obs-space check in `test_model.py` and need retraining. The
  swarm viz tools still load them (obs auto-adapt). `benchmark_models.py`'s `adapt_obs` slices the full
  42-D obs down for older policies.
