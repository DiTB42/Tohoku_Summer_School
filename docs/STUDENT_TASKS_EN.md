# Coursework Tasks: Bring the Implementation Closer to the Paper

This repository is a re-implementation of the paper
[*Hierarchical RL-Guided Large-scale Navigation of a Snake Robot*](https://arxiv.org/abs/2312.03223v1)
(the coursework write-up is [`COBRA_paper_EN.md`](COBRA_paper_EN.md)).
However, **some parts have deliberately been left inconsistent with the paper**. Your task is to
implement those gaps yourself and bring the implementation closer to the paper's design.

- An overview of the full set of differences between the repo and the paper is summarized in [`COBRA_implementation_diff_EN.md`](COBRA_implementation_diff_EN.md).
- The relevant spots in the code have `# TODO(student task 1 …)` / `# TODO(student task 2 …)` comments.
- **The code works as-is in its current state** (a reduced version using only a scalar θ). Get it running first, then extend it.

The expected total time is **about 10 hours**. It's recommended to tackle the tasks in order.

---

## Prerequisites: Environment Setup and Sanity Check

Set up the uv environment following the README's instructions.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

First, confirm that the current (reduced) version runs.

```bash
# Does the snake move forward with the CPG alone (no RL)? (the __main__ in cpg/snake_cpg.py)
uv run python cpg/snake_cpg.py

# Run training for a short time (you can also edit the config to shrink total_timesteps)
uv run python train_sac.py
```

Visualize training:

```bash
uv run tensorboard --logdir sac_snake_tensorboard/
```

---

## Task 1: Extend the Action Space to R⁴ (paper §3 / diff item 6)

### Goal

Currently, the action RL outputs is **only a single scalar phase difference θ**, while the amplitude R, frequency ω,
and offset δ are fixed values. This task extends it to the paper's 4-dimensional action space **(R, ω, θ, δ)**.

**δ (the offset)** is especially important. Shifting the sinusoid's center with δ bends the whole body into an arc,
letting the snake **turn** ([`COBRA_paper_EN.md`](COBRA_paper_EN.md) §2.1). Right now δ=0 is fixed, so the snake
can effectively only move in a straight line.

### Relevant files/locations

| Location | What to do |
|---|---|
| `action_space` in [`env_snake.py`](../env_snake.py) `__init__` | Change `Box` from 1D to 4D. Set each component's low/high to the parameter's range |
| `theta = action` in [`env_snake.py`](../env_snake.py) `step()` | Decompose `action` into `R, omega, theta, delta` and pass them to `cpg.set_parameters(...)` |
| `action_low/high` under `env` in [`config/default.yaml`](../config/default.yaml) | Redesign so it can hold a range per parameter (adding new keys is fine) |

`cpg.set_parameters(R, omega, theta, delta)` is already designed to accept 4 arguments, so
the CPG core ([`cpg/snake_cpg.py`](../cpg/snake_cpg.py)) should generally require no changes.

### Hints

- Rough action ranges (paper values): `R∈[0, 1.5]`, `ω∈[-0.1, 0.1]`, `θ∈[-π, π]`, `δ∈[-0.1, 0.1]`.
- **Stable-Baselines3's SAC automatically scales the tanh output to the `Box`'s low/high.**
  You don't need to write your own explicit scaling logic (just set the `action_space` range correctly).
- Rather than jumping straight to checking with RL, it's easier to debug if you first remove RL,
  feed parameters in by hand, and check **whether varying δ produces turning** in the
  `__main__` of [`cpg/snake_cpg.py`](../cpg/snake_cpg.py) first
  (milestone 2 in [`COBRA_paper_EN.md`](COBRA_paper_EN.md) §6).
- Once the action becomes 4-dimensional, the reward term `r3` (the difference from `last_action`) automatically
  becomes the norm of a 4-dimensional vector.

### How to verify (acceptance criteria)

- Manually setting δ to a positive value causes the head to trace an arc and the heading to change.
- `uv run python train_sac.py` runs training without errors on the 4-dimensional action space.

---

## Task 2: Design the Reward Weights (paper §4.2 / diff item 9)

### Goal

The reward function has been reduced to **just the 3 terms** of the paper's Eq. (12)
(`step()` in [`env_snake.py`](../env_snake.py)):

```
reward = w_progress * r1  +  w_velocity * r2  -  w_smoothness * r3
```

- `r1 = 1/(0.1 + d_t)` … larger the closer to the goal (proximity reward)
- `r2 = d_{t-1} - d_t`  … speed of approaching the goal (approach reward)
- `r3 = ‖a_t - a_{t-1}‖` … penalty for abrupt action changes (**subtracted**)

The paper **does not specify the weighting coefficients** for these 3 terms. Designing and tuning
`w_progress / w_velocity / w_smoothness` yourself is this task.

### Relevant files/locations

| Location | What to do |
|---|---|
| `reward` in [`config/default.yaml`](../config/default.yaml) | Tune the 3 weights (you may also create a separate YAML and pass it via `--config`) |

### Hints

- `r1` dominates near the goal and `r2` dominates far away — they are a **complementary pair**
  ([`COBRA_paper_EN.md`](COBRA_paper_EN.md) §4.2). If their scales differ too drastically, only one of them will have an effect.
- If the weight on `r3` is too large, "not moving" becomes optimal and the snake stops. If too small, the behavior becomes jerky.
- Note that the distance scale (meters) differs between `r1` and `r2`. `r2` is a single step's displacement, so its values tend to be small.
- Example of experimenting with a separate config: `uv run python train_sac.py --config config/exp1.yaml`

### How to verify (acceptance criteria)

- You can find a combination of weights via TensorBoard for which episode cumulative reward and goal-reaching improve.
- The snake's behavior moves toward the waypoint direction, rather than just "staying still" or "flailing around."

---

## Further Work (if you have time to spare)

The diff document [`COBRA_implementation_diff_EN.md`](COBRA_implementation_diff_EN.md) organizes other points where
the implementation diverges from the paper beyond the above (making observations ego-centric, mod-normalizing the phase,
the 1:100 frequency ratio, etc.). If you have time left over, try tackling those as well.
