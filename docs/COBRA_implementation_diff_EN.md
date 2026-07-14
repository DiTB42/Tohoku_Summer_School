# Summary of Differences Between the Design Document and the Implementation

Subjects:
- **Design document**: [`docs/COBRA_paper_EN.md`](COBRA_paper_EN.md) (a coursework-oriented write-up revising the original paper arXiv:2312.03223v1)
- **Implementation**: the current code in this repository (`env_snake.py` / `cpg/snake_cpg.py` / `config/default.yaml` / `scenes/*.xml` / `mazes/*.py` / `train_sac.py`)

This document exhaustively lists the **differences** between what `COBRA_paper_EN.md` says "should be built" and what is actually implemented in the repository.
It distinguishes between "matching parts," "parts intentionally simplified (per the README)," "parts that diverge from the document," and "parts that used to diverge but have since been fixed."

> Note: the separate document [`AUDIT_REPORT.md`](../AUDIT_REPORT.md) is an audit task list of "**the original paper vs. the implementation**." This document prioritizes a different goal (understanding the gap between the paper — `COBRA_paper_EN.md` — and the current implementation), and **does not mind overlapping in content with the audit report**; it surveys the differences from the paper comprehensively. Items pointed out in the audit and already fixed are also organized in §6 as "former diff / current state."

> **⚠️ This repository is for coursework.** Some differences are **intentionally left unresolved as "tasks for students to implement in order to match the paper."** Currently the items left as student tasks are **item 6 (extending the action space to R⁴)** and **item 9 (reward weight design)**, both marked with `# TODO(student task…)` comments in the code. See [`STUDENT_TASKS_EN.md`](STUDENT_TASKS_EN.md) for how to work through the tasks. All other differences are organized as either "already matches the paper," "an acceptable simplification per the README," or "an acceptable unresolved difference."

---

## 0. Summary (table of differences)

| # | Item | Document (COBRA_paper_EN.md) | Current implementation | Category |
|---|---|---|---|---|
| 1 | CPG phase/amplitude dynamics | Eqs. (9)–(11) | Implemented almost faithfully | ✅ Matches |
| 2 | Snake configuration | Yaw axis only, single CPG, lateral undulation | 12 yaw joints, single CPG | ✅ Matches |
| 3 | Anisotropic friction | Passive wheels or direction-dependent friction | Passive wheels + friction `3 0.005 0.0001` | ✅ Matches |
| 4 | Gait Control | PID or position actuator | Position actuator `kp=0.1` | ✅ Matches (PID not used) |
| 5 | Global Planner | A* + waypoints | A* implemented | ✅ Matches |
| 6 | **Action space** | **$\mathbb{R}^4$** (R, ω, θ, δ), θ∈[-π,π] | **Scalar θ only** ([0.5, 1.0]), R=ω=1, δ=0 fixed | 🎓 **Student task** (TODO scaffolding in place) |
| 7 | **Observation space** | Planar, ego-centric (2D relative position, sin/cos relative heading) | World-coordinate relative vector(3) + absolute pose axis-angle(4) + angular velocity(3) + joint position/velocity(24) = 34D | ⚠️ Diverges (unresolved acceptable difference) |
| 8 | **RL algorithm** | **DDPG** | **SAC** | ⚠️ Diverges (stated explicitly in the README) |
| 9 | **Reward function** | 3 terms of Eq. (12) (r1, r2, r3) | **Already reduced to the 3 terms of Eq. (12)** (all terms not in the paper removed). **Weights are a student design task** | ✅ 3 terms match / 🎓 weight design is a student task |
| 10 | **Maze generation** | Kruskal's algorithm, full maze | Recursive backtracker, `height=3` corridor-only | ⚠️ Diverges |
| 11 | **Frequency ratio** | RL 0.5Hz : CPG 50Hz = 1:100 | Effectively 1:500 (`cpg_frequency=50` is a nominal value) | ⚠️ Diverges |
| 12 | **Network** | Actor 512-256-128, Critic 512-512-256 | SB3 default (256-256), Actor output=1 | ⚠️ Diverges |
| 13 | Training scale | 40,000 episodes / 160 seconds | `total_timesteps=10000` / `max_episode_steps=300` | ⚠️ Diverges |
| 14 | Phase mod normalization | Recommended in §2.3 Tips | Not implemented (φ grows unbounded) | ℹ️ Minor |

Each category is detailed below.

---

## 1. Parts That Match the Document (✅)

Not just differences — the parts that are faithfully implemented are also documented here.

### 1.1 CPG equations (§2) — faithful
[`cpg/snake_cpg.py`](../cpg/snake_cpg.py) implements Eqs. (9)–(11) almost exactly as-is.

- **Phase dynamics** `φ̇ = ω + A·φ + B·θ` → [snake_cpg.py:64](../cpg/snake_cpg.py#L64)
- **Amplitude dynamics (2nd order, critically damped)** `r̈ = a·[a/4·(R−r) − ṙ]` → [snake_cpg.py:67-72](../cpg/snake_cpg.py#L67-L72) (Euler integration)
- **Output** `x = r·sin(φ) + δ` → [snake_cpg.py:75](../cpg/snake_cpg.py#L75)
- **Coupling matrix A** (tridiagonal, zero row sums = graph Laplacian, both ends −μ / interior −2μ) → [snake_cpg.py:31-42](../cpg/snake_cpg.py#L31-L42). Verifying by hand, the diagonal is `[−μ, −2μ, …, −2μ, −μ]`, the off-diagonals are `μ`, and each row sums to zero, matching the definition in document §2.3(a).
- **Coupling matrix B** ($n\times(n-1)$, `(Bθ)_1=θ_1`, `(Bθ)_i=θ_i−θ_{i-1}`, `(Bθ)_n=−θ_{n-1}`) → [snake_cpg.py:44-50](../cpg/snake_cpg.py#L44-L50). Matches document §2.3(a).
- **Phase difference θ replicated as a single value** (document §2.3 / paper IV.D.3 "single value") → [snake_cpg.py:52-55](../cpg/snake_cpg.py#L52-L55) `np.full(n-1, phase_shift)`. The former `(i+1)*θ` increasing-sequence bug (AUDIT item 9) has been fixed.

### 1.2 Yaw axis only, single CPG (§3) — matches
Every actuated joint in [`scenes/snake.xml`](../scenes/snake.xml) is `axis="0 0 1"` (yaw), and there is a single CPG system (`PaperCPG(n_joints=12)`, [env_snake.py:112](../env_snake.py#L112)). This follows document §3's "one CPG, lateral undulation."

### 1.3 Anisotropic friction (§5-1) — implemented with passive wheels
Of the two options recommended in document §5-1, "**passive wheels**" was adopted. Each link in [`scenes/snake.xml`](../scenes/snake.xml) has a cylindrical wheel free to rotate (`axis="0 1 0"`, `limited="false"`). In addition, there is a default friction `friction="3 0.005 0.0001"` ([snake.xml:7](../scenes/snake.xml#L7)). This configuration rolls forward while suppressing sideways slip, consistent with what the document requires.

### 1.4 Gait Control (§4.5) — position actuator
Document §4.5's "substitute PID with MuJoCo's position actuator" was adopted. [snake.xml:239-250](../scenes/snake.xml#L239-L250) uses `<position kp="0.1">`. However, `kv` (damping) is unspecified (default value), and `kp=0.1` is on the low side, leaving room for tuning tracking performance (discussed further in §3.4).

### 1.5 Global Path Planning (§4.1) — A*
`astar()` in [`mazes/make_maze.py`](../mazes/make_maze.py) computes the shortest path on the occupancy grid, and path cells are used as waypoints ([env_snake.py:80-97](../env_snake.py#L80-L97)).

---

## 2. Major Divergences (⚠️)

### 2.1 Action space: $\mathbb{R}^4$ (document) → scalar θ only (implementation) 🎓 Student task

> **This difference is intentionally left in place (Student Task 1).** While keeping a working θ-only implementation,
> `# TODO(student task 1 …)` markers are placed at the extension points
> ([env_snake.py `action_space`](../env_snake.py#L36) and near
> [`theta = action` in `step()`](../env_snake.py#L208)). See Task 1 in [`STUDENT_TASKS_EN.md`](STUDENT_TASKS_EN.md) for implementation steps.

**Document §3** specifies the action space as $\mathbb{R}^4$:
- Amplitude $R \in [0, 1.5]$
- Frequency $\omega \in [-0.1, 0.1]$
- Phase difference $\theta \in [-\pi, \pi]$
- Offset $\delta \in [-0.1, 0.1]$
- → Actor output is tanh(4)

**The implementation** reduces the action to a **single scalar dimension (θ only)**:
- [env_snake.py:36-40](../env_snake.py#L36-L40): `action_space = Box(low=[0.5], high=[1.0])` → 1D
- [env_snake.py:208-209](../env_snake.py#L208-L209): only `theta = action` is passed to the CPG; `R, ω, δ` are fixed values ([env_snake.py:26-28](../env_snake.py#L26-L28), `config.cpg.base_amplitude=1.0` / `base_omega=1.0` / `base_delta=0.0`)

This is an **intentional simplification** following the README's experiment table (Exp3 "Reduced: θ only is best"). However, it diverges from document §3's design in the following ways:

1. **Dimensionality**: 4 → 1.
2. **Range of θ**: the document specifies $[-\pi, \pi]$, while the implementation uses **[0.5, 1.0]** ([config/default.yaml:49-50](../config/default.yaml#L49-L50)). The wavelength is restricted to a narrow positive range.
3. **Value of ω**: the document assumes small values within $\omega \in [-0.1, 0.1]$, whereas the implementation has **ω=1.0 fixed** (the CPG class's default `0.05` ([snake_cpg.py:19](../cpg/snake_cpg.py#L19)) is also overridden to 1.0 via `set_parameters`). This is 10–20x larger than the frequency the document envisions.
4. **δ fixed at 0, disabling the turning mechanism**: document §2.1 explains that "the offset $\delta$ bends the body into an arc to produce **turning**," but since the implementation fixes δ=0, this turning mechanism is unavailable. Behavior is determined solely by the learned parameter θ (wavelength), leaving few active steering means. This works only because the task uses a corridor (nearly straight).

### 2.2 Observation space: planar, ego-centric (document) → 34D leaning on world coordinates (implementation)

> **This difference is left unaddressed in this iteration of the coursework (acceptable difference).** It is not selected as a student task, but
> could become an extension task for students with time to spare (see recommended action 1 in §6).

**Document §3's proposal** (reduction for planar motion, **everything ego-centric**):
- Joint angles $\mathbb{R}^n$
- IMU $\mathbb{R}^{2\text{–}3}$
- Waypoint's **relative position** as seen from the robot's frame $\mathbb{R}^2$
- Waypoint's **relative heading** as seen from the robot's frame ($\sin\psi, \cos\psi$) $\mathbb{R}^2$
- ※ Absolute position in world coordinates is not included (ego-centric design philosophy)

**The implementation's** 34 dimensions ([env_snake.py:182-202](../env_snake.py#L182-L202)):
- Joint position (12) + joint velocity (12)
- `head_to_target_vec` (3): `current_target − head_pos`
- `head_orientation` axis-angle (4)
- `head_angular_vel` (3)

Differences:
1. **Not reduced to planar**: rather than the document's planar version (2D, sin/cos), it's a 3D vector + axis-angle(4), a configuration **closer to the original paper's 21 dimensions**.
2. **Not ego-centric (robot frame)**: `head_to_target_vec` is a **difference vector in world coordinates** and is not rotated by the head's orientation ([env_snake.py:186-187](../env_snake.py#L186-L187)). Likewise, `head_orientation` is an **absolute pose in the world frame** ([env_snake.py:165-176](../env_snake.py#L165-L176)). This departs from the ego-centric principle the document emphasizes — "relative quantities as seen from the robot's frame / no absolute position."
3. **Relative heading ($\sin\psi,\cos\psi$) is absent from the observation**: the document recommends including the relative heading to the goal in the observation. In the implementation, this relative heading appears in neither the observation nor the reward (previously there was a heading-alignment term `cos(angle_to_target)` on the reward side, but it was removed in §2.4 since it's not in the paper's Eq. (12)). The policy has no direct access to heading information.
4. **Joint velocity (12) added**: an item not on the document's proposed list.

### 2.3 RL algorithm: DDPG (document) → SAC (implementation)

**Document §4.3** specifies **DDPG** as the RL algorithm.
**The implementation** uses **SAC** (Stable-Baselines3) ([train_sac.py:53](../train_sac.py#L53)). The README's "System Overview / Technical Details" also states SAC explicitly, making this an intentional change. Since the document follows the original paper's description (DDPG) as-is, this is where they diverge.

### 2.4 Reward function: already reduced to the 3 terms of Eq. (12) 🎓 weight design is a student task

**Document §4.2 / Eq. (12)**'s reward has 3 terms:
- $r_1 = 1/(0.1 + d_t)$ (proximity)
- $r_2 = d_{t-1} - d_t$ (approach speed)
- $r_3 = \|a_t - a_{t-1}\|$ (action-change penalty)
- The total reward takes the form $w_1 r_1 + w_2 r_2 - w_3 r_3$ (weights are not in the paper and are designed as coursework)

**The current implementation** consists of **only** these 3 terms of Eq. (12) (`step()` in [env_snake.py](../env_snake.py#L262)):

```python
reward = (
    reward_cfg.w_progress * r1      # r1: proximity
    + reward_cfg.w_velocity * r2    # r2: approach speed
    - reward_cfg.w_smoothness * r3  # r3: action-change penalty (subtracted)
)
```

Previously, this implementation had many additional terms not in the paper's Eq. (12)
(`waypoint_bonus` / `goal_reward` / `control_penalty` /
`w_heading * cos(angle_to_target)` / `self_collision_penalty` / `tail_proximity_penalty` / `wall_penalty`)
stacked on top, but they have **all been removed to match the paper's reward** (see §6). At the same time, the old
implementation's hardcoded coefficient `r3 = -0.3 * ||Δa||` (a double coefficient) was also removed, unifying it into
`r3 = ||Δa||` subtracted via `w_smoothness`, matching the paper's sign and form.

Additional notes:
- The waypoint **index-advancement logic** and **episode termination on reaching the goal** (`terminated`) are
  essential for navigation and have therefore been **kept as control flow** (only their contribution to the reward was removed).
  Hence, as in the paper, "there is no explicit bonus for reaching the goal — the `r1` proximity reward alone drives goal-seeking."
- The **weighting coefficients $w_1, w_2, w_3$** (`w_progress` / `w_velocity` / `w_smoothness`) are not specified in the
  paper, and are left as a task for **students to design and tune**
  ([`STUDENT_TASKS_EN.md`](STUDENT_TASKS_EN.md) Task 2,
  the `reward` section of [config/default.yaml](../config/default.yaml#L6)). Aside from the design of these 3 weights, the reward's term structure matches the paper's Eq. (12).

### 2.5 Maze generation: Kruskal, full maze (document) → recursive backtracker, corridor-only (implementation)

**Document §4.1** calls for "random maze generation via **Kruskal's algorithm**" and "randomizing during training for zero-shot transfer."

**The implementation**:
- The generation algorithm is a **recursive backtracker (DFS)**, `_carve` ([make_maze.py:9-29](../mazes/make_maze.py#L9-L29)), not Kruskal's algorithm.
- The maze size is **`height=3`** ([config/default.yaml:47](../config/default.yaml#L47)). Since the recursive backtracker carves in units of 2 cells, at `height=3` no vertical branching can be generated in principle, making it effectively a **single corridor**. The README's "Limitations" section explicitly states that a full maze fails with NaN, so it is intentionally run corridor-only.
- Consequently, the "**zero-shot transfer** to random mazes" touted in document §1/§4.1 is not currently something this configuration verifies.

### 2.6 Frequency ratio: 1:100 (document) → effectively 1:500 (implementation)

**Document §4.3** specifies RL 0.5Hz : CPG 50Hz = **1:100**.

**The implementation's** settings and effective values:
- [config/default.yaml:44-45](../config/default.yaml#L44-L45): `cpg_frequency=50`, `rl_frequency=0.1`
- [env_snake.py:24](../env_snake.py#L24): `sim_steps_per_rl_step = int(50/0.1) = 500` → one action is **held for 500 physics steps**
- The physics timestep is unspecified in the XML, so it defaults to MuJoCo's **0.002s** (no `<option timestep>` in `scene.xml`/`snake.xml`). So one action spans `500 × 0.002 = 1.0 second`.
- The CPG's dt is `model.opt.timestep` (=0.002s, [env_snake.py:112](../env_snake.py#L112)), so **the CPG actually updates at roughly 500Hz**.

Result:
1. **The RL:CPG ratio is 1:500** (different from the document's 1:100).
2. **`cpg_frequency=50` is a nominal value** and does not match the actual CPG update rate (≈500Hz). This "50Hz" label is only used as the numerator in `sim_steps_per_rl_step`.
3. RL's decision period is roughly **1Hz** (the document specifies 0.5Hz).

> Note that matching the CPG dt to `model.opt.timestep` was itself a valid fix for AUDIT item 2 (a 10x dt mismatch). However, it is no longer consistent with the document's "50Hz / 0.5Hz / 1:100" figures.

### 2.7 Network architecture (§4.4)

**Document §4.4**:
- Actor: State → 512 → 256 → 128 → tanh(**4**)
- Critic: (State, Action) → 512 → 512 → 256 → 1

**The implementation**: `SAC(policy="MlpPolicy", …)` does not specify `net_arch` ([train_sac.py:53-68](../train_sac.py#L53-L68)), so **SB3's default (256-256)** is used. Also, the Actor's output dimension is **1** (θ only). Both the layer configuration and output dimension differ from the document.

### 2.8 Training scale / episode settings (§4.3)

| Item | Document §4.3 | Implementation |
|---|---|---|
| Total training amount | 40,000 episodes | `total_timesteps=10000` (environment steps) [config/default.yaml:38](../config/default.yaml#L38) |
| Episode length | 160 seconds | `max_episode_steps=300` (RL steps) → roughly 300 seconds in real time [config/default.yaml:46](../config/default.yaml#L46) |
| Goal relocation range | 8m × 8m | Random valid cell within the corridor |

The training amount is far smaller than the document envisions (on the order of a few thousand steps), configured for quick verification runs.

---

## 3. Minor Differences / Implementation Notes (ℹ️)

### 3.1 Phase mod normalization not implemented (§2.3 Tips)
Document §2.3's implementation tip warns: "take φ mod $2\pi$, and normalize the coupling term's difference into $[-\pi,\pi)$ with `arctan2(sin, cos)`." The implementation computes the coupling term with φ **left to increase monotonically without taking mod** ([snake_cpg.py:64,70](../cpg/snake_cpg.py#L64-L70)). Since there's no wraparound, this doesn't break anything for now, but as φ grows large over long episodes it becomes disadvantageous from a floating-point precision standpoint. This differs from the document's recommended implementation.

### 3.2 Separation of waypoint-reached detection and the reward target
The implementation uses two separate references: reward shaping (r1, r2) uses "the midpoint (lookahead) between the current and next waypoint" ([env_snake.py:155-164](../env_snake.py#L155-L164)), while reached-detection uses "the waypoint's actual coordinates" ([env_snake.py:234-249](../env_snake.py#L234-L249)). This is a reasonable outcome of the fix for AUDIT item 6, but this "midpoint lookahead" trick is an implementation-specific element not documented in the design doc.

### 3.3 Waypoint placement is per grid cell
Document §4.1 says "place waypoints so the distance between adjacent waypoints stays within a fixed range," but the implementation directly uses the grid cells returned by A* **as-is as waypoints** (1 cell = 1m in world units, [env_snake.py:97](../env_snake.py#L97)). While evenly spaced, this is not "repositioning based on a distance constraint."

### 3.4 PID gains
The position actuator's `kp=0.1` is on the low side, and `kv` is unspecified. There remains room for the staged tuning that document §4.5/§5-3 calls for: "kp, kv tuning is needed" / "first confirm straight-line motion and turning with the CPG manually fixed."

---

## 4. Differences Accepted as "Intentional Simplifications" (per the README)

The following diverge from the document but are intentional design choices explicitly stated in the README — they are not "bugs that should be fixed to match the document" (in the same spirit as [KEEP] items in AUDIT_REPORT).

- **Adoption of SAC** (instead of DDPG) (§2.3) — stated explicitly in the README.
- **Corridor-only (`height=3`)** (§2.5) — stated explicitly in the README's "Limitations / Future Work." Expected to be revisited together when full-maze support is addressed.
- **A wheeled planar snake (single CPG)**, a different hardware model from COBRA (pitch/yaw 2-CPG sidewinding) — document §3 itself premises "reinterpreting for yaw-axis only," so this is directionally consistent.
- **1:500 frequency ratio, non-ego-centric observations, default network, and reduced training scale** (§2.2, §2.6, §2.7, §2.8) — unresolved acceptable differences to make iteration easier within a short coursework timeframe. Extension tasks in §6.

> **Differences left as student tasks** (item #6 action space θ→R⁴ in §2.1, item #9 reward weight design in §2.4) are distinct from the "acceptable differences" above. These are **unfinished parts students are expected to implement to match the paper**, and are explicitly listed as tasks in [`STUDENT_TASKS_EN.md`](STUDENT_TASKS_EN.md). Note that while the README's Exp3 rates the θ-only reduction in #6 as "best," this coursework frames it as a task to extend to the $\mathbb{R}^4$ of paper §3.

---

## 5. Items That Used to Diverge From the Paper but Now Match (fixed)

Among the inconsistencies with the paper pointed out in the audit report ([`AUDIT_REPORT.md`](../AUDIT_REPORT.md)), the ones that have **already been resolved in the current code and now match the paper (`COBRA_paper_EN.md`)** are recorded here, even at the risk of duplication. These are not "current differences," but they matter for the overall picture of how the implementation corresponds to the paper.

| Former difference | Basis in paper | Former implementation | Current state (after fix) | Reference |
|---|---|---|---|---|
| CPG amplitude's 2nd-order dynamics | §2.2 Eq. (9) `r̈=a[a/4(R−r)−ṙ]` | Disabled, jumping instantly to `x=R·sinφ+δ` | Integrates `r,ṙ` and computes `x=r·sinφ+δ`. Tracks R changes smoothly | [snake_cpg.py:61-77](../cpg/snake_cpg.py#L61-L77) |
| Array of phase differences θ | §2.3 / IV.D.3 "single value" | Increasing sequence `θ[i]=(i+1)·θ` (breaks the traveling wave) | Constant vector via `np.full(n−1, θ)` | [snake_cpg.py:52-55](../cpg/snake_cpg.py#L52-L55) |
| CPG dt vs. physics timestep | §5 / §4.3 | CPG dt=0.02 vs. physics 0.002, a 10x mismatch causing the phase to advance 10x too fast | CPG dt unified to `model.opt.timestep` | [env_snake.py:112](../env_snake.py#L112) |
| CPG reset between episodes | §4.3 (gait reproducibility) | Generated once in `__init__`; phase φ carried over into the next episode | `PaperCPG` regenerated and state reinitialized on every `reset()` | [env_snake.py:109-116](../env_snake.py#L109-L116) |
| Reward terms not in the paper mixed in | §4.2 / Eq. (12) (reward has only 3 terms) | Many extra terms were stacked on: `waypoint_bonus`, `goal_reward`, `control_penalty`, heading alignment, self-collision, head-tail, wall | Terms not in the paper **all removed**, reduced to the 3 terms `w_progress·r1 + w_velocity·r2 − w_smoothness·r3` | [env_snake.py `step()`](../env_snake.py#L262) |
| Magic number in r3 (double coefficient) | §4.2 (r3 = ‖Δa‖ subtracted via a weight) | `r3 = -0.3·‖Δa‖`, with `w_smoothness` multiplied on top — a double coefficient | `r3 = ‖Δa‖`, subtracted via `- w_smoothness·r3` | [env_snake.py `step()`](../env_snake.py#L262) |
| Waypoint-reached detection | §4.1 (judged by waypoint coordinates) | Reached when approaching the midpoint (lookahead) | Judged by distance to the actual waypoint coordinates; the midpoint is separated out for reward shaping only | [env_snake.py:234-249](../env_snake.py#L234-L249) |
| No heading information in the observation | §3 (IMU + relative pose) | Only 27 dimensions: joint_pos + joint_vel + relative vector | Added head pose axis-angle(4) + angular velocity(3), bringing it to 34 dimensions | [env_snake.py:182-202](../env_snake.py#L182-L202) |
| Hardcoded absolute path / debug print | — | Personal path in `test_model.py`, leftover `print(current_step)` | Converted to a relative path, print removed | [test_model.py:7](../test_model.py#L7) |
| Magic numbers in configuration | §4.2 (weights are meant to be designed) | Reward coefficients, CPG hyperparameters, and SAC hyperparameters hardcoded in code | Externalized via `config/default.yaml` + a dataclass loader | [config/default.yaml](../config/default.yaml), [config_utils.py](../config_utils.py) |

> Note: while these are "already fixed to match the paper," bear in mind that the fixes gave rise to **new differences**, such as the **nominal-vs-effective frequency mismatch in §2.6** (matching the CPG dt to the physics timestep pushed the actual CPG rate to roughly 500Hz, turning `cpg_frequency=50` into a nominal value).

---

## 6. Recommended Actions (priority order if aligning the implementation with the document)

> **Assigned as student tasks ([`STUDENT_TASKS_EN.md`](STUDENT_TASKS_EN.md)) are item 2 below (extending the action space to R⁴) and the reward weight design in §2.4.** Items 1, 3, 4, and 5 are unresolved acceptable differences that could become extension tasks for students with time to spare.

Rough order of attack if you want to bring the implementation closer to the design in document §3/§4:

1. **Make the observation ego-centric** (§2.2): rotate `head_to_target_vec` into the robot frame using the head's rotation matrix, and add the target's relative heading to the observation as $(\sin\psi,\cos\psi)$. This lets the policy directly observe the relationship between the snake's heading and the target's direction, tending to stabilize learning (extending AUDIT item 10).
2. **Extend the action space** (§2.1) 🎓 **Student Task 1**: first add δ to the action to **enable the turning mechanism** (a prerequisite for escaping the corridor and full-maze support). Then progressively open up R, ω toward $\mathbb{R}^4$. Also widen θ's range toward $[-\pi,\pi]$.
3. **Clean up frequency naming** (§2.6): resolve the mismatch between `cpg_frequency`'s nominal value and the actual CPG update rate (=1/`model.opt.timestep`), and explicitly manage the RL:CPG ratio. If needed, specify the physics `timestep` explicitly in the XML.
4. **Extend maze generation** (§2.5): set `height` to an odd number ≥5 (enabling vertical corridors to be generated), and replace with Kruskal's algorithm if needed. Pair this with reproducing/addressing the full-maze NaN issue.
5. **Network / training scale** (§2.7, §2.8): match `net_arch=[512,256,128]` etc. to the document, and bring `total_timesteps` up to a production-scale value.

---

*Based on the implementation as of the referenced commit. Source: S. Jiang et al., "Hierarchical RL-Guided Large-scale Navigation of a Snake Robot," arXiv:2312.03223v1, 2023 / this repository's `COBRA_paper_EN.md`.*
