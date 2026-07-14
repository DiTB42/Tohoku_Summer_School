# Intelligent Snake: Autonomous Robot Navigation via RL & CPG

An autonomous snake robot navigation system using Reinforcement Learning in MuJoCo simulation. Building on a gesture-controlled base from the 2024 edition, the 2025 iteration empowers the robot to navigate fully autonomously using a hierarchical control scheme combining A* global planning, SAC-based local planning, and a Central Pattern Generator for biologically-inspired locomotion.

> **Authors:** Charlotte L. Primiceri · Serena Trovalusci · Davide de Ciutiis · Harini Satyavada · Vishnu Anand  
> **Program:** TESP 2025 — Tohoku University (Japan)                   
> **Slides:** [`IntelligentSnake.pdf`](IntelligentSnake.pdf)

---

## What is this project?

Snake robots are ideal for exploring hard-to-reach or hazardous environments, from disaster zones to planetary surfaces. When operating on unknown complex terrains, real-time gait adjustment based on environmental perception is key for navigation.

This project takes a gesture-controlled snake robot and makes it fully autonomous, using artificial intelligence to **perceive, plan, and move without human intervention**:

```
┌─────────────────────────────────────────────────────────────────┐
│               Hierarchical Control Scheme                       │
│                                                                 │
│  Occupancy Grid                                                 │
│       │                                                         │
│       ▼                                                         │
│  Global Planning  ──  A* Algorithm  ──►  Waypoints              │
│                                              │                  │
│                                              ▼                  │
│  Local Planning   ──  SAC (RL)      ──►  CPG params (R,θ,ω,δ)  │
│                                              │                  │
│                                              ▼                  │
│  Gait Generation  ──  CPG           ──►  Joint targets x[i]     │
│                                              │                  │
│                                              ▼                  │
│  Gait Tracking    ──  MuJoCo        ──►  Snake locomotion       │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Overview

| Component | Technology | Role |
|-----------|-----------|------|
| **Gesture Control** *(base)* | MediaPipe + TCP Socket | Human wrist tracking → snake direction |
| **Global Planning** | A* Algorithm | Computes waypoints from occupancy grid |
| **Local Planning** | SAC (Stable-Baselines3) | Learns CPG parameters to navigate toward waypoints |
| **Gait Generation** | Central Pattern Generator (CPG) | Translates parameters into sinusoidal joint trajectories |
| **Simulation** | MuJoCo | Physics engine for training and evaluation |

---

## Repository Structure

```
.
├── env_snake.py              # Custom Gym environment
├── train_sac.py              # SAC training script
├── test_model.py             # Rollout a trained model in the viewer
├── config_utils.py           # YAML -> dataclass config loader
├── requirements.txt
├── config/
│   └── default.yaml          # Default reward/CPG/training/env parameters
├── scenes/                   # MuJoCo XML simulation files
├── models/                   # Trained model weights (.zip files, created by train_sac.py)
├── mazes/
│   ├── make_maze.py          # Maze generation and A* algorithm
│   └── mujoco_tools.py       # Dynamic XML generation for mazes
└── cpg/
    └── snake_cpg.py          # Central Pattern Generator for joint control
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create environment and install dependencies

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

### 3. Train the snake

```bash
uv run python train_sac.py
```

Optionally point at a custom config, and/or force a specific device (defaults to `auto`, which uses CUDA if available and falls back to CPU):

```bash
uv run python train_sac.py --config config/exp1.yaml --device cpu
```

This will initialize the MuJoCo environment, begin SAC training, log statistics to TensorBoard, and save model weights to `models/`.

### 4. Roll out a trained model

```bash
uv run python test_model.py
```

Loads `models/sac_snake_final.zip` and runs it in the MuJoCo viewer for a few episodes.

### 5. Monitor training

```bash
uv run tensorboard --logdir sac_snake_tensorboard/
```

Then open [http://localhost:6006](http://localhost:6006).

---

## Technical Details

### Gesture-Controlled Snake *(base system)*

The 2024 base uses **MediaPipe** Pose Landmark Detection to track the right wrist position from a webcam feed. The wrist position `[x, y, is_hand_in_frame]` is streamed via a **TCP socket server** to the MuJoCo control system, which maps it to sinusoidal CPG parameters:

```
target q[i] = amp * sin(θ + φ * i) + B

ω = f(y)    angular velocity
B = f(x)    offset
φ = π/4     phase
θ_new = θ_old + ω·t
```

### SAC Algorithm *(local planning)*

The **Soft Actor-Critic** agent learns to navigate by interacting with a custom snake environment:

- **Observation space** (34-dim): joint positions (12) + joint velocities (12) + vector to current waypoint (3) + head orientation as axis-angle (4) + head angular velocity (3)
- **Action space**: scalar `θ` (CPG phase-shift parameter), with `R=1, ω=1, δ=0` fixed — the reduced space that performed best in the experiments below
- **Reward**: progress toward the waypoint, closing velocity, heading alignment with the target, a one-shot bonus on reaching each waypoint, and a bonus on reaching the goal
- **Penalties**: jerky action changes, control effort, self-collision, wall collision, and head/tail proximity (coiling)

### Central Pattern Generator *(gait generation)*

The CPG is a biologically-inspired neural circuit that generates coordinated rhythmic joint trajectories, following the paper's Equation 9: phase dynamics `φ̇ = ω + A·φ + B·θ` and amplitude dynamics `r̈ = a·[a/4·(R − r) − ṙ]`, integrated each step and combined into the serpenoid wave `x = r · sin(φ) + δ`.

---

## Experiments

| Experiment | Environment | Action Space | Result |
|-----------|------------|-------------|--------|
| **Exp 1** | Maze | Full CPG: R, θ, ω, δ | Poor — unstable and inefficient behavior |
| **Exp 2** | Corridor (simplified) | Full CPG: R, θ, ω, δ | Still erratic movements |
| **Exp 3** *(best)* | Corridor | Reduced: θ only (R=1, ω=1, δ=0) | Improved stability and behavior |

Reducing the action space to a single learned parameter (θ, phase shift) while fixing the others significantly improved learning stability. The corridor environment was used instead of the full maze due to computational complexity.

---

## Demo

**Gesture-Controlled Snake** *(base system)*

https://github.com/user-attachments/assets/01d69eb5-d380-405c-8512-f4a1f9781c14

**Autonomous RL Snake** *(Experiment 3 — best performance)*

https://github.com/user-attachments/assets/5ae85dd3-5b0f-47d9-a050-bf8340631566

---

## Limitations

- Training environment prone to computational instability in complex scenarios.
- Currently limited to corridor environments — full maze produces NaN errors in complex configurations.
- Ring and pinky finger movements of the operator are not tracked in the gesture-controlled version.

---

## Future Work

- Training to enable full maze navigation (first with A* waypoints, then without any waypoints).
- Improved rewards and penalty functions.
- Memory and vision modules integration.
- Real-life bot to validate simulations.
- Improved simulation stability.

---

## References

- Jiang et al., [*Hierarchical RL-Guided Large-scale Navigation of a Snake Robot*](https://arxiv.org/abs/2312.03223v1), 2023
- Bing et al., *Smooth Gait Transition of Body Shape and Locomotion Speed Based on CPG Control For Snake-like Robot*, 2017
- Qiao et al., *Sigmoid transition approach of the central pattern generator-based controller for the snake-like robot*, 2016
- [MuJoCo Physics Engine](https://mujoco.org/)
- [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
