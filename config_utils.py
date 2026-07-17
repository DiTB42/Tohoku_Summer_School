"""Dataclass-based config loading for the snake environment / CPG / SAC training.

Usage:
    from config_utils import load_config
    config = load_config("config/exp1.yaml")  # or load_config(None) for defaults
"""
from dataclasses import dataclass, fields
import yaml


@dataclass
class RewardConfig:
    # Paper reward eq.(12) has three terms only (r1, r2, r3). The three
    # weights below are the design surface left to the student (see
    # docs/STUDENT_TASKS_JP.md, task 2).
    w_progress: float = 15.0
    w_velocity: float = 200.0
    w_smoothness: float = 1.0
    waypoint_threshold: float = 0.7


@dataclass
class CPGConfig:
    frequency_hz: float = 50.0
    convergence_rate: float = 63.0
    coupling_strength: float = 3.0
    base_amplitude: float = 1.0
    base_omega: float = 1.0
    base_delta: float = 0.0
    base_phase_shift: float = 0.7853981633974483


@dataclass
class TrainingConfig:
    algorithm: str = "SAC"
    learning_rate: float = 1.0e-3
    batch_size: int = 1028
    buffer_size: int = 9_000_000
    tau: float = 0.18
    gamma: float = 0.5
    ent_coef: str = "auto"
    train_freq: list = None
    gradient_steps: int = 1
    # Number of parallel environments (SubprocVecEnv) for data collection.
    # 1 = single-env DummyVecEnv (original behavior). The CLI --n-envs overrides
    # this. gradient_steps is multiplied by n_envs at train time to keep one
    # gradient update per collected transition (see train_sac.py).
    n_envs: int = 1
    total_timesteps: int = 10000
    log_interval: int = 4
    tensorboard_log: str = "sac_snake_tensorboard/"
    model_save_path: str = "models/sac_snake_final"

    def __post_init__(self):
        if self.train_freq is None:
            self.train_freq = [1, "step"]


@dataclass
class EnvConfig:
    cpg_frequency: int = 50
    rl_frequency: float = 0.1
    max_episode_steps: int = 300
    maze_height: int = 3
    maze_width: int = 7
    # 学生課題1: 行動空間 R⁴ = (R, omega, theta, delta). Per-parameter bounds,
    # each a 4-element list ordered [R, omega, theta, delta].
    action_low: list = None
    action_high: list = None
    # Minimum Manhattan distance (in maze cells) between the snake's start
    # and the randomly chosen goal, so the goal never spawns right next to it.
    min_goal_distance: int = 2
    # Playback speed for the human viewer. 1.0 = real physics time; larger =
    # slower (e.g. 4.0 = 4x slower), smaller = faster. Overridable per-run via
    # the --render-slowdown CLI arg in test_model.py.
    render_slowdown: float = 2.0

    def __post_init__(self):
        if self.action_low is None:
            self.action_low = [0.2, 0.5, 0.2, -0.5]
        if self.action_high is None:
            self.action_high = [1.0, 3.0, 0.8, 0.5]


@dataclass
class Config:
    reward: RewardConfig
    cpg: CPGConfig
    training: TrainingConfig
    env: EnvConfig


def _build_dataclass(cls, data):
    data = data or {}
    valid_keys = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid_keys})


def load_config(path=None):
    """Load a Config from a YAML file. Missing keys/sections fall back to
    defaults, so passing path=None reproduces the original hardcoded behavior."""
    raw = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    return Config(
        reward=_build_dataclass(RewardConfig, raw.get("reward")),
        cpg=_build_dataclass(CPGConfig, raw.get("cpg")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        env=_build_dataclass(EnvConfig, raw.get("env")),
    )
