import argparse
import numpy as np
from env_snake import SnakeEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
import ctypes
import torch

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

# Each CPU physics worker (and the main process) would otherwise spin up its own
# pool of BLAS/torch threads, oversubscribing the cores when running many envs in
# parallel. Pin to 1 so the parallelism comes from the env processes, not BLAS.
# Set at import time so it also applies inside spawned SubprocVecEnv workers.
torch.set_num_threads(1)


def make_env(config, rank):
    """Factory for one SnakeEnv, picklable so SubprocVecEnv can send it to a
    worker process. Each env gets a distinct env_index so its generated maze
    XML path is unique (see SnakeEnv.__init__)."""
    def _init():
        return SnakeEnv(render_mode=None, config=config, env_index=rank)
    return _init


class RewardTermCallback(BaseCallback):
    """Logs the per-term reward breakdown (from env info dicts) to TensorBoard
    so the balance between the three reward components can be monitored during
    training. Each value is averaged over the logging interval."""

    TERMS = (
        "reward_total",
        "term_progress",
        "term_velocity",
        "term_smoothness",
        "term_goal",
        "term_waypoint",
        "term_time",
        "raw_r1_proximity",
        "raw_r2_closing",
        "raw_r3_action_delta",
    )

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self.TERMS:
                if key in info:
                    self.logger.record_mean(f"reward_terms/{key}", info[key])
            # Log the finish rate (% of episodes that reach the goal) only at
            # episode boundaries. VecMonitor injects an "episode" key into info
            # on the step an episode ends; "goal_reached" distinguishes reaching
            # the goal (terminated) from timing out (truncated). Recorded as a
            # percentage and averaged over the logging interval.
            if "episode" in info and "goal_reached" in info:
                self.logger.record_mean(
                    "rollout/finish_rate_pct",
                    100.0 if info["goal_reached"] else 0.0,
                )
                # Average episode length over ONLY the episodes that finished
                # (reached the goal). VecMonitor stores the episode length in
                # info["episode"]["l"]; the overall ep_len_mean is dominated by
                # truncated (timed-out) episodes, so this isolates "how many
                # steps does a successful run take".
                if info["goal_reached"]:
                    self.logger.record_mean(
                        "rollout/finish_ep_len_mean",
                        float(info["episode"]["l"]),
                    )
        return True
import os
import sys
import signal
from config_utils import load_config

model = None  # Declare model globally to access it in save_model_on_exit


def save_model_on_exit(signal, frame):
    if model:  # Check if model is initialized before trying to save
        print("\nInterrupt received, saving model...")
        model.save("models/sac_snake_interrupted")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file (default: config/default.yaml)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device to train on: 'auto', 'cuda', or 'cpu' (default: auto)")
    parser.add_argument("--render", action="store_true",
                        help="Render the MuJoCo viewer during training. Only "
                             "honored when n_envs == 1; parallel training is "
                             "always headless.")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments (SubprocVecEnv). "
                             "Overrides config training.n_envs. 1 = single-env "
                             "(original behavior). ~8 recommended on this machine.")
    parser.add_argument("--no-terrain", action="store_true",
                        help="Train without per-cell terrain (grass/ice/dirt "
                             "tiles) or variable-width caves: bare 1-wide maze. "
                             "Obs stays 42-D (terrain dims read the constant grass "
                             "mu, width dims read the constant 1.0).")
    args = parser.parse_args()

    # Keep the machine awake during long training (Windows). Done inside the
    # main guard so it does NOT re-run in every spawned SubprocVecEnv worker.
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )

    config_path = args.config or "config/default.yaml"
    config = load_config(config_path)
    # Mutate the loaded config so the flag reaches every env built from it
    # (probe, make_env closures — pickled into SubprocVecEnv workers — and
    # the render env).
    if args.no_terrain:
        config.env.terrain_enabled = False

    # Resolve the number of parallel envs: CLI overrides config.
    n_envs = args.n_envs if args.n_envs is not None else config.training.n_envs

    # Setup signal handlers to catch interruptions
    signal.signal(signal.SIGINT, save_model_on_exit)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, save_model_on_exit)  # Handle termination signals

    # Resolve device: "auto" picks CUDA if available, else CPU
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("CUDA available: ", torch.cuda.is_available())
    if device == "cuda":
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using device:", device)

    train_cfg = config.training

    # Validate the env once, in the main process, on a throwaway single instance
    # (check_env only works on a single Gym env, not a VecEnv). It uses the
    # default shared maze path, which cannot race with the per-index worker
    # files created below.
    print(f"Validating environment (n_envs={n_envs})...")
    probe = SnakeEnv(render_mode=None, config=config)
    check_env(probe)
    probe.close()

    # Build the (vectorized) environment. n_envs == 1 keeps the original
    # single-env behavior via DummyVecEnv (same process); n_envs > 1 runs each
    # env in its own subprocess for CPU-parallel data collection. SubprocVecEnv
    # workers are always headless (they cannot host a MuJoCo viewer); the
    # --render viewer only applies to single-env runs.
    if n_envs == 1:
        if args.render:
            vec_env = DummyVecEnv([
                lambda: SnakeEnv(render_mode="human", config=config)
            ])
        else:
            vec_env = DummyVecEnv([make_env(config, 0)])
    else:
        if args.render:
            print("Note: --render is ignored for parallel training "
                  "(n_envs > 1). Use test_model.py to watch a trained policy.")
        vec_env = SubprocVecEnv(
            [make_env(config, i) for i in range(n_envs)],
            start_method="spawn",
        )
    # VecMonitor is required for rollout/ep_rew_mean and ep_len_mean to appear
    # in TensorBoard: SB3 only auto-inserts a Monitor for non-VecEnv inputs.
    vec_env = VecMonitor(vec_env)

    # Each rollout collects n_envs transitions (train_freq=(1,"step") = one
    # vectorized step). Scale gradient_steps by n_envs so we keep the config's
    # intended ratio of one gradient update per collected transition.
    gradient_steps = train_cfg.gradient_steps * n_envs

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size,
        buffer_size=train_cfg.buffer_size,
        tau=train_cfg.tau,
        gamma=train_cfg.gamma,
        ent_coef=train_cfg.ent_coef,
        target_entropy="auto",
        train_freq=tuple(train_cfg.train_freq),
        gradient_steps=gradient_steps,
        tensorboard_log=train_cfg.tensorboard_log,
        # 2 hidden layers of 64 units for both actor (pi) and critic (qf),
        # instead of SB3's default [256, 256].
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[64, 64])),
        device=device,
    )

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # Checkpoint roughly every 2000 env steps. SB3 compares save_freq against the
    # per-vec-step call count (not raw env steps), so divide by n_envs; the saved
    # filenames still use the true timestep count.
    save_freq = max(2000 // n_envs, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path="models/checkpoints",
        name_prefix="sac_snake",
    )
    reward_term_callback = RewardTermCallback()

    print("Starting training...")
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        log_interval=train_cfg.log_interval,
        callback=[checkpoint_callback, reward_term_callback],
    )

    # Save the final model
    model.save(train_cfg.model_save_path)
    vec_env.close()
