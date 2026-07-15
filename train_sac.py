import argparse
import numpy as np
from env_snake import SnakeEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback


class RewardTermCallback(BaseCallback):
    """Logs the per-term reward breakdown (from env info dicts) to TensorBoard
    so the balance between the three reward components can be monitored during
    training. Each value is averaged over the logging interval."""

    TERMS = (
        "reward_total",
        "term_progress",
        "term_velocity",
        "term_smoothness",
        "raw_r1_proximity",
        "raw_r2_closing",
        "raw_r3_action_delta",
    )

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in self.TERMS:
                if key in info:
                    self.logger.record_mean(f"reward_terms/{key}", info[key])
        return True
import os
import torch
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
                        help="Render the MuJoCo viewer during training (slower; off by default)")
    args = parser.parse_args()

    config_path = args.config or "config/default.yaml"
    config = load_config(config_path)

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

    # Instantiate and validate the environment
    env = SnakeEnv(render_mode="human" if args.render else None, config=config)
    check_env(env)

    # Create the SAC model on GPU
    train_cfg = config.training
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size,
        buffer_size=train_cfg.buffer_size,
        tau=train_cfg.tau,
        gamma=train_cfg.gamma,
        ent_coef=train_cfg.ent_coef,
        target_entropy="auto",
        train_freq=tuple(train_cfg.train_freq),
        gradient_steps=train_cfg.gradient_steps,
        tensorboard_log=train_cfg.tensorboard_log,
        # 2 hidden layers of 64 units for both actor (pi) and critic (qf),
        # instead of SB3's default [256, 256].
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[64, 64])),
        device=device,
    )

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # Save a checkpoint roughly every 200 updates so training progress can be
    # visualized. With train_freq=[1, step] and gradient_steps=1, one env step
    # corresponds to one gradient update, so save_freq is in env steps.
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
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
    env.close()
