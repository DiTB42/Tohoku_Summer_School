import argparse
import numpy as np
from env_snake import SnakeEnv
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
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
                        help="Path to a YAML config file (defaults built in if omitted)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device to train on: 'auto', 'cuda', or 'cpu' (default: auto)")
    parser.add_argument("--render", action="store_true",
                        help="Render the MuJoCo viewer during training (slower; off by default)")
    args = parser.parse_args()

    config = load_config(args.config)

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
        device=device,
    )

    # Create output directories
    os.makedirs("models", exist_ok=True)

    print("Starting training...")
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        log_interval=train_cfg.log_interval,
    )

    # Save the final model
    model.save(train_cfg.model_save_path)
    env.close()
