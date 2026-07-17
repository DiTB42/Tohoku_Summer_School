import argparse
import numpy as np
from stable_baselines3 import SAC
from env_snake import SnakeEnv
from config_utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, nargs="?",
                        default="models/sac_snake_final.zip",
                        help="Path to the saved model .zip "
                             "(e.g. models/checkpoints/sac_snake_600_steps.zip)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to the YAML config (must match what the "
                             "model was trained with, e.g. maze size). "
                             "Default: config/default.yaml")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--render-slowdown", type=float, default=None,
                        help="Playback speed for the viewer. 1.0 = real "
                             "physics time; larger = slower (e.g. 4.0 = 4x "
                             "slower), smaller = faster. If omitted, uses the "
                             "config value or the env default (2.0).")
    args = parser.parse_args()

    # Load the model
    model = SAC.load(args.model_path)

    # Create the environment in rendering mode, using the same config the
    # model was trained with (maze size, frequencies, reward, ...).
    config = load_config(args.config)
    env = SnakeEnv(render_mode="human", config=config,
                   render_slowdown=args.render_slowdown)
    n_episodes = args.episodes

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
        print(f"Episode {ep+1}: Total reward = {total_reward:.2f}, Steps = {step}")

    env.close()
