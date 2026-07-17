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

    # Collect every observation seen across all episodes so we can report the
    # per-dimension standard deviation at the end. A dimension whose std is
    # ~0 is (near-)constant and can be dropped from the observation with
    # essentially no information loss. See _get_obs() in env_snake.py for the
    # layout these labels mirror.
    obs_log = []
    act_dim = int(env.action_space.shape[0])
    obs_labels = (
        [f"jpos{i}" for i in range(12)]    # actuated joint positions (by name)
        + [f"jvel{i}" for i in range(12)]  # actuated joint velocities (by name)
        + ["dx", "dy"]                     # head->target vector xy (world frame)
        + ["az", "angle"]                  # head orientation (axis z-comp, angle)
        + ["wx", "wy", "wz"]               # head angular velocity
        + ["cos_e", "sin_e"]               # heading error (head-forward vs target)
        + [f"act{i}" for i in range(act_dim)]  # last action (R, omega, theta, delta)
    )

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_log.append(np.asarray(obs, dtype=np.float64))
        done = False
        total_reward = 0
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            obs_log.append(np.asarray(obs, dtype=np.float64))
            total_reward += reward
            done = terminated or truncated
            step += 1
        print(f"Episode {ep+1}: Total reward = {total_reward:.2f}, Steps = {step}")

    # ------------------------------------------------------------------
    # Per-dimension observability report.
    # ------------------------------------------------------------------
    O = np.asarray(obs_log)
    std = O.std(axis=0)
    mean = O.mean(axis=0)

    if len(obs_labels) != O.shape[1]:
        # Guard: obs layout changed but labels didn't. Fall back to indices.
        obs_labels = [f"obs{i}" for i in range(O.shape[1])]

    print(f"\nPer-dimension observation stats over {O.shape[0]} samples "
          f"({n_episodes} episodes):")
    print(f"{'dim':>8s} {'mean':>12s} {'std':>12s}")
    # Sort ascending by std so the most droppable (near-constant) dims are on top.
    for name, m, s in sorted(zip(obs_labels, mean, std), key=lambda t: t[2]):
        flag = "  <- ~constant, droppable" if s < 1e-3 else ""
        print(f"{name:>8s} {m:>12.5f} {s:>12.5f}{flag}")

    env.close()
