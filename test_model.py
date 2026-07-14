import numpy as np
from stable_baselines3 import SAC
from env_snake import SnakeEnv

if __name__ == "__main__":
    # Carica il modello
    model = SAC.load("models/sac_snake_final.zip")

    # Crea l'ambiente in modalità rendering (opzionale)
    env = SnakeEnv(render_mode="human")
    n_episodes = 5

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