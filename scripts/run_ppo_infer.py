import argparse
import importlib.util
import os
import sys

from stable_baselines3 import PPO
import numpy as np
import types
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GO1_ENV_PATH = os.path.join(REPO_ROOT, "envs", "go1_env.py")
spec = importlib.util.spec_from_file_location("go1_env", GO1_ENV_PATH)
go1_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(go1_env)
Go1Env = go1_env.Go1Env

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="PPO inference with GUI")
    parser.add_argument("--checkpoint", required=True, help="Path to SB3 PPO model .zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    env = Go1Env(gui=True)
    # Compatibility shim for checkpoints saved with NumPy 2.x
    if "numpy._core.numeric" not in sys.modules:
        core_mod = types.ModuleType("numpy._core")
        sys.modules.setdefault("numpy._core", core_mod)
        sys.modules["numpy._core.numeric"] = np.core.numeric
    model = PPO.load(args.checkpoint, env=env)

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            ep_reward = 0.0
            for i in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                ep_reward += reward
                time.sleep(0.033)
                if done:
                    print(f"oops {i}")
                    break
            print(f"Episode {ep}: reward={ep_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
