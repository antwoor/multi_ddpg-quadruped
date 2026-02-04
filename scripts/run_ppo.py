import os
import sys
import importlib.util

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GO1_ENV_PATH = os.path.join(REPO_ROOT, "envs", "go1_env.py")
spec = importlib.util.spec_from_file_location("go1_env", GO1_ENV_PATH)
go1_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(go1_env)
Go1Env = go1_env.Go1Env


def make_env():
    def _init():
        # Ensure child process can import project modules.
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        return Go1Env(gui=True)

    return _init


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

    num_envs = 2
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=128,
        verbose=1,
        tensorboard_log="./ppo_quadruped/",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000_000 // num_envs,
        save_path="./model_checkpoints/",
        name_prefix="ppo_go1",
    )

    model.learn(total_timesteps=50_000_000, callback=checkpoint_callback)
    model.save("ppo_go1")
    env.close()
