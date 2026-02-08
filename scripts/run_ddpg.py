import argparse
import importlib.util
import os
import sys

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GO1_ENV_PATH = os.path.join(REPO_ROOT, "envs", "go1_env.py")
spec = importlib.util.spec_from_file_location("go1_env", GO1_ENV_PATH)
go1_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(go1_env)
Go1Env = go1_env.Go1Env


def make_env():
    def _init():
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        return Go1Env(gui=False)

    return _init


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

    parser = argparse.ArgumentParser(description="Train DDPG on Go1Env")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to SB3 DDPG .zip checkpoint")
    args = parser.parse_args()

    num_envs = 2
    env = SubprocVecEnv([make_env() for _ in range(num_envs)])
    env = VecMonitor(env)

    if args.checkpoint:
        model = DDPG.load(args.checkpoint, env=env, tensorboard_log="./ddpg_quadruped/")
    else:
        model = DDPG(
            "MlpPolicy",
            env,
            learning_rate=1e-3,
            buffer_size=1_000_000,
            batch_size=256,
            verbose=1,
            tensorboard_log="./ddpg_quadruped/",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000_000 // num_envs,
        save_path="./model_checkpoints_ddpg/",
        name_prefix="ddpg_go1",
    )

    model.learn(
        total_timesteps=100_000_000,
        callback=checkpoint_callback,
        reset_num_timesteps=not bool(args.checkpoint),
    )
    model.save("ddpg_go1")
    env.close()
