import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.src.ppo_agent import PPOAgent
from envs.go1_env import Go1Env


def evaluate(agent, env, episodes, max_steps):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action, _, _ = agent.act(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards)) if rewards else 0.0


def train(args):
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    train_log_dir = os.path.join("runs", f"train_{timestamp}")
    eval_log_dir = os.path.join("runs", f"eval_{timestamp}")
    checkpoint_dir = os.path.join("checkpoints", f"ppo_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = Go1Env(gui=args.gui)
    train_writer = SummaryWriter(log_dir=train_log_dir)
    eval_writer = SummaryWriter(log_dir=eval_log_dir)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

    try:
        for episode in range(1, args.episodes + 1):
            state = env.reset()
            ep_reward = 0.0
            for step in range(args.max_steps):
                action, log_prob, value = agent.act(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)

                agent.store_transition(state, action, log_prob, reward, done, value)
                ep_reward += reward
                state = next_state

                if agent.should_update():
                    if done:
                        last_value = 0.0
                    else:
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0)
                            _, value_t = agent.model(state_t.to(agent.model.actor_fc1.weight.device))
                        last_value = float(value_t.squeeze(0).cpu().numpy())
                    agent.update(last_value=last_value)

                if done:
                    break

            train_writer.add_scalar("train/episode_reward", ep_reward, episode)
            train_writer.add_scalar("train/episode_length", step + 1, episode)

            if episode % args.eval_interval == 0:
                avg_eval_reward = evaluate(agent, env, args.eval_episodes, args.max_steps)
                eval_writer.add_scalar("eval/avg_reward", avg_eval_reward, episode)
                if avg_eval_reward > 0.0:
                    agent.save(os.path.join(checkpoint_dir, f"ppo_ep{episode}"))
    finally:
        train_writer.close()
        eval_writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on Go1Env")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
