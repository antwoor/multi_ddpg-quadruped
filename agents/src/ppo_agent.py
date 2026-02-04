import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), log_std_init=-0.5):
        super().__init__()
        h1, h2 = hidden_sizes
        self.actor_fc1 = nn.Linear(state_dim, h1)
        self.actor_fc2 = nn.Linear(h1, h2)
        self.actor_mean = nn.Linear(h2, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        self.critic_fc1 = nn.Linear(state_dim, h1)
        self.critic_fc2 = nn.Linear(h1, h2)
        self.critic_out = nn.Linear(h2, 1)

    def forward(self, state):
        actor = F.relu(self.actor_fc1(state))
        actor = F.relu(self.actor_fc2(actor))
        mean = self.actor_mean(actor)

        critic = F.relu(self.critic_fc1(state))
        critic = F.relu(self.critic_fc2(critic))
        value = self.critic_out(critic)
        return mean, value

    def get_dist(self, mean):
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        lr=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        rollout_steps=2048,
        minibatch_size=64,
        update_epochs=10,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.reset_buffer()

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def _tanh_correction(self, pre_tanh):
        # Avoid log(0) from numerical issues.
        return torch.log(1.0 - torch.tanh(pre_tanh) ** 2 + 1e-6)

    def act(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, value = self.model(state_t)
        dist = self.model.get_dist(mean)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = dist.rsample()

        action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh).sum(-1, keepdim=True)
        log_prob -= self._tanh_correction(pre_tanh).sum(-1, keepdim=True)

        return (
            action.squeeze(0).detach().cpu().numpy(),
            log_prob.squeeze(0).detach().cpu().numpy(),
            value.squeeze(0).detach().cpu().numpy(),
        )

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def should_update(self):
        return len(self.states) >= self.rollout_steps

    def _compute_gae(self, last_value):
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        gae = 0.0
        values = np.append(self.values, last_value)
        for step in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[step]
                + self.gamma * values[step + 1] * (1 - self.dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, last_value=0.0):
        if not self.states:
            return

        advantages, returns = self._compute_gae(last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(device)

        data_size = states.size(0)
        for _ in range(self.update_epochs):
            indices = np.random.permutation(data_size)
            for start in range(0, data_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                mean, value = self.model(batch_states)
                dist = self.model.get_dist(mean)

                pre_tanh = torch.atanh(torch.clamp(batch_actions, -0.999, 0.999))
                new_log_probs = dist.log_prob(pre_tanh).sum(-1, keepdim=True)
                new_log_probs -= self._tanh_correction(pre_tanh).sum(-1, keepdim=True)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(value, batch_returns)
                entropy = dist.entropy().sum(-1, keepdim=True).mean()

                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.reset_buffer()

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}_ppo.pth")

    def load(self, path):
        self.model.load_state_dict(torch.load(f"{path}_ppo.pth", map_location=device))
