import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Сеть Актор
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

# Сеть Критик
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Объединяем состояние и действие
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Выходное значение — Q(s, a)
# Агент DDPG
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99  # Дисконтирование
        self.tau = 0.005   # Коэффициент мягкого обновления

        # Инициализация сетей актора и критика
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)

        # Копирование весов в целевые сети
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Память для опыта
        self.replay_buffer = []

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().cpu().numpy()[0]

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
    
        # Преобразуем память в отдельные массивы
        batch = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
    
        for i in batch:
            state, action, reward, next_state, done = self.replay_buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
    
        # Преобразуем в тензоры
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
    
        # Обновление критика
        next_action = self.actor_target(next_states)
        target_q_value = self.critic_target(next_states, next_action)
        target_q_value = rewards + (1 - dones) * self.gamma * target_q_value
        current_q_value = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_value, target_q_value.detach())
    
        # Оптимизация критика
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
        # Обновление актора
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
        # Мягкое обновление целевых сетей
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    # Сохранение модели
    def save(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    # Загрузка модели
    def load(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))

if __name__ == '__main__':
    # Основные настройки
    if torch.cuda.is_available():
        print("ESHKEREEEE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = 100  # Размер входного вектора
    action_dim = 60  # Размер выходного вектора

    # Инициализация агента
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim)

    # Пример обучения агента
    for episode in range(1000):
        state = np.random.randn(state_dim)  # Замена на реальное начальное состояние среды
        for t in range(200):
            action = agent.select_action(state)
            next_state = np.random.randn(state_dim)  # Замена на реальный переход
            reward = np.random.randn()  # Замена на реальное вознаграждение
            done = np.random.choice([False, True], p=[0.99, 0.01])  # Пример случайного завершения

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            print(reward)
            if done:
                break
