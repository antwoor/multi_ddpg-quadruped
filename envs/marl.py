import numpy as np
import torch
from go1_env import *

class ActionSelector:
    def __init__(self, agent_paths, robot, state_size, action_size):
        self.agents = []
        self.robot = robot
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device('cuda')
        
        # Загрузка всех агентов
        for path in agent_paths:
            agent = self.load_agent(path['actor'], path['critic'], state_size=self.state_size, action_size=self.action_size)
            self.agents.append(agent)
        
    def load_agent(self, actor_path, critic_path, state_size, action_size):
        # Ваша реализация создания и загрузки агента
        agent = ddpg_agent(state_size, action_size, random_seed=2)
        agent.actor_local.load_state_dict(torch.load(actor_path, map_location=self.device))
        agent.critic_local.load_state_dict(torch.load(critic_path, map_location=self.device))
        return agent
    
    def get_best_action(self, state):
        
        if isinstance(state, list):
            state = np.array(state)
        elif not isinstance(state, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(state)}")
        
        batch_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Параллельное вычисление для всех агентов
        with torch.no_grad():
            # Генерируем действия и сразу конвертируем в тензоры
            actions = [
                torch.FloatTensor(agent.act(state, add_noise=False)).to(self.device)
                for agent in self.agents
            ]

            # Вычисляем Q-значения
            q_values = [
                agent.critic_local(batch_state, a.unsqueeze(0)) 
                for agent, a in zip(self.agents, actions)
            ]

        # Выбор лучшего действия
        best_idx = torch.argmax(torch.stack(q_values))
        self.last_q_values = [q.item() for q in q_values]
        self.best_agent = best_idx
        return actions[best_idx].squeeze(0).cpu().numpy()

class MultiAgentSystem:
    def __init__(self):
        self.env = Go1Env()
        self.robot = self.env.robot  # Инициализация робота
        agent_paths = [
            {'actor': 'weights/actor_weights_4000.pth', 'critic': 'weights/critic_weights_4000.pth'},
            {'actor': 'weights/actor_weights_2000.pth', 'critic': 'weights/critic_weights_2000.pth'},
            {'actor': 'weights/actor_weights_1000.pth', 'critic': 'weights/critic_weights_1000.pth'},
            {'actor': 'good_weights/actor_weights_4000.pth', 'critic': 'good_weights/critic_weights_4000.pth'},
            {'actor': 'weights/actor_weights_5000.pth', 'critic': 'weights/critic_weights_5000.pth'},
            {'actor': 'good_weights/actor_weights_max_reward.pth', 'critic': 'good_weights/critic_weights_max_reward.pth'}
        ]
        self.selector = ActionSelector(agent_paths, self.robot, state_size=self.env.observation_space.shape[0] , action_size=self.env.action_space.shape[0])
    def run_episode(self, max_steps=1000):
        state = np.concatenate([self.robot.GetTrueObservation(), self.robot.GetFootContacts()])
        for _ in range(max_steps):
            action = self.selector.get_best_action(state)
            self.apply_action(action)
            #print(len(action))
            state = np.concatenate([self.robot.GetTrueObservation(), self.robot.GetFootContacts()])
            
    def apply_action(self, action):
        # Применение действия с проверкой ограничений
        # clipped_action = np.clip(action, 
        #                        self.robot.action_space.low, 
        #                        self.robot.action_space.high)
        #self.robot.ApplyAction(action)
        done = False
        next_state, reward, done, _ = self.env.step(action)
        #self.robot.ReceiveObservation()

if __name__ == "__main__":
    system = MultiAgentSystem()
    
    # Запуск 10 эпизодов с визуализацией
    for ep in range(1000):
        print(f"Starting Episode {ep+1}")
        done = False
        system.run_episode()
        # Логирование метрик
        print(f"Q-values: {system.selector.last_q_values}, Best Agent: {system.selector.best_agent}")
        system.env.reset()