import numpy as np
import torch
from go1_env import *

class ActionSelector:
    def __init__(self, agent_paths, robot, state_size, action_size):
        self.agents = []
        self.robot = robot
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device('cpu')
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
        batch_state = torch.FloatTensor(state).unsqueeze(0)
        
        # Параллельное вычисление для всех агентов
#TODO
        with torch.no_grad():
            actions = [agent.actor_local(batch_state) for agent in self.agents]
            q_values = [agent.critic_local(batch_state, a) for agent, a in zip(self.agents, actions)]
            
        # Выбор лучшего действия
        best_idx = torch.argmax(torch.stack(q_values))
        self.last_q_values = [q.item() for q in q_values]
        
        return actions[best_idx].squeeze(0).numpy()
    
    def get_metrics(self):
        return {
            'q_values': self.last_q_values,
            'best_agent': np.argmax(self.last_q_values)
        }

class MultiAgentSystem:
    def __init__(self):
        self.env = Go1Env()
        self.robot = self.env.robot  # Инициализация робота
        agent_paths = [
            {'actor': 'weights/actor_weights_4000.pth', 'critic': 'weights/critic_weights_4000.pth'},
            {'actor': 'weights/actor_weights_2000.pth', 'critic': 'weights/critic_weights_2000.pth'},
            {'actor': 'weights/actor_weights_1000.pth', 'critic': 'weights/critic_weights_1000.pth'}
        ]
        self.selector = ActionSelector(agent_paths, self.robot, state_size=self.env.observation_space.shape[0] , action_size=self.env.action_space.shape[0])
        
    def run_episode(self, max_steps=1000):
        state = self.robot.GetTrueObservation()
        for _ in range(max_steps):
            action = self.selector.get_best_action(state)
            self.apply_action(action)
            state = self.robot.GetTrueObservation()
            
    def apply_action(self, action):
        # Применение действия с проверкой ограничений
        clipped_action = np.clip(action, 
                               self.robot.action_space.low, 
                               self.robot.action_space.high)
        self.robot.Step(clipped_action)

if __name__ == "__main__":
    system = MultiAgentSystem()
    
    # Запуск 10 эпизодов с визуализацией
    for ep in range(10):
        print(f"Starting Episode {ep+1}")
        pyb.connect(pyb.DIRECT)  # Визуальный режим
        system.run_episode()
        pyb.disconnect()
        
        # Логирование метрик
        metrics = system.selector.get_metrics()
        print(f"Q-values: {metrics['q_values']}, Best Agent: {metrics['best_agent']}")