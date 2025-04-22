import os
import inspect
import gym
from gym import spaces
import numpy as np
import pybullet as pyb
import pybullet_data as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
print(currentdir)
print(parentdir)
from motion_imitation.robots import go1
from motion_imitation.envs import env_builder
from motion_imitation.robots import robot_config
from agents.src.ddpg_agent import Agent as ddpg_agent
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
#bullet
#pyb.connect(pyb.GUI)
#pyb.setAdditionalSearchPath(pd.getDataPath())
#pyb.setGravity(0,0,-9.8)
#pyb.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
#pyb.setRealTimeSimulation(0)  # Отключаем реальное время
import time
MAX_TORQUE = np.array([28.7, 28.7, 40] * 4)

#robot = go1.Go1(pybullet_client =pyb, motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
#                self_collision_enabled=True, motor_torque_limits=MAX_TORQUE)
#robot.ReceiveObservation()
'''
for episode in range(1,episodes+1):
    robot.ReceiveObservation()
    state = robot.GetTrueMotorAngles()
    print("STATE",state)
    action = agent.act(state)
    print("ACTION",action)
    pyb.stepSimulation()
    next_state = robot.GetTrueMotorAngles()
    print("NEXT_STATE", next_state)
    reward = np.random.uniform(low=-np.pi, high=np.pi)
    agent.step(state, action, reward, next_state, False)
    print("LOL")
    robot.Step(action)
    print("KEK")
    #agent.learn() '''

def reward(v_x, theta, u_prev, Ts, Tf, done, contacts):
    """
    Вычисляет значение функции вознаграждения.

    Параметры:
    v_x : float - скорость центра масс туловища в направлении x
    y : float - текущая высота центра масс туловища
    theta : float - угол наклона туловища (в радианах)
    u_prev : list - список значений действий для суставов из предыдущего временного шага
    Ts : float - текущее время симуляции
    Tf : float - общее время симуляции
    done : bool - флаг завершения эпизода (робот упал)

    Возвращает:
    float - значение вознаграждения
    """
    #штраф за 4 ноги в воздухе
    contact_penalty = -50 * (np.prod(contacts))
    # Штраф за наклон туловища
    tilt_penalty = -20 * (theta ** 2)

    # Штраф за резкие изменения в действиях (стабильность управления)
    action_penalty = -0.01 * np.sum(np.square(u_prev))

    # Награда за скорость движения вперёд
    velocity_reward = 10 * v_x

    # Штраф за завершение эпизода (падение робота)
    if done:
        termination_penalty = -1000
    else:
        termination_penalty = 0

    # Временной бонус (поощрение за продвижение во времени)
    time_bonus = 25 * (Ts / Tf)

    # Итоговое вознаграждение
    total_reward = (
        velocity_reward
        + contact_penalty
        + tilt_penalty
        + action_penalty
        + termination_penalty
        + time_bonus
    )

    return total_reward

# Определяем класс среды
class Go1Env(gym.Env):
    def __init__(self, pyb_client = None, gui = True):
        super(Go1Env, self).__init__()
        
        # Инициализация PyBullet
        if pyb_client == None:
            if gui:
                pyb.connect(pyb.GUI)
            else:
                pyb.connect(pyb.DIRECT)
            pyb.setAdditionalSearchPath(pd.getDataPath())
            pyb.setGravity(0, 0, -9.8)
            pyb.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
            pyb.setRealTimeSimulation(0)
            self.pyb_client = pyb
        else:
            self.pyb_client = pyb_client
        
        # Инициализация робота
        self.MAX_TORQUE = np.array([28.7, 28.7, 40] * 4)
        self.robot = go1.Go1(pybullet_client=self.pyb_client, motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
                             self_collision_enabled=True, motor_torque_limits=self.MAX_TORQUE)
        self.robot.ReceiveObservation()
        
        # Определение пространства действий и состояний
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.robot.GetTrueObservation()+self.robot.GetFootContacts()),), dtype=np.float32)
        
        # Инициализация TensorBoard
        self.writer = SummaryWriter()
        self.episode_reward = 0
        self.episode_count = 0

    def reset(self):
        # Сброс состояния робота
        self.pyb_client.resetBasePositionAndOrientation(self.robot.quadruped, [0, 0, 0.25], [0, 0, 0, 1])
        self.pyb_client.resetBaseVelocity(self.robot.quadruped, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        self.robot.ResetPose(add_constraint=False)
        self.robot.ReceiveObservation()
        
        # Возвращаем начальное состояние
        state = np.concatenate([self.robot.GetTrueObservation(), self.robot.GetFootContacts()])
        self.episode_reward = 0
        return state

    def step(self, action):
        # Применяем действие
        action = self.robot._ClipMotorCommands(
            motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
            motor_commands=10 * action
        )
        self.robot.ApplyAction(action)
        self.pyb_client.stepSimulation()
        self.robot.ReceiveObservation()
        #self.robot.Step(action)
        # Получаем следующее состояние
        next_state = np.concatenate([self.robot.GetTrueObservation(), self.robot.GetFootContacts()])

        # Вычисляем награду
        _reward = self.reward(
            v_x=self.robot.GetBaseVelocity()[0],
            roll=self.robot.GetBaseRollPitchYaw()[0],
            pitch=self.robot.GetBaseRollPitchYaw()[1],
            yaw=self.robot.GetBaseRollPitchYaw()[2],
            y=self.robot.GetBasePosition()[2],
            Ts=0,
            Tf=1000,
            contacts=self.robot.GetFootContacts(),
            joint_torques=self.robot.GetMotorTorques()
        )
        self.episode_reward += _reward
        
        # Проверяем завершение эпизода
        done = self.robot.GetBasePosition()[2] < 0.18 or np.sum(self.robot.GetBaseRollPitchYaw()**2) >= 0.73
        
        # Логируем награду в TensorBoard
        if done:
            self.writer.add_scalar('Reward/Episode', self.episode_reward, self.episode_count)
            self.episode_count += 1
        
        return next_state, _reward, done, {}

    def reward(self, v_x, y, roll, pitch, yaw, Ts, Tf, contacts, joint_torques=None):
        """
        Улучшенная функция вознаграждения для шагающего робота
        """
        target_height = 0.25  # Целевая высота центра масс
        #print("HIGH IS", y)
        # Базовые компоненты
        velocity_reward = 5.0 * np.clip(v_x, -2.0, 2.0)  # Нелинейное поощрение скорости
        height_penalty = -80.0 * ((y - target_height) ** 2)

        # Стабильность и ориентация
        orientation_penalty = -15.0 * (roll**2 + pitch**2) - 5.0 * yaw**2
        angular_velocity_penalty = -0.1 * np.linalg.norm(self.robot.GetBaseRollPitchYawRate())**2

        # Контакты с поверхностью
        contact_bonus = 2.0 * np.sum(contacts)  # Поощрение за большее число контактов
        flight_penalty = -50.0 if np.sum(contacts) == 0 else 0.0  # Полный полёт

        # Энергоэффективность (если доступны моменты)
        energy_penalty = 0.0
        if joint_torques is not None:
            energy_penalty = -0.01 * np.sum(np.square(joint_torques))

        # Плавность управления
        #if hasattr(self, 'last_action'):
        #    action_smoothness = -0.1 * np.sum(np.square(self.last_action - self.current_action))

        # Временные компоненты
        time_progress = 25.0 * (Ts / Tf)
        survival_bonus = 10.0  # Поощрение за каждый шаг

        # Терминальные условия
        termination_penalty = 0.0
        #if done:
        #    if y < 0.15:  # Падение
        #        termination_penalty = -1000.0
        #    elif Ts >= Tf:  # Успешное завершение
        #        termination_penalty = 1000.0

        # Собираем все компоненты
        total_reward = (
            velocity_reward
            + height_penalty
            + orientation_penalty
            + angular_velocity_penalty
            + contact_bonus
            + flight_penalty
            + energy_penalty
            + time_progress
            + survival_bonus
            + termination_penalty
        )

        # Логирование компонентов (для отладки)
        # if self.writer:
        #     components = {
        #         'velocity_reward': velocity_reward,
        #         'height_penalty': height_penalty,
        #         'orientation_penalty': orientation_penalty,
        #         'contact_bonus': contact_bonus,
        #         'energy_penalty': energy_penalty,
        #         'time_progress': time_progress
        #     }
        #     for name, value in components.items():
        #         self.writer.add_scalar(f'reward/{name}', value, self.step_count)

        return total_reward

    def render(self, mode='human'):
        pass  # PyBullet уже визуализирует среду

    def close(self):
        self.pyb_client.disconnect()
        self.writer.close()

def train(n_episodes=1000, max_t=1000, print_every=100, prefill_steps=5000, robot = go1.Go1, pyb = pyb):
    done = False
    scores_deque = deque(maxlen=print_every)
    scores = []
    experience_buffer = deque(maxlen=prefill_steps)

    # Initialize agent
    state_size = np.size(np.array(robot.GetTrueObservation() + robot.GetFootContacts()))
    agent = ddpg_agent(state_size=state_size, action_size=12, random_seed=2)

    # Step 1: Prefill experience buffer with random actions
    print("Filling experience buffer with random actions...")
    for _ in range(prefill_steps):
        robot.ReceiveObservation()
        state = robot.GetTrueObservation() + robot.GetFootContacts()
        action = np.random.uniform(low=-1, high=1, size=12)  # Random action
        print("NON_CLIPPED ACTION",action)
        action = robot._ClipMotorCommands(
            motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
            motor_commands=10*action,
        )
        print("CLIPPED ACTION",action)
        robot.ApplyAction(action)
        pyb.stepSimulation()
        robot.ReceiveObservation()
        next_state = robot.GetTrueObservation() + robot.GetFootContacts()
        _reward = reward(
            v_x=robot.GetBaseVelocity()[0],
            y=robot.GetBasePosition()[2],
            roll=robot.GetBaseRollPitchYaw()[0],
            pitch=robot.GetBaseRollPitchYaw()[1],
            yaw=robot.GetBaseRollPitchYaw()[2],
            Ts=0,
            Tf=max_t,
            target_height=robot.GetDefaultInitPosition[2]
        )
        done = robot.GetBasePosition()[2] < 0.1 or np.sum(robot.GetBaseRollPitchYaw()) >= 0.73
        experience_buffer.append((state, action, _reward, next_state, done))
        if done:
            pyb.resetBasePositionAndOrientation(robot.quadruped, [0, 0, 0.25], [0, 0, 0, 1])
            pyb.resetBaseVelocity(robot.quadruped, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            robot.ResetPose(add_constraint=False)

    # Step 2: Train agent using DDPG
    print("Starting DDPG training...")

    done = False
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        done = False
        pyb.resetBasePositionAndOrientation(robot.quadruped, [0, 0, 0.25], [0, 0, 0, 1])
        pyb.resetBaseVelocity(robot.quadruped, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        robot.ResetPose(add_constraint=False)
        #pyb.stepSimulation()  # Выполняем шаг симуляции
        robot.ReceiveObservation()
        state = robot.GetTrueObservation() + robot.GetFootContacts()
        #print("the size of state is ", np.size(state))
        #print("the size of true obs is ", np.size(robot.GetTrueObservation()))
        agent.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(np.array(state), add_noise=True)
            action = robot._ClipMotorCommands(
                motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
                motor_commands=10*action
            )
            robot.ApplyAction(action)
            #print(robot.GetBasePosition()[2])
            pyb.stepSimulation()
            robot.ReceiveObservation()

            next_state = robot.GetTrueObservation() + robot.GetFootContacts()
            _reward = reward(
                v_x=robot.GetBaseVelocity()[0], 
                y=robot.GetBasePosition()[2], 
                theta=robot.GetBaseRollPitchYaw()[1], 
                Ts=t,
                Tf=max_t,
                contacts=robot.GetFootContacts()
            )
            if robot.GetBasePosition()[2] < 0.18 or np.sum(robot.GetBaseRollPitchYaw())>=0.73:
                done = True
                print("TERMINATED", robot.GetBasePosition()[2],"  RPY", np.abs(np.sum(robot.GetBaseRollPitchYaw())))
            agent.step(state=np.array(state), action=action, reward=_reward, next_state=np.array(next_state), done=done)
            if done:
                break
            state = next_state
            score += _reward
            #print("KEK")

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 500:
                torch.save(agent.actor_local.state_dict(), 'actor_weights_{}.pth'.format(i_episode))
                torch.save(agent.critic_local.state_dict(), 'critic_weights_{}.pth'.format(i_episode)) 


    return scores

scores = []

if __name__ == '__main__':
    env = Go1Env()
    agent = ddpg_agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=2)

    max_reward = -float('inf')
    weights_dir = 'weights'
    # Загрузка последних предобученных весов (если они существуют)
    actor_weights_path = os.path.join(weights_dir, 'actor_weights_max_reward.pth')
    critic_weights_path = os.path.join(weights_dir, 'critic_weights_max_reward.pth')

    if os.path.exists(actor_weights_path) and os.path.exists(critic_weights_path):
        agent.actor_local.load_state_dict(torch.load(actor_weights_path))
        agent.critic_local.load_state_dict(torch.load(critic_weights_path))
        print("Loaded pre-trained weights.")
    else:
        print("No pre-trained weights found. Starting from scratch.")

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    # Обучение или эвалюация
    for episode in range(10000):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, add_noise=True)  # Включить шум для обучения
            next_state, reward, done, _ = env.step(action)
            #agent.step(state, action, reward, next_state, done)
            # if agent.last_tgQ is not 0 and done:
            #     env.writer.add_scalar('target_Q', agent.last_tgQ.detach().cpu().numpy().mean(), env.episode_count-1)
            #     env.writer.add_scalar('actual_Q', agent.last_actQ.detach().cpu().numpy().mean(), env.episode_count-1)
            #     env.writer.add_scalar('excpected_Q', agent.last_expQ.detach().cpu().numpy().mean(), env.episode_count-1)
            time.sleep(0.01)
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        #print("tgQ", agent.last_tgQ)
        #print("actQ", agent.last_actQ)
        #print("expQ", agent.last_expQ)
        # Сохранение весов, если награда больше 250 и больше предыдущей максимальной
        # if total_reward > 250 and total_reward > max_reward:
        #     max_reward = total_reward  # Обновляем максимальную награду
        #     actor_path = os.path.join(weights_dir, f'actor_weights_max_reward.pth')
        #     critic_path = os.path.join(weights_dir, f'critic_weights_max_reward.pth')
        #     torch.save(agent.actor_local.state_dict(), actor_path)
        #     torch.save(agent.critic_local.state_dict(), critic_path)
        #     print(f"New max reward: {max_reward}. Weights saved to {weights_dir}.")
        # elif episode % 1000 == 0:
        #     actor_path = os.path.join(weights_dir, f'actor_weights_{episode}.pth')
        #     critic_path = os.path.join(weights_dir, f'critic_weights_{episode}.pth')
        #     torch.save(agent.actor_local.state_dict(), actor_path)
        #     torch.save(agent.critic_local.state_dict(), critic_path)
        #     print(f"New max reward: {max_reward}. Weights saved to {weights_dir}.")
    env.close()