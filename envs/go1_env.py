import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
print(currentdir)
print(parentdir)
from motion_imitation.robots import go1
from agents.src.ddpg_agent import Agent as ddpg_agent
import pybullet as pyb # type: ignore 
import pybullet_data as pd
import numpy as np
from collections import deque
import torch
#bullet
pyb.connect(pyb.GUI)
pyb.setAdditionalSearchPath(pd.getDataPath())
pyb.setGravity(0,0,-9.8)
pyb.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
pyb.setRealTimeSimulation(0)  # Отключаем реальное время
import time
MAX_TORQUE = np.array([28.7, 28.7, 40] * 4)

robot = go1.Go1(pybullet_client =pyb, motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
                self_collision_enabled=True, motor_torque_limits=MAX_TORQUE)
robot.ReceiveObservation()
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

def reward(v_x, y, theta, Ts, Tf):
    """
    Вычисляет значение функции вознаграждения.
    
    Параметры:
    v_x : float - скорость центра масс туловища в направлении x
    Ts : float - время выборки
    Tf : float - время окончания моделирования
    y_error_squared : float - квадрат ошибки измерения высоты центра масс туловища
    theta : float - угол наклона туловища
    u_prev : list - список значений действий для суставов из предыдущего временного шага
    
    Возвращает:
    float - значение вознаграждения
    """
    
    reward = (
        v_x
        - 50 * ((0.25-y)**2)
        - 20 * theta**2
        +25*(Ts/Tf)
    )
    #print(100*v_x)
    
    return reward

done = False
agent = ddpg_agent(state_size=np.size(np.array(robot.GetTrueObservation()+robot.GetFootContacts())), action_size=12, random_seed=2)
episodes = 10000
def ddpg(n_episodes=1000, max_t=1000, print_every=100, prefill_steps=5000):
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
            theta=robot.GetBaseRollPitchYaw()[1],
            Ts=0,
            Tf=max_t,
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
                Tf=max_t
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
            if np.mean(scores_deque) >= 1000:
                torch.save(agent.actor_local.state_dict(), 'actor_weights_{}.pth'.format(i_episode))
                torch.save(agent.critic_local.state_dict(), 'critic_weights_{}.pth'.format(i_episode)) 


    return scores

scores = []

if __name__ == '__main__':
    print(robot.GetTrueBaseRollPitchYaw()[1])
    np.array(robot.GetTrueObservation())
    scores = ddpg(n_episodes=10000, prefill_steps=10000)