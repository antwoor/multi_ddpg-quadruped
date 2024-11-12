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
import matplotlib.pyplot as plt
#learning

#robot_connect(sim)
robot = go1.Go1(pybullet_client =pyb, motor_control_mode=go1.robot_config.MotorControlMode.TORQUE)
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

def reward(v_x, y_error_squared, theta, u_prev):
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
    sum_u_squared = sum(u**2 for u in u_prev)
    
    reward = (
        v_x
        - 40 * y_error_squared
        - 30 * theta**2
        - 0.02 * sum_u_squared
    )
    
    return reward

done = False
agent = ddpg_agent(state_size=np.size(np.array(robot.GetTrueObservation())), action_size=12, random_seed=2)
episodes = 1000
def ddpg(n_episodes=1000, max_t=300, print_every=100):
    done = False
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        pyb.resetBasePositionAndOrientation(robot.quadruped, [0, 0, 0.5], [0, 0, 0, 1])  # устанавливаем позицию и ориентацию
        pyb.resetBaseVelocity(robot.quadruped, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])  # обнуляем скорости
        robot.ResetPose(add_constraint=False)
        #pyb.stepSimulation()
        robot.ReceiveObservation()
        state = np.array(robot.GetTrueObservation())
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(np.array(state))
            action = robot._ClipMotorCommands(
                motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
                motor_commands=action
                )
            robot._StepInternal(action, motor_control_mode=None)
            next_state = robot.GetTrueObservation()
            _reward = reward(
                v_x=robot.GetBaseVelocity()[0], 
                y_error_squared=robot.GetBasePosition()[1], 
                theta=robot.GetBaseRollPitchYaw()[0], 
                u_prev=robot.GetMotorTorques()
            )
            agent.step(state, action, _reward, next_state, done)
            state = next_state
            score += reward(
                v_x=robot.GetBaseVelocity()[0], 
                y_error_squared=robot.GetBasePosition()[1], 
                theta=robot.GetBaseRollPitchYaw()[0], 
                u_prev=robot.GetMotorTorques()
            )
            print("AGENT_STEP")
            if robot.GetBasePosition()[2] < 0.1:  # Если робот упал ниже допустимой высоты
                done = True
                break

        print("EPISODE_END")
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        #torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        #torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores

scores = []

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

if __name__ == '__main__':
    print(robot.GetTrueBaseRollPitchYawRate())
    np.array(robot.GetTrueObservation())
    scores = ddpg()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()