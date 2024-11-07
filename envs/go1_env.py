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

pyb.connect(pyb.GUI)
pyb.setAdditionalSearchPath(pd.getDataPath())
pyb.setGravity(0,0,-9.8)
pyb.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
agent = ddpg_agent(state_size=12, action_size=12, random_seed=2)
robot = go1.Go1(pybullet_client =pyb, motor_control_mode=go1.robot_config.MotorControlMode.TORQUE)
robot.ReceiveObservation()
while True:
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
    #agent.learn()