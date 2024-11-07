import os
import inspect
import numpy as np
import math
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
robot_interface_dir = os.path.join(currentdir, "..", "build")
os.sys.path.insert(0, robot_interface_dir)

from robot_interface_a1 import RobotInterface  # type: ignore # pytype: disable=import-error


LOWLEVEL = 0xff

d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5, 
     'RR_0':6, 'RR_1':7, 'RR_2':8, 
     'RL_0':9, 'RL_1':10, 'RL_2':11}

sdk = RobotInterface(LOWLEVEL)
sdk.send_command(np.zeros(60, dtype=np.float32))
motiontime = 0
while True:    
    state = sdk.receive_observation()
    #print(f"RPY: {state.imu.rpy}")
    print(f"FR_0: {state.motorState[d['FR_0']].q}")
    print(f"FR_1: {state.motorState[d['FR_1']].q}")
    print(f"FR_2: {state.motorState[d['FR_2']].q}")
    print(f"FL_0: {state.motorState[d['FL_0']].q}")
    print(f"FL_1: {state.motorState[d['FL_1']].q}")
    print(f"FL_2: {state.motorState[d['FL_2']].q}")
    print(f"FL_2: {state.motorState[d['FL_2']].q}")
    print(f"RR_0: {state.motorState[d['RR_0']].q}")
    print(f"RR_1: {state.motorState[d['RR_1']].q}")
    print(f"RR_2: {state.motorState[d['RR_2']].q}")
    print(f"RL_0: {state.motorState[d['RL_0']].q}")
    print(f"RL_1: {state.motorState[d['RL_1']].q}")
    print(f"RL_2: {state.motorState[d['RL_2']].q}")
    time.sleep(0.06)