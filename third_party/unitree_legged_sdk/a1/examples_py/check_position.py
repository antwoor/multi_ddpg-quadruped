import os
import inspect
import numpy as np
import math
import time
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
robot_interface_dir = os.path.join(currentdir, "..", "build")
os.sys.path.insert(0, robot_interface_dir)

from robot_interface_a1 import RobotInterface  # type: ignore # pytype: disable=import-error

command = np.zeros(60, dtype=np.float32)
LOWLEVEL = 0xff
sdk = RobotInterface(LOWLEVEL)
sdk.send_command(np.zeros(60, dtype=np.float32))

def get_motor_name(d, key):
    try:
        if key in d:
            return key
        else:
            raise KeyError
    except KeyError:
        return f"error motor '{key}'is not known"

LOWLEVEL = 0xff
Kp = 0 
Kd = 0

d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5, 
     'RR_0':6, 'RR_1':7, 'RR_2':8, 
     'RL_0':9, 'RL_1':10, 'RL_2':11}

command = np.zeros(60, dtype=np.float32)

if __name__ == '__main__':
    motor = get_motor_name(d, sys.argv[1])
    print(d[motor])
    command[d[motor] * 5] = float(sys.argv[2])* (math.pi / 180) # .q
    command[d[motor] * 5 + 2] = 0 # .dq
    command[d[motor] * 5 + 1] = Kp # .Kp
    command[d[motor] * 5 + 3] = Kd # .Kp
    command[d[motor] * 5 + 4] = 0 # .tau
    while True:
        state = sdk.receive_observation()
        sdk.send_command(command)
        print(command[d[motor] * 5], " ", state.motorState[d[motor]].q)