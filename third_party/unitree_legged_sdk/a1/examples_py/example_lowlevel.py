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

PosStopF  = math.pow(10,9)
VelStopF  = 16000.0
sin_count = 0
Tpi = 0
motiontime = 0
while True:
    time.sleep(0.002)
    motiontime += 1

    freq_Hz = 2
    freq_rad = freq_Hz * 2* math.pi
    command = np.zeros(60, dtype=np.float32)
    
    state = sdk.receive_observation()
    
    print(f"quat: {state.imu.quaternion}")

    if motiontime >= 500:
        sin_count += 1
        torque = (0 - state.motorState[d['FR_1']].q)*10.0 + (0 - state.motorState[d['FR_1']].dq)*1.0
        torque = np.fmin(np.fmax(torque, -5.0), 5.0)
        
        command[d['FR_1'] * 5] = PosStopF
        command[d['FR_1'] * 5 + 2] = VelStopF
        command[d['FR_1'] * 5 + 4] = torque

    sdk.send_command(command)
    