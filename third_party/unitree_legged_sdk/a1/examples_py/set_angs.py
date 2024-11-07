import os
import inspect
import numpy as np
import math
import time

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
robot_interface_dir = os.path.join(currentdir, "..", "build")
os.sys.path.insert(0, robot_interface_dir)

from robot_interface_a1 import RobotInterface  # type: ignore # pytype: disable=import-error
from robot_interface_a1 import LowCmd  # type: ignore # pytype: disable=import-error
from robot_interface_a1 import MotorCmd # type: ignore # pytype: disable=import-error

command = np.zeros(60, dtype=np.float32)

def jointLinearInterpolation(initPos, targetPos, rate):

    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5, 
     'RR_0':6, 'RR_1':7, 'RR_2':8, 
     'RL_0':9, 'RL_1':10, 'RL_2':11}

PosStopF  = math.pow(10,9)
VelStopF  = 16000.0
LOWLEVEL = 0xff
sin_mid_q = [0.0, 1.2, -2.0]
dt = 0.002
qInit = [0, 0, 0]
qDes = [0, 0, 0]
sin_count = 0
rate_count = 0
Kp = [0, 0, 0]
Kd = [0, 0, 0]
sdk = RobotInterface(LOWLEVEL)
sdk.send_command(np.zeros(60, dtype=np.float32))
#udp = sdk.UDP()
Tpi = 0
motiontime = 0
cmd = LowCmd()
while True:    
    time.sleep(0.002)
    motiontime += 1
    state = sdk.receive_observation()
    if( motiontime >= 0):
        # first, get record initial position
        if( motiontime >= 0 and motiontime < 10):
            """first, get record initial position"""
            qInit[0] = state.motorState[d['FR_0']].q
            qInit[1] = state.motorState[d['FR_1']].q
            qInit[2] = state.motorState[d['FR_2']].q
        # second, move to the origin point of a sine movement with Kp Kd
        if( motiontime >= 10 and motiontime < 400):
            rate_count += 1
            rate = rate_count/200.0                       # needs count to 200
            Kp = [5, 5, 5]
            Kd = [1, 1, 1]
            # Kp = [20, 20, 20]
            # Kd = [2, 2, 2]
            
            qDes[0] = jointLinearInterpolation(qInit[0], sin_mid_q[0], rate)
            qDes[1] = jointLinearInterpolation(qInit[1], sin_mid_q[1], rate)
            qDes[2] = jointLinearInterpolation(qInit[2], sin_mid_q[2], rate)
        # last, do sine wave
        freq_Hz = 1
        # freq_Hz = 5
        freq_rad = freq_Hz * 2* math.pi
        t = dt*sin_count
        if( motiontime >= 400):
            sin_count += 1
            # sin_joint1 = 0.6 * sin(3*M_PI*sin_count/1000.0)
            # sin_joint2 = -0.9 * sin(3*M_PI*sin_count/1000.0)
            sin_joint1 = 0.6 * math.sin(t*freq_rad)
            sin_joint2 = -0.9 * math.sin(t*freq_rad)
            qDes[0] = sin_mid_q[0]
            qDes[1] = sin_mid_q[1] + sin_joint1
            qDes[2] = sin_mid_q[2] + sin_joint2

    #EXECUTONG MOTION
        command = np.zeros(60, dtype=np.float32)

        command[d['FR_0'] * 5] = qDes[0] # .q
        command[d['FR_0'] * 5 + 2] = 0 # .dq
        command[d['FR_0'] * 5 + 1] = Kp[0] # .Kp
        command[d['FR_0'] * 5 + 3] = Kd[0] # .Kp
        command[d['FR_0'] * 5 + 4] = -0.65 # .tau
        
        command[d['FR_1'] * 5] = qDes[1] # .q
        command[d['FR_1'] * 5 + 2] = 0 # .dq
        command[d['FR_1'] * 5 + 1] = Kp[1] # .Kp
        command[d['FR_1'] * 5 + 3] = Kd[1] # .Kp
        command[d['FR_1'] * 5 + 4] = 0 # .tau

        command[d['FR_2'] * 5] = qDes[2] # .q
        command[d['FR_2'] * 5 + 2] = 0 # .dq
        command[d['FR_2'] * 5 + 1] = Kp[2] # .Kp
        command[d['FR_2'] * 5 + 3] = Kd[2] # .Kp
        command[d['FR_2'] * 5 + 4] = 0 # .tau

        command[d['FL_0'] * 5] = qDes[0] # .q
        command[d['FL_0'] * 5 + 2] = 0 # .dq
        command[d['FL_0'] * 5 + 1] = Kp[0] # .Kp
        command[d['FL_0'] * 5 + 3] = Kd[0] # .Kp
        command[d['FL_0'] * 5 + 4] = -0.65 # .tau
        
        command[d['FL_1'] * 5] = qDes[1] # .q
        command[d['FL_1'] * 5 + 2] = 0 # .dq
        command[d['FL_1'] * 5 + 1] = Kp[1] # .Kp
        command[d['FL_1'] * 5 + 3] = Kd[1] # .Kp
        command[d['FL_1'] * 5 + 4] = 0 # .tau

        command[d['FL_2'] * 5] = -1.2 # .q
        command[d['FL_2'] * 5 + 2] = 0 # .dq
        command[d['FL_2'] * 5 + 1] = Kp[2] # .Kp
        command[d['FL_2'] * 5 + 3] = Kd[2] # .Kp
        command[d['FL_2'] * 5 + 4] = 0 # .tau
    
    print(qDes[0], " ", state.motorState[d['FR_0']].q)
    print(qDes[1], " ", state.motorState[d['FR_1']].q)
    print(qDes[2], " ", state.motorState[d['FR_2']].q)
    sdk.send_command(command)





    """"Должно работать, но не работает"""
        # cmd = LowCmd()
        # cmd.motorCmd[d['FR_0']].q = qDes[0]
        # cmd.motorCmd[d['FR_0']].dq = 0
        # cmd.motorCmd[d['FR_0']].Kp = Kp[0]
        # cmd.motorCmd[d['FR_0']].Kd = Kd[0]
        # cmd.motorCmd[d['FR_0']].tau = -0.65
        # cmd.motorCmd[d['FR_1']].q = qDes[1]
        # cmd.motorCmd[d['FR_1']].dq = 0
        # cmd.motorCmd[d['FR_1']].Kp = Kp[1]
        # cmd.motorCmd[d['FR_1']].Kd = Kd[1]
        # cmd.motorCmd[d['FR_1']].tau = 0.0
        # cmd.motorCmd[d['FR_2']].q =  qDes[2]
        # cmd.motorCmd[d['FR_2']].dq = 0
        # cmd.motorCmd[d['FR_2']].Kp = Kp[2]
        # cmd.motorCmd[d['FR_2']].Kd = Kd[2]
        # cmd.motorCmd[d['FR_2']].tau = 0.0
        # sdk.send_command(cmd)
    """works_herovo"""
    # state = sdk.receive_observation()
    # torque = (0 - state.motorState[d['FR_1']].q)*10.0 + (0 - state.motorState[d['FR_1']].dq)*1.0
    # torque = np.fmin(np.fmax(torque, -5.0), 5.0)
    
    # command[d['FR_1'] * 5] = PosStopF*2
    # command[d['FR_1'] * 5 + 2] = VelStopF
    # command[d['FR_1'] * 5 + 4] = torque
    # sdk.send_command(command)
    # time.sleep(0.06)