#!/usr/bin/python

import sys
import time
import math
#import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

# low cmd
TARGET_PORT = 8007
#LOCAL_PORT = 8082
LOCAL_PORT = 8080
TARGET_IP = "192.168.123.10"   # target IP address

LOW_CMD_LENGTH = 610
LOW_STATE_LENGTH = 771


if __name__ == '__main__':

    d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    HIGHLEVEL = 0x00
    LOWLEVEL  = 0xff
    dt = 0.002
    sin_count = 0

    # udp = sdk.UDP(8080, "192.168.123.10", 8007, 614, 807, False, sdk.RecvEnum.nonBlock)
    udp = sdk.UDP(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH, -1)
    safe = sdk.Safety(sdk.LeggedType.Aliengo)
    
    cmd = sdk.LowCmd()
    state = sdk.LowState()
    udp.InitCmdData(cmd)
    cmd.levelFlag = LOWLEVEL

    Tpi = 0
    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime += 1

        cmd.motorCmd[d['FR_0']].q = PosStopF
        cmd.motorCmd[d['FR_0']].dq = VelStopF
        cmd.motorCmd[d['FR_0']].Kp = 0
        cmd.motorCmd[d['FR_0']].Kd = 0
        cmd.motorCmd[d['FR_0']].tau = -1.6

        # freq_Hz = 1
        #freq_Hz = 2
        # freq_Hz = 5;
        #freq_rad = freq_Hz * 2* math.pi
        # t = dt*sin_count

        # print(motiontime)
        # print(state.imu.rpy[0])
        
        
        udp.Recv()
        udp.GetRecv(state)

        if( motiontime >= 500):
            #sin_count += 1
            torque = (0 - state.motorState[d['FR_1']].q)*10.0 + (0 - state.motorState[d['FR_1']].dq)*1.0
            # torque = (0 - state.motorState[d['FR_1']].q)*20.0 + (0 - state.motorState[d['FR_1']].dq)*2.0
            # torque = 2 * sin(t*freq_rad)
            #torque = np.fmin(np.fmax(torque, -5.0), 5.0)
            if torque > 5.0:
                torque = 5.0
            elif torque < -5.0:
                torque = -5.0
            # torque = np.fmin(np.fmax(torque, -15.0), 15.0)


            cmd.motorCmd[d['FR_1']].q = PosStopF
            cmd.motorCmd[d['FR_1']].dq = VelStopF
            cmd.motorCmd[d['FR_1']].Kp = 0
            cmd.motorCmd[d['FR_1']].Kd = 0
            cmd.motorCmd[d['FR_1']].tau = torque
        
        # if(motiontime > 10):
        #     safe.PowerProtect(cmd, state, 1)

        udp.SetSend(cmd)
        udp.Send()
