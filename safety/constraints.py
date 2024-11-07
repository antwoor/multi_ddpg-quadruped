import os
import inspect
import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
robot_interface_dir = os.path.join(currentdir, "..")
os.sys.path.insert(0, robot_interface_dir)

from absl import flags
from motion_imitation.robots import a1
import pybullet
import pybullet_data
from pybullet_utils import bullet_client

BAD_POSITIONS = np.array([[3,3,3,3,3,3,3,3,3,3,3,3],
                         [0,0,0,0,0,0,0,0,0,0,0,0]
                         ])
BAD_TORQUES = np.array([
     [0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4],
     [0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3]
]) 
JOINT_DICT = {
    "FR_hip_motor": 0, "FR_upper_joint": 1, "FR_lower_joint": 2,
    "FL_hip_motor": 3, "FL_upper_joint": 4, "FL_lower_joint": 5,
    "RR_hip_motor": 6, "RR_upper_joint": 7, "RR_lower_joint": 8,
    "RL_hip_motor": 9, "RL_upper_joint": 10, "RL_lower_joint": 11,
    }

flags.DEFINE_multi_string("joint", JOINT_DICT.keys(),
                          "Names of joints to measure.")
FLAGS = flags.FLAGS

p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
a1 = a1_robot.A1Robot(pybullet_client=p, action_repeat=1)

def print_angles():
     a1.ReceiveObservation()
     angs = a1.GetTrueMotorAngles()
     for joint in JOINT_DICT:
          print(joint, angs[JOINT_DICT[joint]])

def check_collision():
     a1.ReceiveObservation()
     angs = a1.GetTrueMotorAngles()
     for collision in BAD_POSITIONS:
          if np.array_equal(angs, collision):
               print("collision detected")
          else:
               print("collision not detected")



if __name__ == "__main__":
     print_angles()
     #check_collision()
