import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
print(currentdir)
print(parentdir)
from motion_imitation.robots import go1

import pybullet as pyb
pyb.connect(pyb.GUI)
robot = go1.Go1(pybullet_client =pyb)