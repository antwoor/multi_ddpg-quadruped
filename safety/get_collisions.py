import os
import inspect
import numpy as np
import math
import time
import re
import subprocess
import pinocchio as pin

LOWLEVEL = 0xff

d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
     'FL_0':3, 'FL_1':4, 'FL_2':5, 
     'RR_0':6, 'RR_1':7, 'RR_2':8, 
     'RL_0':9, 'RL_1':10, 'RL_2':11} #TODO 13 а не 11, найти реальные углы

dt = 1./240.  # шаг симуляции

pair_FL = pin.CollisionPair(8, 10) #TODO: MULtIPLYs
pair_RR = pin.CollisionPair(18, 20) #TODO: MULtIPLYs
pair_DIAG = pin.CollisionPair(10, 20) #TODO: MULtIPYs
bad_pair1 = pin.CollisionPair(2, 3)
bad_pair2 = pin.CollisionPair(3, 4)
bad_pair3 = pin.CollisionPair(7, 8)
bad_pair4 = pin.CollisionPair(8, 9)
bad_pair5 = pin.CollisionPair(12, 13)
bad_pair6 = pin.CollisionPair(13, 14)
bad_pair7 = pin.CollisionPair(17, 18)
bad_pair8 = pin.CollisionPair(18, 19)
pairs = [pair_FL, pair_RR, pair_DIAG]
bad_pairs = [bad_pair1, bad_pair2, bad_pair3, bad_pair4, bad_pair5, bad_pair6, 
             bad_pair7, bad_pair8]
HIP_NAME_PATTERN = re.compile(r"\w+_hip_\w+")
UPPER_NAME_PATTERN = re.compile(r"\w+_upper_\w+")
LOWER_NAME_PATTERN = re.compile(r"\w+_lower_\w+")
TOE_NAME_PATTERN = re.compile(r"\w+_toe\d*")
IMU_NAME_PATTERN = re.compile(r"imu\d*")

_hip_link_ids = [-1]
_leg_link_ids = []
_motor_link_ids = []
_lower_link_ids = []
_foot_link_ids = []
_imu_link_ids = []


d_BULLET = {'FR_0':1, 'FR_1':3, 'FR_2':4,
     'FL_0':6, 'FL_1':8, 'FL_2':9, 
     'RR_0':11, 'RR_1':13, 'RR_2':14, 
     'RL_0':16, 'RL_1':18, 'RL_2':19}

'''d_PINOCCHIO = {'FR_0':, 'FR_1':16, 'FR_2':, #F_thigh_shoulder = F_0
     'FL_0':6, 'FL_1':, 'FL_2':, #
     'RR_0':, 'RR_1':, 'RR_2':, # 
     'RL_0':, 'RL_1':, 'RL_2': #
}'''

def _BuildUrdfIds(quadruped):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = pyb.getNumJoints(quadruped)

    _hip_link_ids = [-1]
    _leg_link_ids = []
    _motor_link_ids = []
    _lower_link_ids = []
    _foot_link_ids = []
    _imu_link_ids = []
    _joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = pyb.getJointInfo(quadruped, i)
      _joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    for i in range(num_joints):
      joint_info = pyb.getJointInfo(quadruped, i)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = _joint_name_to_id[joint_name]
      if HIP_NAME_PATTERN.match(joint_name):
        _hip_link_ids.append(joint_id)
      elif UPPER_NAME_PATTERN.match(joint_name):
        _motor_link_ids.append(joint_id)
      # We either treat the lower leg or the toe as the foot link, depending on
      # the urdf version used.
      elif LOWER_NAME_PATTERN.match(joint_name):
        _lower_link_ids.append(joint_id)
      elif TOE_NAME_PATTERN.match(joint_name):
        #assert self._urdf_filename == URDF_WITH_TOES
        _foot_link_ids.append(joint_id)
      elif IMU_NAME_PATTERN.match(joint_name):
        _imu_link_ids.append(joint_id)
      else:
        raise ValueError("Unknown category of joint %s" % joint_name)


    _leg_link_ids.extend(_lower_link_ids)
    _leg_link_ids.extend(_foot_link_ids)

    #assert len(_foot_link_ids) == NUM_LEGS
    _hip_link_ids.sort()
    _motor_link_ids.sort()
    _lower_link_ids.sort()
    _foot_link_ids.sort()
    _leg_link_ids.sort()
    print("hip links", _hip_link_ids)
    print("motors", _motor_link_ids)
    print("lower_link_ids", _lower_link_ids)
    print("_foot_link_ids", _foot_link_ids)
    print("_leg_link_ids", _leg_link_ids)


def Bullet_collision(quadruped, robot, state, foot_link_ids, hip_link_ids, lower_link_ids, pyb_client, angle_tolerance=0.1, torque_tolerance = 0.1):
  """Check for self-collisions between the robot's links."""
  BAD_POSITIONS = []
  BAD_TORQUES = []
  if robot == "A1":
     BAD_POSITIONS = np.array([[0,1.225, -2.601,
                           0,1.1833,-2.63,
                           0,-0.376,-2.204,
                           0,-0.285,-2.22],
                         [0.535,0.726,-1.765,
                          -0.831,0.902,-1.92,
                          0.55,0.306,-1.44,
                          -0.651,0.6,-1.927]
                         ])

     BAD_TORQUES = np.array([
         [0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4],
         [0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3]
    ])
  elif robot == "Go1": 
     BAD_POSITIONS = np.array([
        [0,1.225, -2.601,0, 1.1833,-2.63,0, -0.376,-2.204,0, -0.285,-2.22],
        [0.535,0.726,-1.765, -0.831, 0.902, -1.92, 0.55, 0.306, -1.44, -0.651, 0.6, -1.927]
                         ])

     BAD_TORQUES = np.array([
         [0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4],
         [0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3]
    ])
     
  elif robot == "Aliengo":
     BAD_POSITIONS = np.array([
        [0,1.225, -2.601, 0, 1.1833,-2.63, 0, -0.376,-2.204, 0, -0.285,-2.22],
        [0.535,0.726,-1.765, -0.831, 0.902, -1.92, 0.55, 0.306, -1.44, -0.651, 0.6, -1.927]
                         ])

     BAD_TORQUES = np.array([
         [0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4, 0.566, 0.566, 4],
         [0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3, 0.444, 0.444, 3]
    ])
     
  # Проверка коллизий при близких углах
  if any(np.allclose(state, bad_angle, atol=angle_tolerance) for bad_angle in BAD_POSITIONS):
    print("danger, collision close\nChecking collisions between foot links and hip links:")
    # Проверка collisions между foot_link_ids и self._hip_link_ids
    for foot_link_id in foot_link_ids:
        for hip_link_id in hip_link_ids:
            if pyb_client.getClosestPoints(bodyA=quadruped, linkIndexA=foot_link_id,
                                                      bodyB=quadruped, linkIndexB=hip_link_id, distance=10):
                print(f"Collision detected between foot link {foot_link_id} and hip link {hip_link_id}")
                subprocess.run(['echo' 'pos_col', '>>', 'col_chec.txt'])
                return True
  elif any(np.allclose(state, bad_torque, atol=torque_tolerance) for bad_torque in BAD_TORQUES):     
    # Проверка коллизий между всеми foot
    print("Checking collisions between foot links:")
    for i, link_id_1 in enumerate(foot_link_ids):
        for link_id_2 in foot_link_ids[i + 1:]:
            if pyb_client.getClosestPoints(bodyA=quadruped, linkIndexA=link_id_1,
                                                      bodyB=quadruped, linkIndexB=link_id_2, distance=10):
                print(f"Collision detected between foot links {link_id_1} and {link_id_2}")
                subprocess.run(['echo' 'torq_col', '>>', 'col_chec.txt'])
                return True
  # Проверка коллизий между всеми  self._lower_link_ids
  #print("\nChecking collisions between lower links:")
  for i, link_id_1 in enumerate(lower_link_ids):
      for link_id_2 in  lower_link_ids[i + 1:]:
          if pyb_client.getClosestPoints(bodyA=quadruped, linkIndexA=link_id_1,
                                                    bodyB=quadruped, linkIndexB=link_id_2, distance=0.1):
              print(f"Collision detected between lower links {link_id_1} and {link_id_2}")
              subprocess.run('echo torq_col >> col_chec.txt', shell=True)
              return True
  return False


def check_collisions_with_angles(urdf_path, q):
    # Загружаем модель робота из URDF
    #q = q[::-1]
    model = pin.buildModelFromUrdf(urdf_path)
    collision_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.COLLISION)
    # Создаем конфигурацию на основе углов
    """!!!!!!ТУТ ПЕРЕБИРАЕТСЯ ЛЕВАЯ НОГА ОДНА, ИСПОЛЬЗОВАТЬ ПО ОБРАЗУ И ПОДОБИЮ СО ВСЕМИ"""
    #print("\nCollision object placements:")
    #for pair in pairs:
    #   collision_model.addCollisionPair(pair)

    collision_model.addAllCollisionPairs()
    for bad_pair in bad_pairs: 
      collision_model.removeCollisionPair(bad_pair)
    data = pin.Data(model)
    collision_data = pin.GeometryData(collision_model)
    # for k, oMg in enumerate(collision_data.oMg):
    #   print(("{:d} : {: .2f} {: .2f} {: .2f}"
    #     .format( k, *oMg.translation.T.flat )))
    #print("num collision pairs - initial:", len(collision_model.collisionPairs))
    # Обновляем состояние модели
    pin.forwardKinematics(model, data, q)
    pin.updateGeometryPlacements(model, data, collision_model, collision_data)
    pin.computeCollisions(model, data, collision_model, collision_data, q, False)
    for i in range(model.njoints):
      joint_model = model.joints[i]
      joint_name = model.names[i]
      #print(f"Joint name: {joint_name}, Joint model: {joint_model}")
    for k in range(len(collision_model.collisionPairs)):
      cr = collision_data.collisionResults[k]
      cp = collision_model.collisionPairs[k]
      if cr.isCollision():
         print("YES collision pair first:",
          cp.first, " ", "name", model.frames[cp.first].name,
          ", and the second",
          cp.second, " name ", model.frames[cp.second].name,
          "- collision:")
      # else:
      #    print("NO collision pair first:",
      #     cp.first, " ", "name", model.frames[cp.first].name,
      #     ", and the second",
      #     cp.second, " name ", model.frames[cp.second].name,
      #     "- collision:")
    # Перебираем пары звеньев для проверки коллизий
    

if __name__ == '__main__':
  currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  robot_interface_dir = os.path.join(currentdir, "../", "motion_imitation/python_interface/go1")
  os.sys.path.insert(0, robot_interface_dir)
  #from robot_interface_go1 import robot_interface as sdk # type: ignore # pytype: disable=import-error
  import robot_interface_go1 as sdk # type: ignore # pytype: disable=import-error
  import pybullet as pyb  
  import pybullet_data as pd
  ''' befin go1 urdf'''
  URDF_FILENAME = os.path.join(currentdir, "../", "motion_imitation/utilities/go1/urdf/go1.urdf")
  pyb.connect(pyb.GUI)
  pyb.setAdditionalSearchPath(pd.getDataPath())
  quadruped = pyb.loadURDF(URDF_FILENAME)
  timer =0
  '''end go1 urdf'''
  # while timer != 10:
  #     # Проходим по всем суставам и задаем случайный угол
  #     for joint_name, joint_index in d.items():
  #         random_angle = np.random.uniform(low=-np.pi, high=np.pi)
  #         pyb.resetJointState(
  #             bodyUniqueId=quadruped,
  #             jointIndex=joint_index,
  #             targetValue=random_angle
  #         )
      
  #     pyb.stepSimulation()
  #     time.sleep(dt)
  #     timer +=1

#check single joint
  LOWLEVEL = 0xff

  d = {'FR_0':0, 'FR_1':1, 'FR_2':2,
       'FL_0':3, 'FL_1':4, 'FL_2':5, 
       'RR_0':6, 'RR_1':7, 'RR_2':8, 
       'RL_0':9, 'RL_1':10, 'RL_2':11}
  
  udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
  safe = sdk.Safety(sdk.LeggedType.Go1)  
  cmd = sdk.LowCmd()
  state = sdk.LowState()
  udp.InitCmdData(cmd)

  #sdk.send_command(np.zeros(60, dtype=np.float32))
  bool = 0
  if bool == 1:
     URDF_FILENAME = os.path.join(currentdir, "../../../../", "motion_imitation/utilities/go1/urdf/norm urdf/go1.urdf") 
     pyb.connect(pyb.GUI)
     pyb.setAdditionalSearchPath(pd.getDataPath())
     quadruped = pyb.loadURDF(URDF_FILENAME)
     pyb.URDF_USE_SELF_COLLISION
     #_BuildUrdfIds(quadruped)
     udp.Recv()
     udp.GetRecv(state)
     while True:
      print(state)
  else:
    #_BuildUrdfIds(quadruped)
    angles = {key: 0.0 for key in d_BULLET.keys()}
    #angle_step = 0.01
    def control_joints(key, quadruped, angle_step=0.001):
      if key == ord('w'):      # FR_1
        angles['FR_1'] += angle_step
      elif key == ord('s'):
        angles['FR_1'] -= angle_step
      elif key == ord('a'):    # FL_1
        angles['FL_1'] += angle_step
      elif key == ord('d'):
        angles['FL_1'] -= angle_step
      elif key == ord('q'):    # RR_1
          angles['RR_1'] += angle_step
      elif key == ord('e'):
          angles['RR_1'] -= angle_step
      elif key == ord('z'):    # RL_1
          angles['FL_1'] += angle_step
      elif key == ord('c'):
          angles['RL_1'] -= angle_step
      # Добавленные условия для остальных двигателей
      elif key == ord('i'):    # FR_2
          angles['FR_2'] += angle_step
      elif key == ord('k'):
          angles['FR_2'] -= angle_step
      elif key == ord('j'):    # FL_2
          angles['FL_2'] += angle_step
      elif key == ord('l'):
          angles['FL_2'] -= angle_step
      elif key == ord('u'):    # RR_2
          angles['RR_2'] += angle_step
      elif key == ord('o'):
          angles['RR_2'] -= angle_step
      elif key == ord('m'):    # RL_2
          angles['RL_2'] += angle_step
      elif key == ord(','):
          angles['RL_2'] -= angle_step
      # Остальные двигатели, если нужно больше условий
      elif key == ord('t'):    # FR_3
          angles['FR_0'] += angle_step
      elif key == ord('g'):
          angles['FR_0'] -= angle_step
      elif key == ord('y'):    # FL_3
          angles['FL_0'] += angle_step
      elif key == ord('h'):
          angles['FL_0'] -= angle_step
      elif key == ord('r'):    # RR_3
          angles['RR_0'] += angle_step
      elif key == ord('f'):
          angles['RR_0'] -= angle_step
      elif key == ord('v'):    # RL_3
          angles['RL_0'] += angle_step
      elif key == ord('b'):
          angles['RL_0'] -= angle_step

      for key, joint_index in d_BULLET.items():
          pyb.resetJointState(bodyUniqueId=quadruped, jointIndex=joint_index, targetValue=angles[key])
    timer = 0
    while True:
      udp.Recv()
      udp.GetRecv(state)
      #pyb.URDF_USE_SELF_COLLISION
      random_angle = np.random.uniform(low=-np.pi, high=np.pi)
      keys = pyb.getKeyboardEvents()
      for k in keys:
          control_joints(k, quadruped)
          #print(URDF_FILENAME)
      check_collisions_with_angles(urdf_path=URDF_FILENAME, q=np.array(list(angles.values())))
      timer = 1
      # if (Bullet_collision(robot="Go1",
      #                  quadruped=quadruped,
      #                   state=np.array(list(angles.values())),
      #                   lower_link_ids=[4, 9, 14, 19],
      #                   foot_link_ids=[5,10,15,20],
      #                   hip_link_ids = [-1, 1, 2, 6, 7, 11, 12, 16, 17],
      #                   pyb_client=pyb
      #                   )):
      #    print("kek")
