import os
import argparse
import inspect
import gym
from gym import spaces
import numpy as np
import pybullet as pyb
import pybullet_data as pd
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
print(currentdir)
print(parentdir)
from motion_imitation.robots import go1
from motion_imitation.envs import env_builder
from motion_imitation.robots import robot_config
from agents.src.ddpg_agent import Agent as ddpg_agent
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
import time
MAX_TORQUE = np.array([28.7, 28.7, 40] * 4)
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

def reward(v_x, theta, u_prev, Ts, Tf, done, contacts):
    """
    Вычисляет значение функции вознаграждения.

    Параметры:
    v_x : float - скорость центра масс туловища в направлении x
    y : float - текущая высота центра масс туловища
    theta : float - угол наклона туловища (в радианах)
    u_prev : list - список значений действий для суставов из предыдущего временного шага
    Ts : float - текущее время симуляции
    Tf : float - общее время симуляции
    done : bool - флаг завершения эпизода (робот упал)

    Возвращает:
    float - значение вознаграждения
    """
    #штраф за 4 ноги в воздухе
    contact_penalty = -50 * (np.prod(contacts))
    # Штраф за наклон туловища
    tilt_penalty = -20 * (theta ** 2)

    # Штраф за резкие изменения в действиях (стабильность управления)
    action_penalty = -0.01 * np.sum(np.square(u_prev))

    # Награда за скорость движения вперёд
    velocity_reward = 10 * v_x

    # Штраф за завершение эпизода (падение робота)
    if done:
        termination_penalty = -1000
    else:
        termination_penalty = 0

    # Временной бонус (поощрение за продвижение во времени)
    time_bonus = 25 * (Ts / Tf)

    # Итоговое вознаграждение
    total_reward = (
        velocity_reward
        + contact_penalty
        + tilt_penalty
        + action_penalty
        + termination_penalty
        + time_bonus
    )

    return total_reward

# Определяем класс среды
class Go1Env(gym.Env):
    def __init__(self, pyb_client = None, gui = True):
        super(Go1Env, self).__init__()
        
        # Инициализация PyBullet
        if pyb_client == None:
            if gui:
                pyb.connect(pyb.GUI)
            else:
                pyb.connect(pyb.DIRECT)
            pyb.setAdditionalSearchPath(pd.getDataPath())
            pyb.setGravity(0, 0, -9.8)
            pyb.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
            pyb.setRealTimeSimulation(0)
            self.pyb_client = pyb
        else:
            self.pyb_client = pyb_client
        
        # Инициализация робота
        self.MAX_TORQUE = np.array([28.7, 28.7, 40] * 4)
        self.robot = go1.Go1(pybullet_client=self.pyb_client, motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
                             self_collision_enabled=True, motor_torque_limits=self.MAX_TORQUE)
        self.robot.ReceiveObservation()
        
        # Определение пространства действий и состояний
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.robot.GetTrueObservation()+self.robot.GetFootContacts()),), dtype=np.float32)
        
        # Инициализация TensorBoard
        self.writer = SummaryWriter()
        self.episode_reward = 0
        self.episode_count = 0
        self.step_count = 0
        self.last_action = None

    def reset(self):
        # Сброс состояния робота
        self.pyb_client.resetBasePositionAndOrientation(self.robot.quadruped, [0, 0, 0.25], [0, 0, 0, 1])
        self.pyb_client.resetBaseVelocity(self.robot.quadruped, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        self.robot.ResetPose(add_constraint=False)
        self.robot.ReceiveObservation()
        
        # Возвращаем начальное состояние
        state = np.concatenate([self.robot.GetTrueObservation(), self.robot.GetFootContacts()])
        self.episode_reward = 0
        self.step_count = 0
        self.last_action = None
        return state

    def step(self, action, apply_action=True):
        self.step_count += 1
        prev_action = self.last_action
        self.last_action = np.array(action, dtype=np.float32)

        # Применяем действие
        if apply_action:
            action = self.robot._ClipMotorCommands(
                motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
                motor_commands=10 * action
            )
            self.robot.ApplyAction(action)
        self.pyb_client.stepSimulation()
        self.robot.ReceiveObservation()
        #self.robot.Step(action)
        # Получаем следующее состояние
        contacts = self.robot.GetFootContacts()
        next_state = np.concatenate([self.robot.GetTrueObservation(), contacts])

        base_rpy = self.robot.GetBaseRollPitchYaw()
        base_pos = self.robot.GetBasePosition()
        done = base_pos[2] < 0.18 or np.sum(base_rpy**2) >= 0.73
        v_xyz = self.robot.GetBaseVelocity()
        yaw_rate = self.robot.GetBaseRollPitchYawRate()[2]

        # Вычисляем награду
        _reward = self.reward(
            v_x=v_xyz[0],
            v_y=v_xyz[1],
            yaw_rate=yaw_rate,
            roll=base_rpy[0],
            pitch=base_rpy[1],
            y=base_pos[2],
            Ts=0,
            Tf=1000,
            contacts=contacts,
            joint_torques=self.robot.GetMotorTorques(),
            last_action=prev_action,
            current_action=self.last_action,
            done=done
        )
        self.episode_reward += _reward
        
        # Логируем награду в TensorBoard
        if done:
            self.writer.add_scalar('Reward/Episode', self.episode_reward, self.episode_count)
            self.episode_count += 1
        
        return next_state, _reward, done, {}

    def _get_contact_positions_world(self):
        foot_pos_base = np.array(self.robot.GetFootPositionsInBaseFrame(), dtype=np.float32)
        base_pos, base_orn = self.pyb_client.getBasePositionAndOrientation(self.robot.quadruped)
        rot = np.array(self.pyb_client.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
        foot_world = (rot @ foot_pos_base.T).T + np.array(base_pos, dtype=np.float32)
        return foot_world

    def _point_to_segment_distance(self, point, a, b):
        ab = b - a
        denom = np.dot(ab, ab)
        if denom <= 1e-8:
            return np.linalg.norm(point - a)
        t = np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(point - proj)

    def _triangle_margin(self, point, tri):
        def _sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = _sign(point, tri[0], tri[1]) >= 0.0
        b2 = _sign(point, tri[1], tri[2]) >= 0.0
        b3 = _sign(point, tri[2], tri[0]) >= 0.0
        inside = (b1 == b2) and (b2 == b3)

        d0 = self._point_to_segment_distance(point, tri[0], tri[1])
        d1 = self._point_to_segment_distance(point, tri[1], tri[2])
        d2 = self._point_to_segment_distance(point, tri[2], tri[0])
        min_dist = min(d0, d1, d2)
        return min_dist if inside else -min_dist

    def compute_stability_margin(self, contacts):
        contact_indices = [i for i, c in enumerate(contacts) if c]
        if len(contact_indices) < 3:
            return 0.0

        foot_world = self._get_contact_positions_world()
        com_proj = np.array(self.robot.GetBasePosition()[:2], dtype=np.float32)

        if len(contact_indices) == 3:
            tri = foot_world[contact_indices, :2]
            return self._triangle_margin(com_proj, tri)

        best_margin = -np.inf
        for i in range(len(contact_indices)):
            for j in range(i + 1, len(contact_indices)):
                for k in range(j + 1, len(contact_indices)):
                    tri_idx = [contact_indices[i], contact_indices[j], contact_indices[k]]
                    tri = foot_world[tri_idx, :2]
                    margin = self._triangle_margin(com_proj, tri)
                    if margin > best_margin:
                        best_margin = margin

        return best_margin if np.isfinite(best_margin) else 0.0

    def reward(self, v_x, v_y, yaw_rate, y, roll, pitch, Ts, Tf, contacts, joint_torques=None, last_action=None, current_action=None, done=False):
        """
        Функция вознаграждения для шагающего робота (PPO)
        """
        v_x_cmd = 0.6
        v_y_cmd = 0.0
        yaw_rate_cmd = 0.0
        target_height = 0.25

        k_v_lin = 2.0
        k_v_ang = 0.5
        k_post = 5.0
        k_h = 10.0
        k_tau = 0.001
        k_delta_a = 0.01
        k_contact = 0.1
        k_eta = 1.0
        k_survive = 0.05
        k_fall = 7.5

        vel_error = (v_x - v_x_cmd) ** 2 + (v_y - v_y_cmd) ** 2
        velocity_reward = k_v_lin * (-vel_error + np.clip(-v_x_cmd, 2*v_x_cmd, v_x))
        yaw_rate_penalty = -k_v_ang * (yaw_rate - yaw_rate_cmd) ** 2
        posture_penalty = -k_post * (roll ** 2 + pitch ** 2)
        height_penalty = -k_h * ((y - target_height) ** 2)

        energy_penalty = 0.0
        if joint_torques is not None:
            energy_penalty = -k_tau * np.sum(np.square(joint_torques))

        action_smoothness = 0.0
        if last_action is not None and current_action is not None:
            action_smoothness = -k_delta_a * np.sum(np.square(current_action - last_action))

        contact_reward = k_contact * (np.sum(contacts) / float(len(contacts)))
        stability_margin = self.compute_stability_margin(contacts)
        stability_reward = k_eta * stability_margin
        survival_bonus = k_survive
        fall_penalty = -k_fall if done else 0.0

        total_reward = (
            velocity_reward
            + yaw_rate_penalty
            + posture_penalty
            + height_penalty
            + energy_penalty
            + action_smoothness
            + contact_reward
            + stability_reward
            + survival_bonus
            + fall_penalty
        )

        if self.writer:
            components = {
                "velocity_reward": velocity_reward,
                "yaw_rate_penalty": yaw_rate_penalty,
                "posture_penalty": posture_penalty,
                "height_penalty": height_penalty,
                "energy_penalty": energy_penalty,
                "action_smoothness": action_smoothness,
                "contact_reward": contact_reward,
                "stability_reward": stability_reward,
                "survival_bonus": survival_bonus,
                "fall_penalty": fall_penalty,
                "total_reward": total_reward,
            }
            for name, value in components.items():
                self.writer.add_scalar(f"reward_{name}", value, self.step_count)

        return total_reward

    def render(self, mode='human'):
        pass  # PyBullet уже визуализирует среду

    def close(self):
        self.pyb_client.disconnect()
        self.writer.close()

def train(n_episodes=1000, max_t=1000, print_every=100, prefill_steps=5000, robot = go1.Go1, pyb = pyb):
    done = False
    scores_deque = deque(maxlen=print_every)
    scores = []
    experience_buffer = deque(maxlen=prefill_steps)

    # Initialize agent
    state_size = np.size(np.array(robot.GetTrueObservation() + robot.GetFootContacts()))
    agent = ddpg_agent(state_size=state_size, action_size=12, random_seed=2)

    # Step 1: Prefill experience buffer with random actions
    print("Filling experience buffer with random actions...")
    for _ in range(prefill_steps):
        robot.ReceiveObservation()
        state = robot.GetTrueObservation() + robot.GetFootContacts()
        action = np.random.uniform(low=-1, high=1, size=12)  # Random action
        print("NON_CLIPPED ACTION",action)
        action = robot._ClipMotorCommands(
            motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
            motor_commands=10*action,
        )
        print("CLIPPED ACTION",action)
        robot.ApplyAction(action)
        pyb.stepSimulation()
        robot.ReceiveObservation()
        next_state = robot.GetTrueObservation() + robot.GetFootContacts()
        _reward = reward(
            v_x=robot.GetBaseVelocity()[0],
            y=robot.GetBasePosition()[2],
            roll=robot.GetBaseRollPitchYaw()[0],
            pitch=robot.GetBaseRollPitchYaw()[1],
            yaw=robot.GetBaseRollPitchYaw()[2],
            Ts=0,
            Tf=max_t,
            target_height=robot.GetDefaultInitPosition[2]
        )
        done = robot.GetBasePosition()[2] < 0.1 or np.sum(robot.GetBaseRollPitchYaw()) >= 0.73
        experience_buffer.append((state, action, _reward, next_state, done))
        if done:
            pyb.resetBasePositionAndOrientation(robot.quadruped, [0, 0, 0.25], [0, 0, 0, 1])
            pyb.resetBaseVelocity(robot.quadruped, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            robot.ResetPose(add_constraint=False)

    # Step 2: Train agent using DDPG
    print("Starting DDPG training...")

    done = False
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        done = False
        pyb.resetBasePositionAndOrientation(robot.quadruped, [0, 0, 0.25], [0, 0, 0, 1])
        pyb.resetBaseVelocity(robot.quadruped, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
        robot.ResetPose(add_constraint=False)
        #pyb.stepSimulation()  # Выполняем шаг симуляции
        robot.ReceiveObservation()
        state = robot.GetTrueObservation() + robot.GetFootContacts()
        #print("the size of state is ", np.size(state))
        #print("the size of true obs is ", np.size(robot.GetTrueObservation()))
        agent.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(np.array(state), add_noise=True)
            action = robot._ClipMotorCommands(
                motor_control_mode=go1.robot_config.MotorControlMode.TORQUE,
                motor_commands=10*action
            )
            robot.ApplyAction(action)
            #print(robot.GetBasePosition()[2])
            pyb.stepSimulation()
            robot.ReceiveObservation()

            next_state = robot.GetTrueObservation() + robot.GetFootContacts()
            _reward = reward(
                v_x=robot.GetBaseVelocity()[0], 
                y=robot.GetBasePosition()[2], 
                theta=robot.GetBaseRollPitchYaw()[1], 
                Ts=t,
                Tf=max_t,
                contacts=robot.GetFootContacts()
            )
            if robot.GetBasePosition()[2] < 0.18 or np.sum(robot.GetBaseRollPitchYaw())>=0.73:
                done = True
                print("TERMINATED", robot.GetBasePosition()[2],"  RPY", np.abs(np.sum(robot.GetBaseRollPitchYaw())))
            agent.step(state=np.array(state), action=action, reward=_reward, next_state=np.array(next_state), done=done)
            if done:
                break
            state = next_state
            score += _reward
            #print("KEK")

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 500:
                torch.save(agent.actor_local.state_dict(), 'actor_weights_{}.pth'.format(i_episode))
                torch.save(agent.critic_local.state_dict(), 'critic_weights_{}.pth'.format(i_episode)) 


    return scores

scores = []

def _set_joint_angle_by_name(env, joint_name, angle):
    joint_id = env.robot._joint_name_to_id[joint_name]
    env.pyb_client.resetJointState(env.robot.quadruped, joint_id, angle, targetVelocity=0)


def _get_default_pose_by_name(env):
    default_pose = env.robot.GetDefaultInitJointPose()
    return {name: float(angle) for name, angle in zip(go1.MOTOR_NAMES, default_pose)}


def _apply_position_control(env, pose_by_name):
    for i, name in enumerate(go1.MOTOR_NAMES):
        joint_id = env.robot._joint_name_to_id[name]
        env.pyb_client.setJointMotorControl2(
            bodyIndex=env.robot.quadruped,
            jointIndex=joint_id,
            controlMode=env.pyb_client.POSITION_CONTROL,
            targetPosition=float(pose_by_name[name]),
            force=float(env.MAX_TORQUE[i]),
            positionGain=0.5,
            velocityGain=0.3,
        )


def _disable_motors(env):
    for name in go1.MOTOR_NAMES:
        joint_id = env.robot._joint_name_to_id[name]
        env.pyb_client.setJointMotorControl2(
            bodyIndex=env.robot.quadruped,
            jointIndex=joint_id,
            controlMode=env.pyb_client.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0,
        )


def _hold_lifted_leg_pose(calf_joint_name, pose_by_name):
    hip_joint = calf_joint_name.replace("calf", "hip")
    thigh_joint = calf_joint_name.replace("calf", "thigh")
    pose_by_name[hip_joint] = 0.2
    pose_by_name[thigh_joint] = 1.0
    pose_by_name[calf_joint_name] = -1.2


def _step_static(env, action=None, lift_calf_joint=None):
    if action is None:
        action = np.zeros(12)
    if lift_calf_joint:
        _set_joint_angle_by_name(env, lift_calf_joint.replace("calf", "hip"), 0.2)
        _set_joint_angle_by_name(env, lift_calf_joint.replace("calf", "thigh"), 1.0)
        _set_joint_angle_by_name(env, lift_calf_joint, -1.2)
    env.robot.ApplyAction(action)
    env.pyb_client.stepSimulation()
    env.robot.ReceiveObservation()


def _step_env(env, action=None, apply_action=True):
    if action is None:
        action = np.zeros(12)
    env.step(action, apply_action=apply_action)


def test_stability(env, steps=500):
    def _foot_index_from_calf_joint(calf_joint):
        prefix = calf_joint.split("_")[0]
        order = ["FR", "FL", "RR", "RL"]
        return order.index(prefix)

    def _avg_margin(lift_calf_joint=None):
        margins = []
        pose = _get_default_pose_by_name(env)
        if lift_calf_joint:
            _hold_lifted_leg_pose(lift_calf_joint, pose)
        for _ in range(steps):
            _apply_position_control(env, pose)
            _step_env(env, action=np.zeros(12), apply_action=False)
            contacts = list(env.robot.GetFootContacts())
            if lift_calf_joint:
                contacts[_foot_index_from_calf_joint(lift_calf_joint)] = False
            margins.append(env.compute_stability_margin(contacts))
        return float(np.mean(margins))

    env.reset()
    baseline = _avg_margin()
    print(f"[stability] baseline avg margin: {baseline:.4f}")

    calf_joints = ["FL_calf_joint", "RR_calf_joint", "RL_calf_joint", "FR_calf_joint"]
    for calf_joint in calf_joints:
        env.reset()
        lifted = _avg_margin(lift_calf_joint=calf_joint)
        print(f"[stability] lifted {calf_joint} avg margin: {lifted:.4f}")
        assert lifted <= baseline + 1e-3, f"Stability increased unexpectedly for {calf_joint}"


def test_velocity_direction(env, steps=500):
    env.reset()
    _disable_motors(env)
    v_x_values = []
    for _ in range(steps):
        base_pos = env.robot.GetBasePosition()
        env.pyb_client.applyExternalForce(
            objectUniqueId=env.robot.quadruped,
            linkIndex=-1,
            forceObj=[200, 0, 0],
            posObj=base_pos,
            flags=env.pyb_client.WORLD_FRAME,
        )
        _step_env(env, action=np.zeros(12), apply_action=False)
        v_x_values.append(env.robot.GetBaseVelocity()[0])
    print(f"[velocity] v_x start/end: {v_x_values[0]:.4f} -> {v_x_values[-1]:.4f}")
    assert v_x_values[-1] > v_x_values[0] + 0.005, "v_x did not increase under +X force"


def test_roll_pitch_logging(env, steps=500):
    env.reset()
    env.pyb_client.setGravity(0, 0, 0)

    base_pos = env.robot.GetBasePosition()
    for i in range(steps):
        roll = (i / steps) * 0.6
        orn = env.pyb_client.getQuaternionFromEuler([roll, 0, 0])
        env.pyb_client.resetBasePositionAndOrientation(env.robot.quadruped, base_pos, orn)
        _step_env(env, action=np.zeros(12))
        rpy = env.robot.GetBaseRollPitchYaw()
        env.writer.add_scalar("test/roll", rpy[0], i)

    for i in range(steps):
        pitch = (i / steps) * 0.6
        orn = env.pyb_client.getQuaternionFromEuler([0, pitch, 0])
        env.pyb_client.resetBasePositionAndOrientation(env.robot.quadruped, base_pos, orn)
        _step_env(env, action=np.zeros(12))
        rpy = env.robot.GetBaseRollPitchYaw()
        env.writer.add_scalar("test/pitch", rpy[1], steps + i)

    print("[angles] roll/pitch logged to TensorBoard (test/roll, test/pitch)")
    env.pyb_client.setGravity(0, 0, -9.8)


def test_height_component(env, steps=500):
    target_height = 0.25
    k_h = 10.0

    env.reset()
    heights = []
    pose = _get_default_pose_by_name(env)
    for _ in range(steps):
        _apply_position_control(env, pose)
        _step_env(env, action=np.zeros(12), apply_action=False)
        heights.append(env.robot.GetBasePosition()[2])
    y_stand = float(np.mean(heights))
    height_penalty_stand = -k_h * ((y_stand - target_height) ** 2)
    print(f"[height] stand y={y_stand:.4f}, height_penalty={height_penalty_stand:.4f}")

    env.reset()
    _disable_motors(env)
    for _ in range(steps):
        _step_env(env, action=np.zeros(12), apply_action=False)
    y_fall = float(env.robot.GetBasePosition()[2])
    height_penalty_fall = -k_h * ((y_fall - target_height) ** 2)
    print(f"[height] fall y={y_fall:.4f}, height_penalty={height_penalty_fall:.4f}")

    assert abs(height_penalty_stand) < 0.05, "Height penalty at stand should be near zero"
    assert abs(height_penalty_fall) > abs(height_penalty_stand) + 0.05, "Height penalty did not increase after fall"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Go1Env tests")
    parser.add_argument(
        "--test",
        choices=["stability", "velocity", "angles", "height", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    args = parser.parse_args()

    env = Go1Env(gui=args.gui)
    try:
        if args.test in ("stability", "all"):
            test_stability(env)
        if args.test in ("velocity", "all"):
            test_velocity_direction(env)
        if args.test in ("angles", "all"):
            test_roll_pitch_logging(env)
        if args.test in ("height", "all"):
            test_height_component(env)
    finally:
        env.close()