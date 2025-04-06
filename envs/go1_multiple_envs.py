import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import gym
from gym import spaces
import numpy as np
import pybullet as pyb
import pybullet_data as pd
from motion_imitation.robots import go1
from motion_imitation.robots import robot_config
from agents.src.ddpg_agent import Agent as ddpg_agent
import torch
from torch.utils.tensorboard import SummaryWriter
import signal
import multiprocessing as mp
from functools import partial
import argparse
import traceback

MAX_TORQUE = np.array([28.7, 28.7, 40] * 4)

class SkillVectorSampler:
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
    
    def sample(self, n_samples=1):
        radius = self.rng.uniform(0, 1, size=n_samples)
        theta = self.rng.uniform(0, 2 * np.pi, size=n_samples)
        return radius[0], theta[0]

class CentralPatternGenerator:
    def __init__(self, N=4, a=0.5, dt=0.01, Omega=np.pi):
        self.N = N
        self.a = a
        self.dt = dt
        self.Omega = Omega
        self.M = np.array((
            (1, 0, 1, 0),
            (0, 1, 0, 1),
            (1, 0, 1, 0),
            (0, 1, 0, 1)
        ))
        self.PSI = np.array((
            (0, 0.5, 0, 0.5),
            (-0.5, 0, -0.5, 0),
            (0, 0.5, 0, 0.5),
            (-0.5, 0, -0.5, 0)
        ))
        self.MU_MIN, self.MU_MAX = 1, 2
        self.Z_MIN, self.Z_MAX = 0.2, 1
        self.OMEGA_MIN = 0
        self.r = np.ones(N)
        self.v = np.zeros(N)
        self.theta = np.zeros(N)
        self.alpha = np.array([np.pi, 0, np.pi, 0])  # Trot: LF+RH, RF+LH
        self.phi = self.theta + self.alpha
        self.r_dot_prev = np.zeros(N)
        self.v_dot_prev = np.zeros(N)
        self.theta_dot_prev = np.zeros(N)
        self.alpha_dot_prev = np.zeros(N)

    @staticmethod
    def map_value(val, val_inp_min, val_inp_max, val_out_min, val_out_max):
        return val_out_min + (val - val_inp_min) / (val_inp_max - val_inp_min) * (val_out_max - val_out_min)

    def omega_max(self, z_norm):
        return self.map_value(z_norm, 0, 1, self.Z_MIN, self.Z_MAX) * self.Omega

    def compute_derivatives(self, z_norm, mu, omega):
        r_dot = self.v.copy()
        f_mu = self.map_value(mu, -1, 1, self.MU_MIN, self.MU_MAX)
        v_dot = self.a**2 / 4 * (f_mu - self.r) - self.a * self.v
        OMEGA_MAX = self.omega_max(z_norm)
        theta_dot = self.map_value(omega, -1, 1, self.OMEGA_MIN, OMEGA_MAX)
        alpha_dot = np.zeros(self.N)
        for i in range(self.N):
            coupling_sum = OMEGA_MAX / 2
            for j in range(self.N):
                coupling_sum += self.r[j] * self.M[i, j] * np.sin(self.alpha[j] - self.alpha[i] - self.PSI[i, j])
            alpha_dot[i] = coupling_sum
        return r_dot, v_dot, theta_dot, alpha_dot

    def update_state(self, r_dot, v_dot, theta_dot, alpha_dot, step=0):
        if step == 0:
            self.r += r_dot * self.dt / 2
            self.v += v_dot * self.dt / 2
            self.theta += theta_dot * self.dt / 2
            self.alpha += alpha_dot * self.dt / 2
        else:
            self.r += (self.r_dot_prev + r_dot) * self.dt / 2
            self.v += (self.v_dot_prev + v_dot) * self.dt / 2
            self.theta += (self.theta_dot_prev + theta_dot) * self.dt / 2
            self.alpha += (self.alpha_dot_prev + alpha_dot) * self.dt / 2
        self.r_dot_prev = r_dot.copy()
        self.v_dot_prev = v_dot.copy()
        self.theta_dot_prev = theta_dot.copy()
        self.alpha_dot_prev = alpha_dot.copy()
        self.theta = np.mod(self.theta, 2 * np.pi)
        self.alpha = np.mod(self.alpha, 2 * np.pi)
        self.phi = np.mod(self.theta + self.alpha, 2 * np.pi)

    def reset(self):
        self.r = np.ones(self.N)
        self.v = np.zeros(self.N)
        self.theta = np.zeros(self.N)
        self.alpha = np.array([np.pi, 0, np.pi, 0])
        self.phi = self.theta + self.alpha
        self.r_dot_prev = np.zeros(self.N)
        self.v_dot_prev = np.zeros(self.N)
        self.theta_dot_prev = np.zeros(self.N)
        self.alpha_dot_prev = np.zeros(self.N)

    def step(self, z_norm, mu, omega, timestep):
        r_dot, v_dot, theta_dot, alpha_dot = self.compute_derivatives(z_norm, mu, omega)
        
        self.update_state(r_dot, v_dot, theta_dot, alpha_dot, timestep)
        
        return self.r, self.phi

def rotation_x(roll):
    return np.array(((1, 0, 0), (0, np.cos(roll), -np.sin(roll)), (0, np.sin(roll), np.cos(roll))))

def rotation_y(pitch):
    return np.array(((np.cos(pitch), 0, np.sin(pitch)), (0, 1, 0), (-np.sin(pitch), 0, np.cos(pitch))))

def rotation_z(yaw):
    return np.array(((np.cos(yaw), -np.sin(yaw), 0), (np.sin(yaw), np.cos(yaw), 0), (0, 0, 1)))

def rotation_full(roll, pitch, yaw):
    return rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)

def translation_full(x, y, z):
    return np.array((x, y, z))

def homogeneous_tf(roll, pitch, yaw, x, y, z):
    res = np.eye(4)
    R = rotation_full(roll, pitch, yaw)
    transl = translation_full(x, y, z)
    res[:3, :3] = R
    res[:3, -1] = transl
    return res

class MultiGo1Env(gym.Env):
    def __init__(self, num_robots=2, gui=False, sim_id=0):
        super(MultiGo1Env, self).__init__()
        self.gui = gui
        self.sim_id = sim_id
        self.pyb_client = pyb.connect(pyb.GUI if gui and sim_id == 0 else pyb.DIRECT)
        if self.pyb_client < 0:
            raise RuntimeError(f"Failed to connect to PyBullet for sim {sim_id}")
        
        self.time_step = 0.01
        self._setup_simulation()
        
        self.num_robots = num_robots
        self.robots = []
        self.prev_actions = [np.zeros(5)] * num_robots  # Updated for 5D action
        
        self.cpg_N = 4
        self.cpg_time_step = 0.01
        self.cpg_convergence_factor = 0.75
        self.cpg_Omega = np.pi
        self.cpgs = [CentralPatternGenerator(self.cpg_N, self.cpg_convergence_factor, self.cpg_time_step, self.cpg_Omega) for _ in range(num_robots)]
        self.timestep_counters = [0] * num_robots

        # Morphological parameters
        self.l = 0.1  # Target step size
        self.L = 0.1  # Target stride length
        self.w_y = 1  # Stride expansion factor
        self.h = 0.25  # Target height
        self.g_c = 0.075  # Max ground clearance
        self.g_p = 0.02  # Max ground penetration

        for i in range(num_robots):
            robot = go1.Go1(pybullet_client=pyb,
                          motor_control_mode=robot_config.MotorControlMode.POSITION,
                          self_collision_enabled=False)
            pyb.resetBasePositionAndOrientation(robot.quadruped, 
                                              [i * 2.0, sim_id * 2.0, self.h], 
                                              [0, 0, 0, 1])
            robot.ReceiveObservation()
            self.robots.append(robot)
        
        self.action_space = spaces.Box(low=np.array((-10, -10, -np.pi, -np.pi, 0)), high=np.array((10, 10, np.pi, np.pi, 1)), shape=(5,), dtype=np.float32)  # mu1, mu2, omega1, omega2, z_norm
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(len(self.robots[0].GetTrueObservation() + 
                                                    self.robots[0].GetFootContacts()),),
                                          dtype=np.float32)
        
        self.episode_rewards = [0] * num_robots

    def _setup_simulation(self):
        pyb.setAdditionalSearchPath(pd.getDataPath())
        pyb.setGravity(0, 0, -9.8)
        pyb.loadURDF("plane.urdf", basePosition=[0, 0, -0.01])
        pyb.setRealTimeSimulation(0)
        pyb.setTimeStep(self.time_step)
        pyb.setPhysicsEngineParameter(numSubSteps=2)

    def compute_target_positions(self, r, phi, l_hip_sign):
        px = self.l * (r - 1) * np.cos(phi)
        if np.abs(px) > self.l:
            print(f"step size {px}")
        py = l_hip_sign * self.L * self.w_y * r
        pz = -self.h + np.where(np.sin(phi) > 0, self.g_c, -self.g_p) * np.sin(phi)
        return np.stack([px, py, pz])

    def reset(self):
        if not pyb.isConnected(self.pyb_client):
            return None
        states = []
        for i, (robot, cpg) in enumerate(zip(self.robots, self.cpgs)):
            pyb.resetBasePositionAndOrientation(robot.quadruped,
                                              [i * 2.0, self.sim_id * 2.0, self.h],
                                              [0, 0, 0, 1])
            pyb.resetBaseVelocity(robot.quadruped,
                                linearVelocity=[0, 0, 0],
                                angularVelocity=[0, 0, 0])
            robot.ResetPose(add_constraint=False)
            robot.ReceiveObservation()
            state = np.concatenate([robot.GetTrueObservation(), robot.GetFootContacts()])
            states.append(state)
            self.episode_rewards[i] = 0
            self.prev_actions[i] = np.zeros(5)  # Updated for 5D
            cpg.reset()
            self.timestep_counters[i] = 0
        return states

    def step(self, actions):
        if not pyb.isConnected(self.pyb_client):
            return None, None, None, {}
        next_states = []
        rewards = []
        dones = []
        for i, (robot, action, cpg) in enumerate(zip(self.robots, actions, self.cpgs)):
            mu1, mu2, omega1, omega2, z_norm = action
            r, phi = cpg.step(z_norm, np.array([mu1, mu2, mu1, mu2]), np.array([omega1, omega2, omega1, omega2]), self.timestep_counters[i])
            self.timestep_counters[i] += 1
            
            joint_angles = []
            
            for leg_idx in range(4):
                l_hip_sign = (-1)**(leg_idx + 1)
                target_positions = self.compute_target_positions(r[leg_idx], phi[leg_idx], l_hip_sign)
                # print(f"{leg_idx}: {target_positions}; {r}; {phi}")
                joint_angles.append(go1.foot_position_in_hip_frame_to_joint_angle(target_positions, l_hip_sign))
            
            # print(joint_angles)
            robot.ApplyAction(np.array(joint_angles).flatten())
        
        pyb.stepSimulation()
        
        for i, robot in enumerate(self.robots):
            robot.ReceiveObservation()
            next_state = np.concatenate([robot.GetTrueObservation(), robot.GetFootContacts()])
            
            x, y, z = robot.GetBasePosition()
            roll, pitch, yaw = robot.GetBaseRollPitchYaw()
            v_x, v_y, v_z = robot.GetBaseVelocity()
            
            tf = homogeneous_tf(0, 0, 0, x, y, z) @ homogeneous_tf(roll, pitch, yaw, 0, 0, 0)
            foot_positions = robot.GetFootPositionsInBaseFrame()
            foot_heights = [(tf @ [*pos, 1])[-2] for pos in foot_positions]
            
            MAX_ROLL = 45
            MAX_PITCH =45
            MIN_HEIGHT = 0.15
            done = (z < MIN_HEIGHT or np.abs(roll) > MAX_ROLL * np.pi / 180 or np.abs(pitch) > MAX_PITCH * np.pi / 180)
            done_penalty = 10 if done else 0

            _reward = self.reward(
                x=x, y=y, z=z,
                roll=roll, pitch=pitch, yaw = yaw,
                v_x=v_x, v_y=v_y, v_z=v_z,
                cur_action=actions[i], prev_action=self.prev_actions[i],
                contacts=robot.GetFootContacts(), foot_heights=foot_heights
            )
            self.episode_rewards[i] += _reward - done_penalty
            
            next_states.append(next_state)
            rewards.append(_reward)
            dones.append(done)
            self.prev_actions[i] = actions[i]
        
        return next_states, rewards, dones, {}

    def reward(
        self,
        x, y, z, 
        roll, pitch, yaw, 
        v_x, v_y, v_z,
        cur_action, prev_action, 
        contacts, foot_heights
    ):
        FORWARD_SPEED_REWARD_WEIGHT = 2
        FORWARD_SPEED_TARGET = 0.5
        FORWARD_SPEED_VARIANCE = 0.5
        
        VERTICAL_SPEED_PENALTY_WEIGHT = 2
        VERTICAL_SPEED_MAX = 0.15
        VERTICAL_SPEED_VARIANCE = 0.5

        TILT_WEIGHT = 1

        HEIGHT_WEIGHT = 5
        HEIGHT_TARGET = 0.25

        ACTION_WEIGHT = 0.01

        FOOT_CLEARANCE_PENALTY_WEIGHT = 3
        FOOT_CLEARANCE_MAX = 0.125
        
        v_z_abs = np.abs(v_z)

        forward_velocity_reward = FORWARD_SPEED_REWARD_WEIGHT * np.exp(-(v_x - FORWARD_SPEED_TARGET)**2 / FORWARD_SPEED_VARIANCE**2)
        vertical_velocity_penalty = -VERTICAL_SPEED_PENALTY_WEIGHT * v_z_abs if v_z_abs > VERTICAL_SPEED_MAX else 0
        tilt_penalty = -TILT_WEIGHT * (roll**2 + pitch**2)
        action_penalty = -ACTION_WEIGHT * np.sum(np.square(cur_action - prev_action))
        
        total_reward = (
            forward_velocity_reward + 
            vertical_velocity_penalty + 
            tilt_penalty + 
            action_penalty
        )
        return total_reward

    def close(self):
        if pyb.isConnected(self.pyb_client):
            pyb.disconnect()
        print(f"Simulation {self.sim_id} closed cleanly")

    def __del__(self):
        self.close()

def env_worker(sim_id, num_robots, gui, conn):
    try:
        env = MultiGo1Env(num_robots=num_robots, gui=gui, sim_id=sim_id)
        conn.send(("ready", sim_id))
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                states = env.reset()
                conn.send(states)
            elif cmd == "step":
                actions = data
                next_states, rewards, dones, info = env.step(actions)
                conn.send((next_states, rewards, dones, info))
            elif cmd == "close":
                env.close()
                conn.send(("closed", sim_id))
                conn.close()
                break
            else:
                raise ValueError(f"Unknown command: {cmd}")
    except Exception as e:
        # Capture and format the full traceback
        tb_str = traceback.format_exc()
        error_msg = f"Error in sim {sim_id}: {str(e)}\nTraceback:\n{tb_str}"
        conn.send(("error", error_msg))
        env.close()
        conn.close()

class ParallelMultiGo1Env:
    def __init__(self, num_sims=2, num_robots_per_sim=2, gui=False):
        self.num_sims = num_sims
        self.num_robots_per_sim = num_robots_per_sim
        self.total_robots = num_sims * num_robots_per_sim
        self.processes = []
        self.conns = []
        
        for sim_id in range(num_sims):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=env_worker, args=(sim_id, num_robots_per_sim, gui and sim_id == 0, child_conn))
            p.start()
            self.processes.append(p)
            self.conns.append(parent_conn)
        
        for conn in self.conns:
            msg, sim_id = conn.recv()
            if msg != "ready":
                raise RuntimeError(f"Simulation {sim_id} failed to start: {msg}")
            print(f"Simulation {sim_id} ready")
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.total_robots, 5), dtype=np.float32)  # Updated
        env = MultiGo1Env(num_robots=num_robots_per_sim)
        single_obs_shape = env.observation_space.shape[0]
        env.close()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(self.total_robots, single_obs_shape),
                                          dtype=np.float32)

    def reset(self):
        states = []
        for conn in self.conns:
            conn.send(("reset", None))
        for conn in self.conns:
            result = conn.recv()
            if isinstance(result, tuple) and result[0] == "error":
                # Print the detailed error message with traceback
                print(f"Error from worker: {result[1]}")
                raise RuntimeError(f"Error in simulation reset: {result[1]}")
            states.extend(result)
        return np.array(states)

    def step(self, actions):
        actions_per_sim = np.split(actions, self.num_sims)
        for conn, sim_actions in zip(self.conns, actions_per_sim):
            conn.send(("step", sim_actions))
        
        next_states = []
        rewards = []
        dones = []
        infos = []
        for conn in self.conns:
            result = conn.recv()
            if isinstance(result, tuple) and len(result) == 2 and result[0] == "error":
                raise RuntimeError(f"Error in simulation step: {result[1]}")
            elif isinstance(result, tuple) and len(result) == 4:
                ns, r, d, i = result
                next_states.extend(ns)
                rewards.extend(r)
                dones.extend(d)
                infos.append(i)
            else:
                raise ValueError(f"Unexpected result format from simulation: {result}")
        
        return np.array(next_states), np.array(rewards), np.array(dones), infos

    def close(self):
        for conn in self.conns:
            conn.send(("close", None))
        for conn, p in zip(self.conns, self.processes):
            try:
                msg, sim_id = conn.recv()
                if msg != "closed":
                    print(f"Simulation {sim_id} closed with message: {msg}")
            except:
                pass
            p.join()
        self.conns = []
        self.processes = []
        print("All simulations closed")

def train_multi_robot(num_sims, num_robots_per_sim, gui, n_episodes, load_weights, weights_path=None):
    env = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Device: {device}")
    writer = SummaryWriter()
    try:
        env = ParallelMultiGo1Env(num_sims=num_sims, num_robots_per_sim=num_robots_per_sim, gui=gui)
        total_robots = num_sims * num_robots_per_sim
        
        state_size = env.observation_space.shape[1]
        action_size = 5  # Updated
        agent = ddpg_agent(state_size=state_size, action_size=action_size, random_seed=0)
        agent.actor_local.to(device)
        agent.critic_local.to(device)
        agent.actor_target.to(device)
        agent.critic_target.to(device)
        print(f"Actor Local Device: {next(agent.actor_local.parameters()).device}")
        print(f"Critic Local Device: {next(agent.critic_local.parameters()).device}")
        
        weights_dir = 'weights_parallel'
        os.makedirs(weights_dir, exist_ok=True)
        
        if load_weights:
            if not weights_path:
                raise ValueError("Weights path must be specified when --load-weights is True")
            actor_path = os.path.join(weights_path, 'actor_max.pth')
            critic_path = os.path.join(weights_path, 'critic_max.pth')
            try:
                if os.path.exists(actor_path) and os.path.exists(critic_path):
                    agent.actor_local.load_state_dict(torch.load(actor_path, map_location=device))
                    agent.critic_local.load_state_dict(torch.load(critic_path, map_location=device))
                    agent.actor_local.train()
                    agent.critic_local.train()
                    print(f"Loaded weights from {weights_path}")
                else:
                    print(f"No weights found at {actor_path} or {critic_path}. Starting from scratch.")
            except Exception as e:
                print(f"Error loading weights: {e}. Starting from scratch.")
        
        print("Pre-filling experience buffer...")
        states = env.reset()
        if states is not None:
            for _ in range(1000):
                actions = np.random.uniform(-1, 1, (total_robots, 5))  # Updated
                next_states, rewards, dones, _ = env.step(actions)
                if next_states is None:
                    break
                for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
                    agent.step(s, a, r, ns, d)
                states = next_states
        
        max_reward = -float('inf')
        for episode in range(n_episodes):
            states = env.reset()
            if states is None:
                break
            total_rewards = np.zeros(total_robots)
            done = np.zeros(total_robots, dtype=bool)
            step_count = 0
            while not all(done):
                step_count += 1
                actions = []
                for i, state in enumerate(states):
                    if not done[i]:
                        action = agent.act(state, add_noise=True)
                        actions.append(action)
                    else:
                        actions.append(np.zeros(5))  # Updated
                actions = np.array(actions)
                
                next_states, rewards, dones, _ = env.step(actions)
                if next_states is None:
                    break
                
                for i, (s, a, r, ns, d) in enumerate(zip(states, actions, rewards, next_states, dones)):
                    if not done[i]:
                        agent.step(s, a, r, ns, d)
                        total_rewards[i] += r
                states = next_states
                done = dones
            
            avg_reward = np.mean(total_rewards)
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.2f}")
            if writer:
                writer.add_scalar('Reward/Average', avg_reward, episode)
                for i in range(total_robots):
                    writer.add_scalar(f'Reward/Robot_{i}', total_rewards[i], episode)
            
            if avg_reward > max_reward:
                print(f"!!! Max reward update: {avg_reward} !!!")
                max_reward = avg_reward
                torch.save(agent.actor_local.state_dict(), os.path.join(weights_dir, 'actor_max.pth'))
                torch.save(agent.critic_local.state_dict(), os.path.join(weights_dir, 'critic_max.pth'))
        
        return max_reward
    
    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt. Cleaning up...")
        if env:
            env.close()
        if writer:
            writer.close()
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        if env:
            env.close()
        if writer:
            writer.close()
        raise
    finally:
        if env:
            env.close()
        if writer:
            writer.close()

def signal_handler(sig, frame):
    print("\nSignal received. Exiting gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-robot simulation with DDPG")
    parser.add_argument('--num-sims', type=int, default=1)
    parser.add_argument('--num-robots-per-sim', type=int, default=1)
    parser.add_argument('--use-gui', action='store_true')
    parser.add_argument('--episodes', type=int, default=1_000_000)
    parser.add_argument('--load-weights', action='store_true')
    parser.add_argument('--weights-path', type=str)

    args = parser.parse_args()
    if args.load_weights and not args.weights_path:
        parser.error("--weights-path is required when --load-weights is specified")

    mp.set_start_method('spawn')
    signal.signal(signal.SIGINT, signal_handler)

    try:
        max_reward = train_multi_robot(
            num_sims=args.num_sims,
            num_robots_per_sim=args.num_robots_per_sim,
            gui=args.use_gui,
            n_episodes=args.episodes,
            load_weights=args.load_weights,
            weights_path=args.weights_path
        )
        if max_reward is not None:
            print(f"Maximum average reward: {max_reward}")
        else:
            print("Training terminated early")
    except SystemExit:
        print("Program terminated by signal")
    except Exception as e:
        print(f"Main execution failed: {e}")
    finally:
        print("Execution completed")