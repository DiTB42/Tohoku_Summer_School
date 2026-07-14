import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time
from mazes.make_maze import create_maze_layout, get_valid_spawn_points, astar
from mazes.mujoco_tools import make_maze_on_mujoco
from cpg.snake_cpg import PaperCPG
from config_utils import load_config

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode=None, config=None):
        super(SnakeEnv, self).__init__()

        if config is None:
            config = load_config(None)
        self.config = config

        self.cpg_frequency = config.env.cpg_frequency
        self.rl_frequency = config.env.rl_frequency
        self.sim_steps_per_rl_step = int(self.cpg_frequency / self.rl_frequency)
        self.max_episode_steps = config.env.max_episode_steps
        self.R = config.cpg.base_amplitude       # Target amplitude R
        self.omega = config.cpg.base_omega       # Base omega ω
        self.delta = config.cpg.base_delta       # Offset δ

        self.start_pos_maze = (1, 1)
        #self.init_qpos = np.zeros(12, dtype=np.float32)
        self.base_xml_path = 'scenes/scene.xml'
        self.maze_xml_path = 'scenes/scene_maze_trial.xml'


        # TODO(学生課題1: 行動空間を R⁴ に拡張 — docs/STUDENT_TASKS_JP.md 参照):
        #   現状はスカラー θ のみの1次元行動。論文 §3 の (R, ω, θ, δ) にするには、
        #   この Box を4次元にし、各成分の low/high を各パラメータ範囲へ広げること。
        #   例: low=[R_lo, ω_lo, θ_lo, δ_lo], high=[R_hi, ω_hi, θ_hi, δ_hi]
        #   (SB3 の SAC は tanh 出力を Box 境界へ自動スケールするので明示スケーリング不要)
        self.action_space = spaces.Box(
            low=np.array([config.env.action_low]),
            high=np.array([config.env.action_high]),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
        self.maze_layout = create_maze_layout(config.env.maze_height, config.env.maze_width)
        make_maze_on_mujoco(
            load_file_path=self.base_xml_path,
            maze=np.array(self.maze_layout),
            start_pos=self.start_pos_maze,
            goal_pos=self.start_pos_maze, # Temporary goal(waypoint)
            save_file_path=self.maze_xml_path
        )

        self.cpg = None  # created in reset(), once model.opt.timestep is known

        self.path_waypoints_world = []
        self.current_waypoint_index = 0
        self.waypoint_threshold = config.reward.waypoint_threshold

        self.reached_waypoints = []

        self.render_mode = render_mode
        self.viewer = None

    def _maze_to_world(self, maze_pos):
        """ maze coordinates in mujoco coordinates"""
        pos_x = 2 * 0.5 * (maze_pos[0] - self.start_pos_maze[0])
        pos_y = -2 * 0.5 * (maze_pos[1] - self.start_pos_maze[1])
        return np.array([pos_x, pos_y, 0.15])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.viewer is not None:
            self.close()
        self.maze_layout = create_maze_layout(self.config.env.maze_height, self.config.env.maze_width)
        self.valid_spawn_points_maze = get_valid_spawn_points(self.maze_layout)
        # Random Goal 
        goal_idx = self.np_random.integers(0, len(self.valid_spawn_points_maze))
        goal_pos_maze = self.valid_spawn_points_maze[goal_idx]
        
        # A* path towards new goal
        path_maze = astar(self.maze_layout, self.start_pos_maze, goal_pos_maze)
        
        # New xml with new goal and waypoints  
        make_maze_on_mujoco(
            load_file_path=self.base_xml_path,
            maze=np.array(self.maze_layout),
            start_pos=self.start_pos_maze,
            goal_pos=goal_pos_maze,
            waypoints=path_maze,
            save_file_path=self.maze_xml_path
        )
        
        self.model = mujoco.MjModel.from_xml_path(self.maze_xml_path)
        self.data = mujoco.MjData(self.model)
        #self.data.qpos[-12:] = self.init_qpos
        #self.data.qpos[3:7] = np.array([1, 0, 0, 0])
        # Waypoint in world coordinate
        self.path_waypoints_world = [self._maze_to_world(p) for p in path_maze]
        #print(f"Path waypoints (world coordinates): {self.path_waypoints_world}")
        self.current_waypoint_index = 0

        # Reset of reached waypoints
        self.reached_waypoints = []

        # Save goal position in world coordinate
        self.goal_pos_world = self._maze_to_world(goal_pos_maze)
        if len(self.path_waypoints_world) > 0:
           self.path_waypoints_world[-1] = self.goal_pos_world.copy()

        # (Re)create the CPG so its phase/amplitude state does not carry over
        # between episodes, with dt matched to the physics timestep so CPG
        # phase and simulated time advance at the same rate.
        self.cpg = PaperCPG(n_joints=12, dt=self.model.opt.timestep)
        self.cpg.set_hyper_parameters(
            convergence_rate=self.config.cpg.convergence_rate,
            coupling_strength=self.config.cpg.coupling_strength,
        )

        # Reset state of snake and CPG
        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        return self._get_obs(), {}
    def _get_current_target(self):
        if self.path_waypoints_world:
            i = self.current_waypoint_index
            if i < len(self.path_waypoints_world) - 1:
                w1 = self.path_waypoints_world[i]
                w2 = self.path_waypoints_world[i + 1]
                return 0.5 * (w1 + w2)
            elif i < len(self.path_waypoints_world):
                return self.path_waypoints_world[i]
        return self.goal_pos_world
    def _get_head_orientation_axis_angle(self):
        """Axis-angle representation (axis(3) + angle(1)) of the head's world
        orientation, analogous to the paper's relative-orientation state."""
        w, x, y, z = self.data.body('frame_0-1').xquat
        w = np.clip(w, -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        sin_half = np.sqrt(max(1.0 - w * w, 0.0))
        if sin_half < 1e-6:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([x, y, z]) / sin_half
        return np.concatenate([axis, [angle]])

    def _get_head_angular_velocity(self):
        """IMU-like signal: angular velocity of the head body (world frame)."""
        return self.data.body('frame_0-1').cvel[:3]

    def _get_obs(self):
        joint_pos = self.data.qpos[-12:]
        joint_vel = self.data.qvel[-12:]
        head_pos = self.data.body('frame_0-1').xpos
        current_target = self._get_current_target()
        head_to_target_vec = current_target - head_pos
        head_orientation = self._get_head_orientation_axis_angle()
        head_angular_vel = self._get_head_angular_velocity()

        obs = np.concatenate([
            joint_pos, joint_vel, head_to_target_vec, head_orientation, head_angular_vel
        ]).astype(np.float32)


        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            raise ValueError(f"[SnakeEnv] Invalid observation detected: {obs}")

        # clip observation to safe range
        obs = np.clip(obs, -1000.0, 1000.0)

        return obs
    def step(self, action):
        """Performs an environment step."""
        self.current_step += 1
        reward_cfg = self.config.reward
        # ------------------------------------------------------------------
        # TODO(学生課題1: 行動空間を R⁴ に拡張 — docs/STUDENT_TASKS_JP.md 参照):
        #   現状は行動がスカラー θ のみで、R・ω・δ は固定値 (self.R / self.omega /
        #   self.delta) を使っている。論文 §3 の行動空間 (R, ω, θ, δ) にするには、
        #   ここで action を4成分に分解して cpg.set_parameters に渡すこと。
        #   例: R, omega, theta, delta = action
        # ------------------------------------------------------------------
        theta = action
        self.cpg.set_parameters(R=self.R, omega = self.omega, theta=theta, delta=self.delta)

        # Retrieve the current waypoint or goal (lookahead target used only
        # for reward shaping, see _get_current_target)
        current_target = self._get_current_target()
        head_pos = self.data.body('frame_0-1').xpos
        distance_to_waypoint = np.linalg.norm(current_target - head_pos)

        joint_range = self.model.jnt_range[-12:]
        for _ in range(self.sim_steps_per_rl_step):
            target_positions = self.cpg.update()
            clipped_targets = np.clip(target_positions, joint_range[:, 0], joint_range[:, 1])
            self.data.ctrl[:] = clipped_targets

            mujoco.mj_step(self.model, self.data)

        new_head_pos = self.data.body('frame_0-1').xpos
        new_distance_to_waypoint = np.linalg.norm(current_target - new_head_pos)
        r2 = distance_to_waypoint - new_distance_to_waypoint

        # Waypoint "reached" is judged against the waypoint's own world
        # coordinates, not the lookahead midpoint used for reward shaping.
        # (Only the waypoint-index advancement is kept here; the one-shot
        # bonus reward that used to be granted is NOT part of the paper's
        # reward eq.(12) and has been removed.)
        if self.current_waypoint_index < len(self.path_waypoints_world):
            actual_waypoint = self.path_waypoints_world[self.current_waypoint_index]
            dist_to_actual_waypoint = np.linalg.norm(actual_waypoint - new_head_pos)
            if dist_to_actual_waypoint < self.waypoint_threshold:
                reached_waypoint = actual_waypoint.copy()

                already_reached = any(np.allclose(reached_waypoint, existing_wp, atol=1e-3)
                                    for existing_wp in self.reached_waypoints)

                if not already_reached:
                    self.reached_waypoints.append(reached_waypoint)
                    self.current_waypoint_index += 1

        # Paper reward, eq.(12): three terms only.
        #   r1: proximity reward (large when close to the target)
        #   r2: closing-speed reward (computed above as distance - new_distance)
        #   r3: action-change penalty (smoothness of the gait transition)
        r1 = 1.0 / (0.1 + new_distance_to_waypoint)
        r3 = np.linalg.norm(action - self.last_action)

        # Goal termination (episode ends on reaching the goal). Note: the paper
        # reward has no explicit goal bonus, so termination is kept but not
        # rewarded here.
        terminated = np.linalg.norm(new_head_pos - self.goal_pos_world) < self.waypoint_threshold

        # ------------------------------------------------------------------
        # TODO(学生課題2: 報酬の重み設計 — docs/STUDENT_TASKS_JP.md 参照):
        #   下の3重み (w_progress, w_velocity, w_smoothness) は config/default.yaml
        #   で調整する。r3 は罰則なので減算している (論文 §4.2: w1·r1 + w2·r2 − w3·r3)。
        # ------------------------------------------------------------------
        reward = (
            reward_cfg.w_progress * r1
            + reward_cfg.w_velocity * r2
            - reward_cfg.w_smoothness * r3
        )

        truncated = self.current_step >= self.max_episode_steps
        self.last_action = action

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, bool(terminated), bool(truncated), {}

    def render(self):
        if self.viewer is None and self.data is not None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
            time.sleep(1.0 / self.metadata['render_fps'])
        elif self.viewer:
            self.close()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None






