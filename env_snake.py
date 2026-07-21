import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import time
from mazes.make_maze import create_maze_layout, get_valid_spawn_points, astar, cell_distances
from mazes.mujoco_tools import make_maze_on_mujoco
from cpg.snake_cpg import PaperCPG
from config_utils import load_config

# Every episode's goal is exactly this many A* moves (edges) from the start, so
# episode difficulty (path length) is constant and training is predictable. If a
# freshly generated maze has no cell at this exact distance, reset() regenerates the
# maze (up to MAX_MAZE_ATTEMPTS times); ~99% of 10x10 mazes qualify on the first try.
GOAL_PATH_LENGTH = 30
MAX_MAZE_ATTEMPTS = 1000

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, render_mode=None, config=None, render_slowdown=None,
                 env_index=None, maze_xml_path=None):
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
        # The generated maze XML must live in scenes/ because it <include>s
        # snake.xml via a path relative to its own directory. For parallel
        # (SubprocVecEnv / DummyVecEnv) training every env needs a UNIQUE output
        # file, or workers clobber each other's XML mid-write. Resolution order:
        #   1. explicit maze_xml_path arg;
        #   2. env_index set -> scenes/scene_maze_env{index}_{pid}.xml
        #      (index disambiguates DummyVecEnv, pid disambiguates SubprocVecEnv
        #      workers and concurrent training runs). Built HERE in __init__ so
        #      os.getpid() is the worker's pid under spawn, not the parent's;
        #   3. default shared path (single-env / test_model.py, unchanged).
        self._env_index = env_index
        if maze_xml_path is not None:
            self.maze_xml_path = maze_xml_path
        elif env_index is not None:
            self.maze_xml_path = f'scenes/scene_maze_env{env_index}_{os.getpid()}.xml'
        else:
            self.maze_xml_path = 'scenes/scene_maze_trial.xml'


        # 学生課題1: 行動空間 R⁴ = (R, ω, θ, δ). 各成分の low/high は
        # config/default.yaml の action_low / action_high (順: [R, omega, theta, delta])。
        # SB3 の SAC は tanh 出力を Box 境界へ自動スケールするので明示スケーリング不要。
        self.action_space = spaces.Box(
            low=np.array(config.env.action_low, dtype=np.float32),
            high=np.array(config.env.action_high, dtype=np.float32),
            dtype=np.float32,
        )

        # Number of actuated snake joints (Actuator1..12). The model also has
        # passive free-spinning wheel joints interleaved with these, so joint
        # state must be looked up by name, never via a qpos[-12:] slice (see
        # _resolve_actuated_joints / _get_obs).
        self.n_actuated = 12
        act_dim = int(self.action_space.shape[0])
        # Observation layout (see _get_obs):
        #   actuated joint pos      : n_actuated (12)
        #   actuated joint vel      : n_actuated (12)
        #   head->target vector xy  : 2   (z dropped: planar maze, ~constant)
        #   head orientation        : 2   (az, angle; ax/ay dropped: ~0 planar)
        #   head angular velocity   : 3
        #   heading error (cos, sin): 2   (head-forward vs target dir in xy plane)
        #   last action             : act_dim (makes the r3 smoothness penalty Markovian)
        obs_dim = 2 * self.n_actuated + 2 + 2 + 3 + 2 + act_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
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
        # Playback speed for the human viewer. 1.0 = real physics time
        # (each substep drawn and paused by one physics timestep). Increase
        # to slow the animation down further (e.g. 4.0 = 4x slower).
        # Resolution order: explicit render_slowdown arg > config value > 2.0.
        if render_slowdown is not None:
            self.render_slowdown = render_slowdown
        else:
            self.render_slowdown = getattr(config.env, "render_slowdown", 2.0)
        # Current CPG parameters, shown as a text overlay in the viewer.
        self._display_params = {"R": self.R, "omega": self.omega,
                                "theta": 0.0, "delta": self.delta}

    def _maze_to_world(self, maze_pos):
        """ maze coordinates in mujoco coordinates"""
        pos_x = 2 * 0.5 * (maze_pos[0] - self.start_pos_maze[0])
        pos_y = -2 * 0.5 * (maze_pos[1] - self.start_pos_maze[1])
        return np.array([pos_x, pos_y, 0.15])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.viewer is not None:
            self.close()
        # Regenerate the maze until it contains a goal that is EXACTLY
        # GOAL_PATH_LENGTH A* moves (edges) from the start, so every episode has the
        # same path length and difficulty. cell_distances() is a single BFS from the
        # start (the maze is a perfect maze / tree, so BFS distance == A* path length).
        # (Supersedes the old min_goal_distance Manhattan filter, which is now unused.)
        self.maze_layout = None
        goal_candidates = []
        for _ in range(MAX_MAZE_ATTEMPTS):
            self.maze_layout = create_maze_layout(self.config.env.maze_height, self.config.env.maze_width)
            dist = cell_distances(self.maze_layout, self.start_pos_maze)
            goal_candidates = [p for p, d in dist.items() if d == GOAL_PATH_LENGTH]
            if goal_candidates:
                break
        else:
            raise RuntimeError(
                f"No cell exactly {GOAL_PATH_LENGTH} A* moves from start after "
                f"{MAX_MAZE_ATTEMPTS} maze regenerations."
            )
        self.valid_spawn_points_maze = get_valid_spawn_points(self.maze_layout)
        goal_idx = self.np_random.integers(0, len(goal_candidates))
        goal_pos_maze = goal_candidates[goal_idx]

        # A* path towards new goal (len(path_maze) - 1 == GOAL_PATH_LENGTH by construction)
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
        # The model is rebuilt every reset(), so (re)resolve which qpos/qvel
        # entries belong to the actuated joints.
        self._resolve_actuated_joints()
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
            # Clamp: once the final waypoint (== goal) is reached,
            # current_waypoint_index advances to len(path_waypoints_world) on the
            # same step that sets terminated=True. That terminated step still calls
            # _get_obs()->_get_current_target(), so an unclamped index would raise
            # IndexError before the episode can return. Keep pointing at the goal.
            i = min(self.current_waypoint_index, len(self.path_waypoints_world) - 1)
            return self.path_waypoints_world[i]
        return self.goal_pos_world
    def _resolve_actuated_joints(self):
        """Cache the qpos/qvel/range addresses of the 12 actuated hinge joints
        (Actuator1..12).

        The snake model interleaves each actuated joint with a passive,
        free-spinning wheel joint (frameN-2, axis 0 1 0, unlimited). MuJoCo lays
        the joints out in DFS order as Actuator_k, wheel_k, Actuator_{k+1}, ...,
        so a plain qpos[-12:] slice does NOT correspond to the actuated joints:
        it mixes in wheels (whose angle is an unbounded integrator, hundreds of
        radians) and silently drops the first actuators. Resolve by name instead."""
        ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"Actuator{i}")
               for i in range(1, self.n_actuated + 1)]
        if any(j < 0 for j in ids):
            raise RuntimeError("SnakeEnv: could not find all Actuator1..12 joints "
                               "in the model; check scenes/snake.xml joint names.")
        self._act_qpos_adr = self.model.jnt_qposadr[ids].copy()
        self._act_dof_adr = self.model.jnt_dofadr[ids].copy()
        self._act_jnt_range = self.model.jnt_range[ids].copy()  # (12, 2), each [-3, 3]

    def _get_heading_error(self, head_to_target_vec):
        """Heading error between the head's forward direction and the direction
        to the current target, in the ground (xy) plane.

        Returned as (cos e, sin e) rather than the raw angle e to avoid the
        +/-pi wraparound discontinuity. sin e > 0 means the target is to the
        snake's LEFT. This pre-computes the "which way do I steer" signal that
        the policy would otherwise have to infer by combining the world-frame
        target vector with the head orientation.

        Forward is the head body's local -x axis (the head's initial quat is a
        180 deg yaw about z), i.e. -xmat[:, 0] in world coordinates."""
        R = self.data.body('frame_0-1').xmat.reshape(3, 3)
        fwd_xy = (-R[:, 0])[:2]
        tgt_xy = head_to_target_vec[:2]
        fn = np.linalg.norm(fwd_xy)
        tn = np.linalg.norm(tgt_xy)
        if fn < 1e-9 or tn < 1e-9:
            return np.array([1.0, 0.0])  # degenerate (target under head): treat as aligned
        fwd_xy = fwd_xy / fn
        tgt_xy = tgt_xy / tn
        cos_e = float(np.dot(fwd_xy, tgt_xy))
        sin_e = float(fwd_xy[0] * tgt_xy[1] - fwd_xy[1] * tgt_xy[0])  # 2D cross; + = left
        return np.array([cos_e, sin_e])

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
        joint_pos = self.data.qpos[self._act_qpos_adr]      # 12 actuated joint angles
        joint_vel = self.data.qvel[self._act_dof_adr]       # 12 actuated joint velocities
        head_pos = self.data.body('frame_0-1').xpos
        current_target = self._get_current_target()
        head_to_target_vec = current_target - head_pos
        # Planar maze: the z component of the head->target vector is ~constant
        # (both pinned near z=0.15), so keep only the (x, y) offset.
        head_to_target_xy = head_to_target_vec[:2]
        # Head orientation as axis-angle; for planar motion the rotation axis is
        # ~vertical (ax, ay ~ 0), so keep only the z-axis component and the angle.
        head_orientation = self._get_head_orientation_axis_angle()  # [ax, ay, az, angle]
        orient_z_angle = head_orientation[2:]                       # [az, angle]
        head_angular_vel = self._get_head_angular_velocity()
        heading_error = self._get_heading_error(head_to_target_vec)  # [cos e, sin e]

        obs = np.concatenate([
            joint_pos, joint_vel, head_to_target_xy, orient_z_angle,
            head_angular_vel, heading_error, np.ravel(self.last_action)
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
        # 学生課題1: 行動空間 R⁴. action = [R, omega, theta, delta] を分解して
        # CPG に渡す (以前は theta のみ学習し R/omega/delta は固定だった)。
        R, omega, theta, delta = (float(x) for x in np.ravel(action))
        self.cpg.set_parameters(R=R, omega=omega, theta=theta, delta=delta)

        # Remember the CPG parameters currently driving the gait so the viewer
        # overlay can display them.
        self._display_params = {
            "R": R,
            "omega": omega,
            "theta": theta,
            "delta": delta,
        }

        # Retrieve the current waypoint or goal (lookahead target used only
        # for reward shaping, see _get_current_target)
        current_target = self._get_current_target()
        head_pos = self.data.body('frame_0-1').xpos
        distance_to_waypoint = np.linalg.norm(current_target - head_pos)

        # Clip the 12 CPG targets to the actuated joints' ranges. NOTE: use the
        # by-name actuator ranges, NOT jnt_range[-12:] — the latter is interleaved
        # with unlimited wheel joints whose range is [0, 0], which would clamp
        # half the actuator targets to zero and cripple the gait.
        act_lo = self._act_jnt_range[:, 0]
        act_hi = self._act_jnt_range[:, 1]
        for _ in range(self.sim_steps_per_rl_step):
            target_positions = self.cpg.update()
            clipped_targets = np.clip(target_positions, act_lo, act_hi)
            self.data.ctrl[:] = clipped_targets

            mujoco.mj_step(self.model, self.data)

            # Draw every physics substep so the motion is smooth and slow
            # enough to follow, instead of jumping once per RL step.
            if self.render_mode == "human":
                self.render()

        new_head_pos = self.data.body('frame_0-1').xpos
        new_distance_to_waypoint = np.linalg.norm(current_target - new_head_pos)
        r2 = distance_to_waypoint - new_distance_to_waypoint
        r2 = np.sign(r2) * r2 * r2
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
        term_progress = reward_cfg.w_progress * r1
        term_velocity = reward_cfg.w_velocity * r2
        term_smoothness = reward_cfg.w_smoothness * r3  # penalty (subtracted below)
        # DELIBERATE DIVERGENCE from the paper's eq.(12) (see RewardConfig):
        #   term_goal: one-shot terminal bonus for reaching the goal. Combined
        #     with gamma<1 this makes finishing SOONER worth more, i.e. it is
        #     what actually rewards speed (r2 alone only rewards net progress).
        #   term_time: per-step living cost, a direct pressure to finish fast.
        term_goal = reward_cfg.w_goal * (1.0 if terminated else 0.0)
        term_time = reward_cfg.time_penalty  # subtracted every step
        reward = (term_progress + term_velocity - term_smoothness
                  + term_goal - term_time)

        truncated = self.current_step >= self.max_episode_steps
        # Stored as a flat float32 vector: it is fed back into the observation
        # (see _get_obs) so the policy can see the action it just took.
        self.last_action = np.asarray(action, dtype=np.float32).ravel()

        # Expose the per-term breakdown so training can log/monitor the balance
        # between the three reward components (see RewardTermCallback in
        # train_sac.py). Raw (unweighted) values are included too.
        info = {
            "reward_total": float(reward),
            "term_progress": float(term_progress),
            "term_velocity": float(term_velocity),
            "term_smoothness": float(term_smoothness),
            "term_goal": float(term_goal),
            "term_time": float(-term_time),  # stored signed, since it is a penalty
            "raw_r1_proximity": float(r1),
            "raw_r2_closing": float(r2),
            "raw_r3_action_delta": float(r3),
            # Whether this episode ended by reaching the goal (terminated) vs
            # timing out at max_episode_steps (truncated). Read at episode end
            # by RewardTermCallback to log the goal-reached (finish) rate: with
            # a short max_episode_steps many episodes truncate before finishing,
            # so ep_rew_mean alone doesn't tell you how often the goal is hit.
            "goal_reached": bool(terminated),
        }

        return self._get_obs(), reward, bool(terminated), bool(truncated), info

    def _draw_param_overlay(self):
        """Draw a text label showing the CPG parameters currently driving the
        gait, pinned to the bottom-center of the camera window (it does not
        track the snake). The label geom is placed relative to the current
        camera pose so it stays at the bottom of the view as the camera moves.
        """
        scn = self.viewer.user_scn
        scn.ngeom = 0
        p = self._display_params
        label = (f"R={p['R']:.2f}  omega={p['omega']:.2f}  "
                 f"theta={p['theta']:.3f}  delta={p['delta']:.2f}")

        # Camera frame from the free-camera spherical parameters.
        cam = self.viewer.cam
        az = np.deg2rad(cam.azimuth)
        el = np.deg2rad(cam.elevation)
        ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
        forward = -np.array([ce * ca, ce * sa, se])   # camera -> lookat
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)
        cam_up /= np.linalg.norm(cam_up)

        lookat = np.array(cam.lookat)
        dist = max(float(cam.distance), 0.1)
        cam_pos = lookat - dist * forward
        fovy = np.deg2rad(self.model.vis.global_.fovy)
        # Sit at the lookat depth, pushed ~85% of the way to the bottom edge
        # of the vertical field of view so the text hugs the bottom of the view.
        pos = (cam_pos + forward * dist
             + right * (dist * np.tan(fovy * 0.5) * 0.62)
             - cam_up * (dist * np.tan(fovy * 0.5) * 0.72))

        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.008, 0.0, 0.0]),
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.9, 0.2, 0.9], dtype=np.float32),
        )
        geom.label = label
        scn.ngeom += 1

        # Draw a red sphere at the next waypoint so it is easy to see which
        # waypoint the policy is currently trying to reach.
        if self.current_waypoint_index < len(self.path_waypoints_world):
            waypoint = self.path_waypoints_world[self.current_waypoint_index]
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.05, 0.0, 0.0]),
                pos=waypoint,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            )
            scn.ngeom += 1

    def _set_tracking_camera(self):
        """Make the free camera follow the snake's head automatically, so the
        snake stays centered without needing to pan the camera with a mouse.
        Re-applied whenever the viewer is (re)created (once per episode)."""
        cam = self.viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.model.body('frame_0-1').id
        cam.distance = 6.0
        cam.azimuth = 90.0
        cam.elevation = -70.0

    def render(self):
        if self.viewer is None and self.data is not None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._set_tracking_camera()
        if self.viewer and self.viewer.is_running():
            self._draw_param_overlay()
            self.viewer.sync()
            # Pause by one physics timestep (scaled by render_slowdown) so the
            # animation plays at real time when render_slowdown == 1.0, and
            # slower for larger values.
            time.sleep(self.model.opt.timestep * self.render_slowdown)
        elif self.viewer:
            self.close()

    def close(self):
        if self.viewer:
            self.viewer.close()
        # Remove this env's per-index maze XML so scenes/ doesn't accumulate
        # one file per worker per run. Only touch files we own (env_index set);
        # never delete the shared default used by single-env / test_model.py.
        if self._env_index is not None:
            try:
                os.remove(self.maze_xml_path)
            except OSError:
                pass
            self.viewer = None






