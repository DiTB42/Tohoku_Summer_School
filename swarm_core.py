"""Multi-snake swarm visualization engine.

Assembles ONE MuJoCo model containing many name-prefixed copies of the snake
(``scenes/snake.xml``) sharing a single generated maze, then drives them all in
one physics loop and renders them in one live viewer, each a distinct color.
Used by:

  * ``viz_stochastic.py``  -- N snakes from ONE checkpoint, sampling actions
    stochastically, so they fan out and reveal the policy's behavior spread.
  * ``viz_checkpoints.py`` -- one snake per training checkpoint, deterministic,
    so you see how the policy improves over training.

Why a new tool instead of SnakeEnv? ``env_snake.py`` is single-snake: it rebuilds
a one-snake MjModel from a maze XML every reset(), and ``scenes/snake.xml`` uses
hard-coded, non-namespaced names (frame_0-1, Actuator1..12), so two snakes in one
model collide on names. This module gives every snake a ``s{k:02d}_`` prefix and
merges them into one scene via ElementTree (the same XML-surgery idiom as
``mazes/mujoco_tools.make_maze_on_mujoco``).

This module does NOT modify training. The observation builder here mirrors
``env_snake.SnakeEnv._get_obs`` EXACTLY (37-D) -- if that layout ever changes,
update ``SnakeState.get_obs`` here to match, or predictions become garbage.
"""

import os
import copy
import time
import colorsys
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import mujoco.viewer

from mazes.make_maze import create_maze_layout, get_valid_spawn_points, astar
from mazes.mujoco_tools import make_maze_on_mujoco
from cpg.snake_cpg import PaperCPG
from config_utils import load_config

# Repo-relative paths (this file lives at the repo root).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_NOSNAKE_XML = os.path.join(_REPO_ROOT, "scenes", "scene_nosnake.xml")
_SNAKE_XML = os.path.join(_REPO_ROOT, "scenes", "snake.xml")

# Scratch dir for the generated combined XML (kept out of scenes/ so it doesn't
# pollute git, mirroring env_snake's cleanup discipline for per-worker files).
_SCRATCH = os.environ.get(
    "SNAKE_SWARM_SCRATCH",
    os.path.join(
        r"C:\Users\TEST-2~1\AppData\Local\Temp\claude",
        "c--Users-test-2026-2-snake-RL-2-Tohoku-Summer-School",
        "swarm",
    ),
)

START_POS_MAZE = (1, 1)  # same fixed spawn cell as SnakeEnv.start_pos_maze


# ---------------------------------------------------------------------------
# Coordinate + color helpers
# ---------------------------------------------------------------------------
def maze_to_world(maze_pos, start_pos=START_POS_MAZE):
    """maze (col,row) -> world (x,y,z). Mirrors SnakeEnv._maze_to_world.

    x = (col - start_col), y = -(row - start_row) [flipped y], z = 0.15."""
    pos_x = 2 * 0.5 * (maze_pos[0] - start_pos[0])
    pos_y = -2 * 0.5 * (maze_pos[1] - start_pos[1])
    return np.array([pos_x, pos_y, 0.15])


def distinct_colors(n):
    """n visually distinct RGB colors by evenly spacing hue (no matplotlib dep)."""
    cols = []
    for i in range(n):
        h = (i / max(n, 1)) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        cols.append((r, g, b))
    return cols


# ---------------------------------------------------------------------------
# Shared maze
# ---------------------------------------------------------------------------
def build_shared_maze(config, seed=None):
    """Generate one maze + goal + A* path, shared by every snake in the run.

    Mirrors the maze logic in SnakeEnv.reset() but fixed for the whole run
    (SnakeEnv regenerates a new maze every reset; here all snakes share one).
    Returns a dict with maze layout, goal cell, and world-space path/goal.
    """
    import random
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    maze = create_maze_layout(config.env.maze_height, config.env.maze_width)
    valid = get_valid_spawn_points(maze)
    sx, sy = START_POS_MAZE
    min_dist = config.env.min_goal_distance
    cands = [p for p in valid if abs(p[0] - sx) + abs(p[1] - sy) >= min_dist]
    if not cands:
        cands = [p for p in valid if tuple(p) != START_POS_MAZE] or valid
    goal_pos = cands[np.random.randint(0, len(cands))]

    path_maze = astar(maze, START_POS_MAZE, goal_pos)
    path_world = [maze_to_world(p) for p in path_maze]
    goal_world = maze_to_world(goal_pos)
    if path_world:
        path_world[-1] = goal_world.copy()  # snap final waypoint to goal (as env does)

    return {
        "maze": maze,
        "goal_pos": goal_pos,
        "goal_world": goal_world,
        "path_maze": path_maze,
        "path_world": path_world,
    }


# ---------------------------------------------------------------------------
# Combined N-snake scene assembly (ElementTree prefixing)
# ---------------------------------------------------------------------------
def _prefix_snake_subtree(elem, prefix, rgba):
    """In-place: prefix every named element in a snake body subtree, and set
    each geom's color + collision mask.

    Names prefixed: body/joint/geom/camera/site/light 'name' attributes.
    (In snake.xml only bodies, the Actuator* joints, and the 'track' camera are
    named; geoms and the free/wheel joints are unnamed.) Geoms get:
      * rgba   -> the snake's color (so each snake is one solid color)
      * contype=2, conaffinity=1 -> disables snake<->snake AND snake self
        contacts, while snake<->wall/floor (env default 1/1) still fire.
        Bitmask: snake-snake (2&1)|(2&1)=0; snake-floor (2&1)|(1&1)=1.
    """
    rgba_str = f"{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} 1"
    for node in elem.iter():
        if "name" in node.attrib:
            node.set("name", prefix + node.get("name"))
        if node.tag == "geom":
            node.set("rgba", rgba_str)
            node.set("contype", "2")
            node.set("conaffinity", "1")


def build_combined_scene(maze_info, prefixes, colors, out_path, jitter=0.0):
    """Write a combined XML: maze-only base + N prefixed, colored snakes.

    prefixes[k] / colors[k] describe snake k. Returns out_path.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 1. Maze-only scene (walls + goal + waypoint spheres) from the snake-free
    #    base. make_maze_on_mujoco appends into <worldbody>.
    maze_only = out_path + ".maze.xml"
    make_maze_on_mujoco(
        load_file_path=_NOSNAKE_XML,
        maze=np.array(maze_info["maze"]),
        start_pos=list(START_POS_MAZE),
        goal_pos=list(maze_info["goal_pos"]),
        waypoints=maze_info["path_maze"],
        save_file_path=maze_only,
    )

    combined = ET.parse(maze_only)
    root = combined.getroot()
    worldbody = root.find("./worldbody")
    actuator_block = root.find("./actuator")
    if actuator_block is None:
        actuator_block = ET.SubElement(root, "actuator")

    # 2. Snake templates: the frame_0-1 body subtree + the 12 <position> actuators.
    snake_tree = ET.parse(_SNAKE_XML)
    snake_root = snake_tree.getroot()
    head_template = snake_root.find('./worldbody/body[@name="frame_0-1"]')
    act_templates = snake_root.findall("./actuator/position")
    if head_template is None or len(act_templates) != 12:
        raise RuntimeError("swarm_core: unexpected snake.xml structure "
                           "(missing frame_0-1 body or 12 actuators).")

    for k, (prefix, color) in enumerate(zip(prefixes, colors)):
        head = copy.deepcopy(head_template)
        # Optional tiny lateral spawn jitter so overlapping snakes are less
        # visually degenerate at t=0 (off by default: divergence should come
        # from the policy, not the initial condition).
        if jitter:
            base = [float(v) for v in head.get("pos", "0.5 0 0").split()]
            base[1] += (k - (len(prefixes) - 1) / 2.0) * jitter
            head.set("pos", f"{base[0]} {base[1]} {base[2]}")
        _prefix_snake_subtree(head, prefix, color)
        worldbody.append(head)

        for act in act_templates:
            a = copy.deepcopy(act)
            a.set("name", prefix + a.get("name"))
            a.set("joint", prefix + a.get("joint"))
            actuator_block.append(a)

    combined.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


# ---------------------------------------------------------------------------
# Per-snake state
# ---------------------------------------------------------------------------
class SnakeState:
    """Handles + rollout state for one snake instance in the combined model."""

    N_ACT = 12

    def __init__(self, model, prefix, color, label, model_policy, deterministic,
                 config):
        self.prefix = prefix
        self.color = color
        self.label = label
        self.policy = model_policy        # SB3 SAC (may be shared across snakes)
        self.deterministic = deterministic
        self.head_name = f"{prefix}frame_0-1"

        # Adapt the observation layout to THIS policy's obs space. Checkpoints in
        # this repo come from two eras (see docs / README student-task 1):
        #   * 37-D obs: full layout incl. the 3-D head angular velocity block.
        #   * 34-D obs: older layout WITHOUT the angular-velocity block.
        # Every checkpoint here uses a 4-D action (R, omega, theta, delta); the
        # scalar-theta era (act_dim=1) is not supported by the CPG mapping below.
        self.act_low = np.asarray(model_policy.action_space.low, dtype=np.float32)
        self.act_high = np.asarray(model_policy.action_space.high, dtype=np.float32)
        self.act_dim = int(model_policy.action_space.shape[0])
        if self.act_dim != 4:
            raise RuntimeError(
                f"swarm_core: policy for {label!r} has action dim {self.act_dim}, "
                f"but only the 4-D (R,omega,theta,delta) action space is supported.")
        obs_dim = int(model_policy.observation_space.shape[0])
        base_no_angvel = 2 * self.N_ACT + 2 + 2 + 2 + self.act_dim  # 34 for act=4
        if obs_dim == base_no_angvel + 3:
            self.include_angvel = True
        elif obs_dim == base_no_angvel:
            self.include_angvel = False
        else:
            raise RuntimeError(
                f"swarm_core: policy for {label!r} has obs dim {obs_dim}, "
                f"expected {base_no_angvel} or {base_no_angvel + 3}; obs layout "
                f"does not match env_snake._get_obs.")

        # Resolve actuated joint qpos/dof/range addresses by prefixed name,
        # exactly like SnakeEnv._resolve_actuated_joints (never a qpos slice:
        # wheels are interleaved).
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                  f"{prefix}Actuator{i}")
                for i in range(1, self.N_ACT + 1)]
        aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                  f"{prefix}Actuator{i}")
                for i in range(1, self.N_ACT + 1)]
        if any(j < 0 for j in jids) or any(a < 0 for a in aids):
            raise RuntimeError(f"swarm_core: could not resolve joints/actuators "
                               f"for prefix {prefix!r}.")
        self.qpos_adr = model.jnt_qposadr[jids].copy()
        self.dof_adr = model.jnt_dofadr[jids].copy()
        self.jnt_range = model.jnt_range[jids].copy()
        self.ctrl_idx = np.array(aids, dtype=int)   # actuator id == ctrl index

        # DOFs of the head's free joint (6: 3 translation + 3 rotation). Zeroed
        # once when the snake reaches the goal so its base momentum doesn't carry
        # it past the goal after freezing.
        head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.head_name)
        free_jid = model.body_jntadr[head_bid]
        free_dof0 = model.jnt_dofadr[free_jid]
        self.free_dof_adr = np.arange(free_dof0, free_dof0 + 6)

        # CPG (dt = physics timestep, matched as env does).
        self.cpg = PaperCPG(n_joints=self.N_ACT, dt=model.opt.timestep)
        self.cpg.set_hyper_parameters(
            convergence_rate=config.cpg.convergence_rate,
            coupling_strength=config.cpg.coupling_strength,
        )

        self.last_action = np.zeros(self.act_dim, dtype=np.float32)
        self.waypoint_index = 0
        self.done = False
        # Joint targets to hold once the snake reaches the goal, so it freezes in
        # place instead of the CPG continuing to wave its joints (which would
        # keep it crawling). Captured at the moment `done` flips True.
        self.frozen_ctrl = None

    # --- observation (mirrors env_snake.SnakeEnv._get_obs, 37-D) -----------
    def _heading_error(self, data, head_to_target_vec):
        R = data.body(self.head_name).xmat.reshape(3, 3)
        fwd_xy = (-R[:, 0])[:2]
        tgt_xy = head_to_target_vec[:2]
        fn, tn = np.linalg.norm(fwd_xy), np.linalg.norm(tgt_xy)
        if fn < 1e-9 or tn < 1e-9:
            return np.array([1.0, 0.0])
        fwd_xy, tgt_xy = fwd_xy / fn, tgt_xy / tn
        cos_e = float(np.dot(fwd_xy, tgt_xy))
        sin_e = float(fwd_xy[0] * tgt_xy[1] - fwd_xy[1] * tgt_xy[0])
        return np.array([cos_e, sin_e])

    def _orient_z_angle(self, data):
        w, x, y, z = data.body(self.head_name).xquat
        w = np.clip(w, -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        sin_half = np.sqrt(max(1.0 - w * w, 0.0))
        az = z / sin_half if sin_half >= 1e-6 else 0.0
        return np.array([az, angle])

    def current_target(self, path_world, goal_world):
        if path_world:
            i = min(self.waypoint_index, len(path_world) - 1)
            return path_world[i]
        return goal_world

    def get_obs(self, data, path_world, goal_world):
        joint_pos = data.qpos[self.qpos_adr]
        joint_vel = data.qvel[self.dof_adr]
        head_pos = data.body(self.head_name).xpos
        target = self.current_target(path_world, goal_world)
        head_to_target = target - head_pos
        head_to_target_xy = head_to_target[:2]
        orient = self._orient_z_angle(data)                  # [az, angle]
        heading_err = self._heading_error(data, head_to_target)  # [cos, sin]
        parts = [joint_pos, joint_vel, head_to_target_xy, orient]
        if self.include_angvel:
            parts.append(data.body(self.head_name).cvel[:3])  # head angular velocity
        parts += [heading_err, np.ravel(self.last_action)]
        obs = np.concatenate(parts).astype(np.float32)
        return np.clip(obs, -1000.0, 1000.0)


# ---------------------------------------------------------------------------
# Swarm runner
# ---------------------------------------------------------------------------
class SnakeSwarm:
    def __init__(self, specs, config, seed=None,
                 render_slowdown=2.0, show_labels=True, jitter=0.0,
                 scratch_dir=None):
        """specs: list of dicts, one per snake, each with keys:
              policy       -- SB3 SAC model (may be shared)
              deterministic-- bool
              label        -- str (shown as a floating tag if show_labels)
              color        -- optional (r,g,b); auto-assigned if omitted
        """
        self.config = config
        self.render_slowdown = render_slowdown
        self.show_labels = show_labels
        self.waypoint_threshold = config.reward.waypoint_threshold
        self.sim_steps = int(config.env.cpg_frequency / config.env.rl_frequency)

        n = len(specs)
        auto = distinct_colors(n)
        colors = [s.get("color") or auto[i] for i, s in enumerate(specs)]
        prefixes = [f"s{i:02d}_" for i in range(n)]

        self.maze_info = build_shared_maze(config, seed=seed)

        scratch = scratch_dir or _SCRATCH
        self.combined_xml = os.path.join(scratch, "swarm_combined.xml")
        build_combined_scene(self.maze_info, prefixes, colors, self.combined_xml,
                             jitter=jitter)

        self.model = mujoco.MjModel.from_xml_path(self.combined_xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.snakes = [
            SnakeState(self.model, prefixes[i], colors[i], specs[i]["label"],
                       specs[i]["policy"], specs[i]["deterministic"], config)
            for i in range(n)
        ]
        self.viewer = None

    # --- prediction: batch snakes that share the same policy object --------
    def _predict_all(self):
        obs = [s.get_obs(self.data, self.maze_info["path_world"],
                         self.maze_info["goal_world"]) for s in self.snakes]
        actions = [None] * len(self.snakes)
        groups = {}
        for i, s in enumerate(self.snakes):
            if s.done:
                actions[i] = s.last_action
                continue
            groups.setdefault((id(s.policy), s.deterministic), []).append(i)
        for (_, det), idxs in groups.items():
            policy = self.snakes[idxs[0]].policy
            batch = np.stack([obs[i] for i in idxs])
            acts, _ = policy.predict(batch, deterministic=det)
            for j, i in enumerate(idxs):
                s = self.snakes[i]
                actions[i] = np.clip(acts[j], s.act_low, s.act_high)
        return actions

    def _advance_targets(self):
        """Advance each snake's waypoint index / done flag (mirrors env logic)."""
        path = self.maze_info["path_world"]
        goal = self.maze_info["goal_world"]
        for s in self.snakes:
            if s.done:
                continue
            head = self.data.body(s.head_name).xpos
            if s.waypoint_index < len(path):
                if np.linalg.norm(path[s.waypoint_index] - head) < self.waypoint_threshold:
                    s.waypoint_index += 1
            if np.linalg.norm(head - goal) < self.waypoint_threshold:
                s.done = True
                # Freeze the servos at the current joint angles so the snake
                # holds its pose and stops crawling, and kill its base + joint
                # momentum once so it doesn't coast past the goal.
                s.frozen_ctrl = self.data.qpos[s.qpos_adr].copy()
                self.data.qvel[s.free_dof_adr] = 0.0
                self.data.qvel[s.dof_adr] = 0.0

    # --- rendering ----------------------------------------------------------
    def _apply_camera(self, cam):
        """Configure a MjvCamera as a top-down free camera framing the whole maze
        (not the env's per-head tracking). Shared by the live viewer and the
        offscreen MP4 recorder."""
        maze = np.array(self.maze_info["maze"])
        h, w = maze.shape
        corners = [maze_to_world((0, 0)), maze_to_world((w - 1, h - 1))]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        extent = max(max(xs) - min(xs), max(ys) - min(ys))
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [cx, cy, 0.0]
        cam.distance = extent * 1.15 + 2.0
        cam.azimuth = 90.0
        cam.elevation = -89.0

    def _draw_markers(self, scn, append=False):
        """Add the goal marker + per-snake label tags to a scene.

        append=False (live user_scn, an initially-empty overlay scene): reset
        ngeom to 0 first. append=True (the offscreen renderer.scene, already
        populated with the model's geoms): append after the existing geoms."""
        if not append:
            scn.ngeom = 0
        maxg = scn.maxgeom

        def add(gtype, size, pos, rgba, label=""):
            if scn.ngeom >= maxg:
                return False
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, type=gtype,
                                size=np.asarray(size, dtype=np.float64),
                                pos=np.asarray(pos, dtype=np.float64),
                                mat=np.eye(3).flatten(),
                                rgba=np.asarray(rgba, dtype=np.float32))
            if label:
                g.label = label
            scn.ngeom += 1
            return True

        # Goal marker (bright red sphere).
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.15, 0, 0],
            self.maze_info["goal_world"], [1.0, 0.1, 0.1, 1.0], "GOAL")

        # Optional floating label tag above each snake head (color matches snake).
        if self.show_labels:
            for s in self.snakes:
                head = self.data.body(s.head_name).xpos
                add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0],
                    [head[0], head[1], head[2] + 0.25],
                    [s.color[0], s.color[1], s.color[2], 1.0], s.label)

    def _draw_overlay(self):
        self._draw_markers(self.viewer.user_scn, append=False)

    def _render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._apply_camera(self.viewer.cam)
        if self.viewer.is_running():
            self._draw_overlay()
            self.viewer.sync()
            time.sleep(self.model.opt.timestep * self.render_slowdown)
            return True
        return False

    # --- offscreen MP4 recording -------------------------------------------
    def _init_recording(self, video_res, video_fps, video_seconds, max_steps,
                        capture_every):
        """Set up an offscreen renderer + camera and work out the substep capture
        cadence. Returns (renderer, cam, capture_every)."""
        w, h = video_res
        # Clamp to the model's offscreen framebuffer (set in scene_nosnake.xml),
        # and to even/macro-block-friendly sizes for the H.264 encoder.
        ow = int(self.model.vis.global_.offwidth)
        oh = int(self.model.vis.global_.offheight)
        w, h = min(w, ow), min(h, oh)
        w -= w % 16
        h -= h % 16
        if (w, h) != tuple(video_res):
            print(f"[swarm] video resolution adjusted to {w}x{h} "
                  f"(offscreen buffer {ow}x{oh}, H.264-friendly).")
        renderer = mujoco.Renderer(self.model, height=h, width=w)
        cam = mujoco.MjvCamera()
        self._apply_camera(cam)
        if capture_every is None:
            total_sub = max_steps * self.sim_steps
            n_target = max(1, int(round(video_fps * video_seconds)))
            capture_every = max(1, round(total_sub / n_target))
        return renderer, cam, capture_every

    def _encode(self, path, frames, video_fps):
        if not frames:
            print("[swarm] no frames captured; nothing to write.")
            return
        try:
            import imageio.v2 as imageio
        except ImportError:
            raise SystemExit(
                "[swarm] MP4 recording needs imageio + ffmpeg. Install with:\n"
                "  uv pip install \"imageio[ffmpeg]\"")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        imageio.mimwrite(path, frames, fps=video_fps, quality=8)
        print(f"[swarm] wrote {len(frames)} frames -> {path} "
              f"(~{len(frames) / video_fps:.1f}s at {video_fps}fps)")

    # --- main loop ----------------------------------------------------------
    def run(self, max_steps=250, render=True, record=None, video_fps=30,
            video_res=(1280, 720), video_seconds=20.0, capture_every=None):
        """Drive all snakes.

        record=<path>  : render offscreen and write an MP4 (no live viewer). The
                         episode's sim time is compressed into ~video_seconds.
        render=True    : live interactive viewer (ignored while recording).
        render=False   : headless (no window, no MP4) -- for smoke tests.
        """
        recording = record is not None
        renderer = cam = None
        frames = []
        if recording:
            render = False
            renderer, cam, capture_every = self._init_recording(
                video_res, video_fps, video_seconds, max_steps, capture_every)
        print(f"[swarm] {len(self.snakes)} snakes | maze "
              f"{self.config.env.maze_height}x{self.config.env.maze_width} | "
              f"goal {self.maze_info['goal_pos']} | "
              f"nq={self.model.nq} nu={self.model.nu} | "
              f"{'recording ' + record if recording else f'render={render}'}")
        substep = 0
        try:
            for step in range(max_steps):
                actions = self._predict_all()
                for s, a in zip(self.snakes, actions):
                    if s.done:
                        continue  # goal reached: frozen, CPG no longer driven
                    R, omega, theta, delta = (float(x) for x in np.ravel(a))
                    s.cpg.set_parameters(R=R, omega=omega, theta=theta, delta=delta)

                for _ in range(self.sim_steps):
                    for s in self.snakes:
                        if s.done:
                            # Hold the frozen pose so the snake stops moving.
                            self.data.ctrl[s.ctrl_idx] = s.frozen_ctrl
                        else:
                            tgt = np.clip(s.cpg.update(),
                                          s.jnt_range[:, 0], s.jnt_range[:, 1])
                            self.data.ctrl[s.ctrl_idx] = tgt
                    mujoco.mj_step(self.model, self.data)
                    if recording and substep % capture_every == 0:
                        renderer.update_scene(self.data, camera=cam)
                        self._draw_markers(renderer.scene, append=True)
                        frames.append(renderer.render().copy())
                    elif render and not self._render():
                        print("[swarm] viewer closed; stopping.")
                        return
                    substep += 1

                for s, a in zip(self.snakes, actions):
                    s.last_action = np.asarray(a, dtype=np.float32).ravel()
                self._advance_targets()

                n_done = sum(s.done for s in self.snakes)
                print(f"[swarm] step {step + 1}/{max_steps}  reached goal: "
                      f"{n_done}/{len(self.snakes)}"
                      + (f"  frames: {len(frames)}" if recording else ""))
                if n_done == len(self.snakes):
                    print("[swarm] all snakes reached the goal.")
                    break
            if recording:
                self._encode(record, frames, video_fps)
            elif render:
                # Hold the final frame so the viewer stays open for inspection.
                print("[swarm] done. Close the viewer window to exit.")
                while self.viewer is not None and self.viewer.is_running():
                    self._draw_overlay()
                    self.viewer.sync()
                    time.sleep(0.05)
        finally:
            if renderer is not None:
                renderer.close()
            self.close()

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


# ---------------------------------------------------------------------------
# Shared CLI argument helpers (used by viz_stochastic.py / viz_checkpoints.py)
# ---------------------------------------------------------------------------
def add_common_args(parser):
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="YAML config (must match what the model was trained "
                             "with, esp. maze size / action bounds).")
    parser.add_argument("--max-steps", type=int, default=250,
                        help="Max RL steps before stopping (default 250).")
    parser.add_argument("--render-slowdown", type=float, default=2.0,
                        help="Playback slowdown (1.0 = real time; larger = slower).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for maze generation (reproducible layout/goal).")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="Lateral spawn jitter between snakes (world units; "
                             "default 0 = all spawn at the start cell).")
    parser.add_argument("--no-render", dest="render", action="store_false",
                        help="Run headless (no viewer window, no real-time "
                             "pacing) -- useful for a quick sanity check.")
    parser.set_defaults(render=True)
    # MP4 recording (offscreen; disables the live viewer while active).
    parser.add_argument("--record", type=str, default=None, metavar="OUT.mp4",
                        help="Render offscreen and write an MP4 to this path "
                             "instead of opening the live viewer.")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="MP4 frame rate (default 30).")
    parser.add_argument("--video-seconds", type=float, default=20.0,
                        help="Target MP4 duration; the episode's sim time is "
                             "compressed to roughly this many seconds (default 20).")
    parser.add_argument("--video-res", type=parse_res, default=(1280, 720),
                        metavar="WxH",
                        help="MP4 resolution as WxH (default 1280x720; clamped to "
                             "the offscreen buffer and rounded for H.264).")
    return parser


def parse_res(s):
    """Parse a 'WxH' string into an (int, int) tuple (for --video-res)."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(f"resolution must be WxH, got {s!r}")


def run_from_args(swarm, args):
    """Shared entry: dispatch a SnakeSwarm.run() from parsed common args."""
    swarm.run(max_steps=args.max_steps, render=args.render,
              record=args.record, video_fps=args.video_fps,
              video_res=args.video_res, video_seconds=args.video_seconds)
