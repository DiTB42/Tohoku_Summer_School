"""TEMPORARY throwaway script.

Renders a SNAKE inside a generated maze (no terrain / caves / decorations) from
a camera placed ABOVE AND BEHIND the snake's head, looking forward along the
direction the snake faces -- roughly "what the snake sees". Produces two PNGs
from ONE maze:

  1. snake_sees_waypoints.png     -- snake in a straight corridor, the yellow A*
                                     waypoint trail recedes ahead in full view.
  2. snake_no_waypoints.png       -- snake at a corner facing a wall; the
                                     waypoints bend away around the corner and
                                     are NOT in its forward view.

Run:  uv run python temp_generate_maze_images.py [--seed 1]
"""

import os
import argparse
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import imageio.v2 as imageio

from mazes.make_maze import create_maze_layout, get_valid_spawn_points, astar
from mazes.mujoco_tools import make_maze_on_mujoco

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SCENE_XML = os.path.join(_REPO_ROOT, "scenes", "scene.xml")  # includes the snake
START_POS = (1, 1)


def maze_to_world(col, row, start=START_POS):
    """maze (col,row) -> world (x,y). x = col-start_col, y = -(row-start_row)."""
    return np.array([col - start[0], -(row - start[1])], dtype=float)


def _patch_offscreen(xml_path, w=1920, h=1080):
    """scene.xml has no offwidth/offheight, so mujoco.Renderer would be capped
    at the 640x480 default. Inject a big offscreen framebuffer into <global>.
    Also rewrite the <include file="snake.xml"> to an absolute path so the
    scene loads from the scratch dir (the include is otherwise relative)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for inc in root.findall("./include"):
        f = inc.get("file")
        if f and not os.path.isabs(f):
            inc.set("file", os.path.join(_REPO_ROOT, "scenes", f))
    visual = root.find("./visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    glob = visual.find("./global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", str(w))
    glob.set("offheight", str(h))
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _free_joint_qadr(model):
    """qpos start address of the snake's (only) free joint."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.jnt_qposadr[j]
    raise RuntimeError("no free joint found (snake missing?)")


def _yaw_quat(heading_rad):
    """Quaternion (w,x,y,z) placing the snake HEAD facing `heading_rad`.

    The snake body trails along its local +x, so the head faces local -x; a pure
    z-rotation by a = heading + pi makes local -x point along `heading`.
    (Check: default quat (0,0,0,1) => a=pi => head faces +x/east, as authored.)"""
    a = heading_rad + np.pi
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def place_and_render(model, data, qadr, cell, heading_vec, out_png,
                     res=(1600, 1000)):
    """Pose the snake at `cell` facing `heading_vec`, put the camera above and
    behind the head looking forward (so the snake sits in the lower foreground
    and we see the corridor ahead), and render one frame to `out_png`."""
    f = np.asarray(heading_vec, dtype=float)
    f = f / np.linalg.norm(f)
    heading = np.arctan2(f[1], f[0])
    head_xy = maze_to_world(*cell)

    mujoco.mj_resetData(model, data)
    data.qpos[qadr:qadr + 3] = [head_xy[0], head_xy[1], 0.03]
    data.qpos[qadr + 3:qadr + 7] = _yaw_quat(heading)
    data.ctrl[:] = 0.0
    # Let it settle onto its wheels (straight pose held by the position servos).
    for _ in range(60):
        mujoco.mj_step(model, data)

    w, h = res
    renderer = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    look = head_xy + f * 0.6                     # aim just ahead of the head
    cam.lookat[:] = [look[0], look[1], 0.05]
    cam.azimuth = float(np.degrees(heading))     # view direction == heading
    cam.elevation = -34.0                        # tilt down over the snake
    cam.distance = 3.2                           # far enough to keep snake in frame

    renderer.update_scene(data, camera=cam)
    imageio.imwrite(out_png, renderer.render())
    print(f"[temp] wrote {out_png}")


def _is_open(maze, c, r):
    return 0 <= r < maze.shape[0] and 0 <= c < maze.shape[1] and maze[r][c] == 0


def find_offpath_corridor(maze, path):
    """Find an off-path corridor to face down so NO waypoints are in view.

    Picks the (cell, direction) whose straight run of open cells ahead stays as
    FAR from the A* path as possible (maximizing the min cell-distance to any
    path cell), so there are no sightlines over the walls to any waypoint. Needs
    >=2 open off-path cells behind the snake for the camera to sit in.
    Returns (cell (col,row), heading_world (x,y))."""
    path_set = set(path)
    # Min Manhattan distance from every cell to the nearest path cell.
    def dist_to_path(c, r):
        return min(abs(c - pc) + abs(r - pr) for pc, pr in path)

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # (dcol, drow)
    best = None  # (score_min_dist, run, cell, heading_world)
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            if not _is_open(maze, c, r) or (c, r) in path_set:
                continue
            for dc, dr in dirs:
                if not all(_is_open(maze, c - k * dc, r - k * dr)
                           and (c - k * dc, r - k * dr) not in path_set
                           for k in (1, 2)):
                    continue
                run, cc, cr, min_d = 0, c + dc, r + dr, dist_to_path(c, r)
                while _is_open(maze, cc, cr) and (cc, cr) not in path_set:
                    run += 1
                    min_d = min(min_d, dist_to_path(cc, cr))
                    cc += dc
                    cr += dr
                if run < 2:
                    continue
                cand = (min_d, run, (c, r), np.array([dc, -dr], dtype=float))
                if best is None or cand[:2] > best[:2]:
                    best = cand
    if best is None:
        raise RuntimeError("no off-path corridor found")
    return best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    maze = np.array(create_maze_layout(11, 11))
    valid = get_valid_spawn_points(maze)
    sx, sy = START_POS
    cands = [p for p in valid
             if abs(p[0] - sx) + abs(p[1] - sy) >= 6 and tuple(p) != START_POS]
    cands = cands or [p for p in valid if tuple(p) != START_POS]
    goal_tuple = cands[np.random.randint(0, len(cands))]
    goal_pos = list(goal_tuple)
    path = astar(maze, START_POS, goal_tuple)  # list of (col,row), goal is a tuple
    path_set = set(path)

    # "Can see": stand a couple cells into the path, facing along it, so the
    # whole yellow waypoint trail recedes straight ahead in view.
    see_i = 2
    see_cell = path[see_i]
    step = np.subtract(path[see_i + 1], path[see_i])           # (dcol, drow)
    see_heading = np.array([step[0], -step[1]], dtype=float)   # world (y flipped)

    # "Cannot see": stand in an off-path corridor facing down it -- no waypoints
    # exist anywhere ahead of the snake.
    blind_cell, blind_heading = find_offpath_corridor(maze, path)

    scratch = os.path.join(_REPO_ROOT, "_temp_maze_scenes")
    os.makedirs(scratch, exist_ok=True)
    combined = os.path.join(scratch, "scene_snake.xml")

    make_maze_on_mujoco(
        load_file_path=_SCENE_XML,          # base scene WITH the snake
        maze=maze,
        start_pos=list(START_POS),
        goal_pos=goal_pos,
        waypoints=path,
        waypoint_rgba="1 0.85 0 1",         # opaque bright yellow
        goal_rgba="0.1 0.85 0.2 0.9",       # opaque green goal
        terrain=None, widths=None, decorations=False,   # --no-terrain look
        save_file_path=combined,
    )
    _patch_offscreen(combined)

    model = mujoco.MjModel.from_xml_path(combined)
    data = mujoco.MjData(model)
    qadr = _free_joint_qadr(model)

    # 1. Snake CAN see waypoints: face along the path, trail recedes ahead.
    place_and_render(model, data, qadr, see_cell, see_heading,
                     os.path.join(_REPO_ROOT, "snake_sees_waypoints.png"))

    # 2. Snake CANNOT see waypoints: face down an off-path corridor (no trail).
    place_and_render(model, data, qadr, blind_cell, blind_heading,
                     os.path.join(_REPO_ROOT, "snake_no_waypoints.png"))


if __name__ == "__main__":
    main()
