import xml.etree.ElementTree as ET
import numpy as np

# Per-terrain tangential friction (mu). "grass" is the baseline: its tile
# friction equals snake.xml's global default <geom friction="3 0.005 0.0001">
# (which the bare floor plane also inherits), so grass physics is identical to
# the pre-terrain floor. env_snake.py imports TERRAIN_FRICTION to build the
# head/tail "felt friction" observation inputs, so this dict is the single
# source of truth for both physics and obs.
TERRAIN_FRICTION = {"grass": 3.0, "ice": 0.5, "dirt": 7.0}
# Full friction triples (tangential, torsional, rolling) + tile colors: every
# open cell gets a tile (grass green, like the kb branch's corridor floor).
# The ROLLING (3rd) and torsional (2nd) components affect the passive wheels far
# more than tangential mu (which mostly governs lateral grip and saturates).
# ICE rolls lower than grass (+ low torsional) so it GLIDES -- measurably faster
# for the trained policy (~+10% head speed). DIRT rolls/twists much higher to
# drag the wheels; note this slows a WEAK/open-loop gait a lot but barely dents
# a competent policy, whose strong position servos hit their joint targets
# regardless of load (rolling drag then costs torque, not speed). Grass keeps the
# baseline triple (identical to the bare floor / --no-terrain). Only the physics
# tiles change; the obs (TERRAIN_FRICTION, tangential only) is untouched.
TERRAIN_TILES = {
    "grass": {"friction": "3.0 0.005 0.0001",  "rgba": "0.18 0.48 0.16 1"},  # green, baseline
    "ice":  {"friction": "0.5 0.001 0.00002",  "rgba": "0.35 0.65 0.95 1"},  # glacial blue, slippery + glidey
    # NOTE: the viewer headlight washes tile colors out, so the rgba must be
    # noticeably more saturated than the target on-screen color.
    "dirt": {"friction": "7.0 0.4 0.02",       "rgba": "0.42 0.28 0.16 1"},  # brown, rough + draggy
}

# ---------------------------------------------------------------------------
# Decorative scene dressing (purely cosmetic; gated by decorations=True, which
# callers wire to env.terrain_enabled == NOT --no-terrain). Every decoration
# geom is NON-COLLIDING (contype=0 conaffinity=0) and carries no friction/
# priority, so it never touches snake physics or the 42-D observation -> all
# existing checkpoints keep working. Randomness comes from a LOCAL generator
# seeded off the maze layout (_decor_salt / _decor_rng), NEVER the env's shared
# np_random, so a seeded maze/goal stays byte-identical whether decorations are
# on or off (preserves the reproducibility invariant in env_snake.reset()).
# ---------------------------------------------------------------------------
_NOCOLLIDE = {"contype": "0", "conaffinity": "0"}
# Goal beacon (translucent glowing marker over the goal). Kept small so it
# reads as a marker, not a temple.
DECOR_BEACON_RGBA = "0.55 0.85 1.0 0.30"  # translucent glow pillar
DECOR_ORB_RGBA = "0.80 0.95 1.0 0.75"     # floating orb on top
DECOR_TRUNK_RGBA = "0.39 0.25 0.13 1"
DECOR_BLOSSOM_RGBA = "0.98 0.74 0.86 0.95"
DECOR_PETAL_RGBA = "0.98 0.66 0.84 0.90"
DECOR_WATERFALL_RGBA = "0.35 0.62 0.90 0.45"
DECOR_WATERFALL_FOAM_RGBA = "0.95 0.97 1.0 0.85"

def _decor_salt(maze):
    """Deterministic 32-bit salt from the maze layout (position-weighted fold
    over the wall pattern). Varies per maze but draws NO shared RNG, so seeded
    reproducibility of the maze/goal holds regardless of decorations."""
    b = np.asarray(maze, dtype=np.uint64).ravel()
    idx = np.arange(1, b.size + 1, dtype=np.uint64)
    return int((b * idx).sum() & np.uint64(0xFFFFFFFF))

# --- Wall rock dressing (ported from the kb branch) ------------------------
# Unlike a centered symmetric stack on every wall, mountains grow OUT of the
# wall face that borders an open corridor (asymmetric, hugging the exposed
# edge) and only a SUBSET of walls get them (corners/exposed faces likelier,
# many walls stay plain). Grey stone caps land on ~half the walls. Per-cell
# seeds fold in the maze-derived salt so the look varies per maze while
# consuming NO shared np_random (reproducibility preserved). Everything stays
# non-colliding (cosmetic only).

def _exposed_wall_faces(maze, row, col):
    """Directions (dx, dy) in which this wall cell borders an open corridor (or
    the maze edge) -- the faces a mountain can grow from."""
    faces = []
    for d_row, d_col, dx, dy in ((-1, 0, 0.0, -1.0), (1, 0, 0.0, 1.0),
                                 (0, -1, -1.0, 0.0), (0, 1, 1.0, 0.0)):
        nr, nc = row + d_row, col + d_col
        if not (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]) or maze[nr][nc] == 0:
            faces.append((dx, dy))
    return faces

def _should_add_mountain_wall(exposed_faces, seed):
    """kb-style corner-biased gate, but sparser: corners 50%, two-face walls
    25%, single-face walls 20%. Yields mountains on ~1/4-1/3 of walls -- about
    half the earlier (~55%) coverage the user asked to thin out."""
    n = len(exposed_faces)
    if n == 0:
        return False
    if n >= 3:
        return seed % 2 == 0
    if n == 2:
        return seed % 4 == 0
    return seed % 5 == 0

def _base_stones_for_cell(row, col, salt, bx, by, bh):
    """kb grey-stone caps: 2-3 randomly shaped/placed boxes on a wall top. Only
    ~half the walls get any (seed gate), so stones are ~half as many as before."""
    seed = (((row + 1) * 41719 + (col + 1) * 11317) ^ salt) & 0xFFFFFFFF
    if (seed % 2) == 0:              # skip ~half the walls
        return []
    rng = np.random.default_rng(seed)
    stones = []
    for k in range(2 + ((seed >> 1) % 2)):
        roll = (seed + k) % 3
        if roll == 0:
            sx = bx * (0.16 + 0.16 * rng.random()); sy = by * (0.16 + 0.16 * rng.random()); sz = bh * (0.10 + 0.10 * rng.random())
        elif roll == 1:
            sx = bx * (0.22 + 0.20 * rng.random()); sy = by * (0.10 + 0.10 * rng.random()); sz = bh * (0.08 + 0.08 * rng.random())
        else:
            sx = bx * (0.10 + 0.10 * rng.random()); sy = by * (0.22 + 0.20 * rng.random()); sz = bh * (0.08 + 0.08 * rng.random())
        px = (rng.random() - 0.5) * max(0.05, bx * 1.35 - sx)
        py = (rng.random() - 0.5) * max(0.05, by * 1.35 - sy)
        pz = bh + sz * (0.90 + 0.12 * k)
        g = 0.40 + 0.18 * rng.random()
        stones.append((sx, sy, sz, px, py, pz, f"{g:.3f} {g:.3f} {g + 0.03:.3f} 1"))
    return stones

def _add_base_stones(body, row, col, salt, bx, by, bh):
    stones = _base_stones_for_cell(row, col, salt, bx, by, bh)
    for k, (sx, sy, sz, px, py, pz, rgba) in enumerate(stones):
        ET.SubElement(body, "geom", dict(_NOCOLLIDE,
            name=f"stone_{row}_{col}_{k}", type="box",
            size=f"{sx} {sy} {sz}", pos=f"{px} {py} {pz}", rgba=rgba))
    return len(stones)

def _mountain_layers_for_cell(maze, row, col, salt, bx, by, bh):
    """kb mountain profile: 1-5 tiers that grow out of an exposed wall face and
    taper up/inward, with an optional peak on corner walls. Returns a list of
    (size_x, size_y, size_z, pos_x, pos_y, pos_z, rgba). Asymmetric by design."""
    seed = (((row + 1) * 92821 + (col + 1) * 68917) ^ salt) & 0xFFFFFFFF
    faces = _exposed_wall_faces(maze, row, col)
    if not faces or not _should_add_mountain_wall(faces, seed):
        return []
    rng = np.random.default_rng(seed)
    primary = faces[seed % len(faces)]
    secondary = None
    if len(faces) >= 2 and seed % 3 != 0:
        secondary = faces[(seed // 3) % len(faces)]
        if secondary == primary:
            secondary = None

    roll = seed % 10
    if roll <= 1:
        count, hscale, along_r, depth_r, taper_r, vspace = 1, 0.75, (0.82, 1.02), (0.72, 0.96), (0.92, 0.98), (0.92, 1.00)
    elif roll <= 3:
        count, hscale, along_r, depth_r, taper_r, vspace = 2, 0.82, (0.62, 0.86), (0.46, 0.70), (0.84, 0.94), (0.82, 0.92)
    elif roll <= 5:
        count, hscale, along_r, depth_r, taper_r, vspace = 3, 0.90, (0.50, 0.78), (0.36, 0.60), (0.80, 0.90), (0.78, 0.88)
    elif roll <= 7:
        count, hscale, along_r, depth_r, taper_r, vspace = 4, 0.98, (0.44, 0.68), (0.30, 0.52), (0.76, 0.88), (0.74, 0.84)
    else:
        count, hscale, along_r, depth_r, taper_r, vspace = 5, 1.08, (0.38, 0.60), (0.24, 0.44), (0.72, 0.84), (0.70, 0.80)

    layers = []
    top_z = bh
    csx = csy = cpx = cpy = None
    for k in range(count):
        fx, fy = primary if (k == 0 or secondary is None) else secondary
        tx, ty = -fy, fx
        if csx is None:
            along = (by if fx else bx) * (along_r[0] + (along_r[1] - along_r[0]) * rng.random())
            depth = (bx if fx else by) * (depth_r[0] + (depth_r[1] - depth_r[0]) * rng.random())
        else:
            taper = taper_r[0] + (taper_r[1] - taper_r[0]) * rng.random()
            along = (csy if fx else csx) * taper
            depth = (csx if fx else csy) * (taper - 0.04 * rng.random())
        lh = bh * max(0.24, hscale - 0.09 * k + 0.10 * rng.random())
        if fx:
            sx, sy = depth, along
            if cpx is None:
                px = (0.0 + fx * bx) - fx * sx * (0.78 + 0.12 * rng.random())
                py = 0.0 + ty * (rng.random() - 0.5) * max(0.06, by - sy)
            else:
                px = cpx - fx * sx * (0.10 + 0.08 * rng.random())
                py = cpy + ty * (rng.random() - 0.5) * max(0.03, sy * 0.25)
        else:
            sx, sy = along, depth
            if cpy is None:
                px = 0.0 + tx * (rng.random() - 0.5) * max(0.06, bx - sx)
                py = (0.0 + fy * by) - fy * sy * (0.78 + 0.12 * rng.random())
            else:
                px = cpx + tx * (rng.random() - 0.5) * max(0.03, sx * 0.25)
                py = cpy - fy * sy * (0.10 + 0.08 * rng.random())
        if secondary is not None and k == count - 1 and count > 1:
            px += secondary[0] * sx * 0.35
            py += secondary[1] * sy * 0.35
        gg = 0.22 + 0.06 * rng.random()
        bb = 0.12 + 0.04 * rng.random()
        layers.append((sx, sy, lh, px, py, top_z + lh, f"0.31 {gg:.3f} {bb:.3f} 1"))
        top_z += 2.0 * lh * (vspace[0] + (vspace[1] - vspace[0]) * rng.random())
        csx, csy, cpx, cpy = sx, sy, px, py

    if len(faces) >= 2 and count >= 2:
        ph = bh * (0.30 + 0.18 * rng.random())
        prx = max(bx * 0.10, layers[-1][0] * (0.62 + 0.12 * rng.random()))
        pry = max(by * 0.10, layers[-1][1] * (0.62 + 0.12 * rng.random()))
        layers.append((prx, pry, ph, layers[-1][3] * (0.60 + 0.16 * rng.random()),
                       layers[-1][4] * (0.60 + 0.16 * rng.random()), top_z + ph, "0.38 0.29 0.18 1"))
    return layers

def _add_mountain_wall_layers(body, maze, row, col, salt, bx, by, bh):
    layers = _mountain_layers_for_cell(maze, row, col, salt, bx, by, bh)
    for k, (sx, sy, sz, px, py, pz, rgba) in enumerate(layers):
        ET.SubElement(body, "geom", dict(_NOCOLLIDE,
            name=f"mountain_{row}_{col}_{k}", type="box",
            size=f"{sx} {sy} {sz}", pos=f"{px} {py} {pz}", rgba=rgba))
    if not layers:
        return 0, None
    peak = max(layers, key=lambda g: g[5] + g[2])
    peak_local = (peak[3], peak[4], peak[5] + peak[2])
    return len(layers), peak_local

def _add_cherry_blossom_tree(body, maze, row, col, salt, bx, by, bh):
    """Add a stylized cherry tree on a decorated wall cell.

    Tree structure:
      * brown trunk (cylinder)
      * pink canopy (spheres) mounted on a hinge joint named cherry_sway_*
      * tiny petals on slide joints named cherry_petal_drop_* (animated in code)

    All geoms are non-colliding and purely cosmetic.
    """
    seed = (((row + 1) * 12379 + (col + 1) * 28411) ^ (salt * 3 + 17)) & 0xFFFFFFFF
    faces = _exposed_wall_faces(maze, row, col)
    if not faces:
        return False
    # Higher coverage than before, but still not every decorated wall.
    if (seed % 5) == 0:
        return False

    rng = np.random.default_rng(seed)
    fx, fy = faces[seed % len(faces)]
    tx, ty = -fy, fx

    def _add_one_tree(tree_idx, force_overhang=False):
        height_mode = tree_idx % 3
        if height_mode == 0:
            trunk_h = bh * (1.25 + 0.35 * rng.random())   # short
        elif height_mode == 1:
            trunk_h = bh * (1.85 + 0.45 * rng.random())   # medium
        else:
            trunk_h = bh * (2.45 + 0.60 * rng.random())   # tall

        narrow = ((seed + tree_idx) % 2) == 0
        if narrow:
            canopy_x = bx * (0.24 + 0.10 * rng.random())
            canopy_y = by * (0.32 + 0.14 * rng.random())
        else:
            canopy_x = bx * (0.36 + 0.22 * rng.random())
            canopy_y = by * (0.42 + 0.26 * rng.random())
        canopy_z = bh * (0.34 + 0.20 * rng.random())

        if force_overhang:
            face_push = 0.62 + 0.22 * rng.random()
        else:
            face_push = 0.36 + 0.24 * rng.random()
        base_x = fx * bx * face_push + tx * (rng.random() - 0.5) * bx * 0.45
        base_y = fy * by * face_push + ty * (rng.random() - 0.5) * by * 0.45
        trunk_r = min(bx, by) * (0.07 + 0.03 * rng.random())

        tag = f"{row}_{col}_{tree_idx}"
        tree = ET.SubElement(body, "body", {
            "name": f"cherry_{tag}",
            "pos": f"{base_x} {base_y} {bh + 0.003}",
        })
        ET.SubElement(tree, "geom", dict(_NOCOLLIDE,
            name=f"cherry_trunk_{tag}", type="cylinder",
            size=f"{trunk_r} {trunk_h * 0.5}", pos=f"0 0 {trunk_h * 0.5}",
            rgba=DECOR_TRUNK_RGBA))

        canopy = ET.SubElement(tree, "body", {
            "name": f"cherry_canopy_{tag}",
            "pos": f"0 0 {trunk_h}",
        })
        ET.SubElement(canopy, "joint", {
            "name": f"cherry_sway_{tag}",
            "type": "hinge",
            "axis": "0 1 0",
            "range": "-0.45 0.45",
            "damping": "0.02",
        })
        # Dense clustered small rectangular leaves for textured canopy.
        n_leaf_blocks = 24 if not narrow else 18
        for b in range(n_leaf_blocks):
            u = rng.normal(0.0, 0.42)
            v = rng.normal(0.0, 0.40)
            w = rng.normal(0.35, 0.30)
            lx = np.clip(u, -1.1, 1.1) * canopy_x
            ly = np.clip(v, -1.1, 1.1) * canopy_y
            lz = max(-0.35, min(1.35, w)) * canopy_z
            sx = canopy_x * (0.12 + 0.10 * rng.random())
            sy = canopy_y * (0.12 + 0.10 * rng.random())
            sz = canopy_z * (0.16 + 0.12 * rng.random())
            ET.SubElement(canopy, "geom", dict(_NOCOLLIDE,
                name=f"cherry_blossom_{tag}_{b}", type="box",
                size=f"{sx} {sy} {sz}", pos=f"{lx} {ly} {lz}", rgba=DECOR_BLOSSOM_RGBA))

        petal_count = 10 if force_overhang else 7
        for k in range(petal_count):
            px = fx * canopy_x * (0.35 + 0.95 * rng.random()) + tx * (rng.random() - 0.5) * canopy_x * 1.6
            py = fy * canopy_y * (0.35 + 0.95 * rng.random()) + ty * (rng.random() - 0.5) * canopy_y * 1.6
            pz = canopy_z * (0.30 + 1.05 * rng.random())
            drop_span = 1.40 + 1.30 * rng.random() if force_overhang else 1.00 + 0.90 * rng.random()
            petal = ET.SubElement(tree, "body", {
                "name": f"cherry_petal_{tag}_{k}",
                "pos": f"{px} {py} {trunk_h + pz}",
            })
            ET.SubElement(petal, "joint", {
                "name": f"cherry_petal_drop_{tag}_{k}",
                "type": "slide",
                "axis": "0 0 -1",
                "range": f"0 {drop_span}",
                "damping": "0",
            })
            ET.SubElement(petal, "geom", dict(_NOCOLLIDE,
                name=f"cherry_petal_geom_{tag}_{k}",
                type="box", size="0.008 0.004 0.006", rgba=DECOR_PETAL_RGBA))

    # Mix single trees and clusters for variation.
    cluster_mode = seed % 10
    if cluster_mode <= 4:
        n_trees = 1
    elif cluster_mode <= 7:
        n_trees = 2
    else:
        n_trees = 3
    for idx in range(n_trees):
        _add_one_tree(idx, force_overhang=(idx == n_trees - 1 and n_trees >= 2 and (seed % 2 == 1)))
    return True

def _add_waterfall_from_peak(worldbody, peak_xyz, salt):
    """Add a decorative VERTICAL waterfall from mountain peak to floor.

    Starts exactly at the detected mountain peak (same x/y/z) and falls straight
    down in world -z to the route/floor height.
    """
    px, py, pz = peak_xyz
    floor_z = 0.10
    length = float(max(0.5, pz - floor_z))
    axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    wf = ET.SubElement(worldbody, "body", {
        "name": "waterfall_main",
        "pos": f"{px} {py} {pz}",
    })

    n_segments = max(6, int(length / 0.16))
    for s in range(n_segments):
        t = (s + 0.5) / n_segments
        pos = axis * (t * length)
        ET.SubElement(wf, "geom", dict(_NOCOLLIDE,
            name=f"waterfall_blue_{s}", type="box",
            size="0.070 0.026 0.080",
            pos=f"{pos[0]} {pos[1]} {pos[2]}", rgba=DECOR_WATERFALL_RGBA))

    rng = np.random.default_rng((salt * 7 + 19) & 0xFFFFFFFF)
    for k in range(14):
        dx = (rng.random() - 0.5) * 0.12
        dy = (rng.random() - 0.5) * 0.06
        strip = ET.SubElement(wf, "body", {
            "name": f"waterfall_flow_body_{k}",
            "pos": f"{dx} {dy} 0.0",
        })
        ET.SubElement(strip, "joint", {
            "name": f"waterfall_flow_{k}",
            "type": "slide",
            "axis": "0 0 -1",
            "range": f"0 {length}",
            "damping": "0",
        })
        ET.SubElement(strip, "geom", dict(_NOCOLLIDE,
            name=f"waterfall_white_{k}", type="box",
            size="0.010 0.006 0.050", rgba=DECOR_WATERFALL_FOAM_RGBA))

def _add_goal_beacon(worldbody, gx, gy):
    """A small translucent glowing pillar + floating orb over the goal, so it
    reads clearly from the top-down swarm camera. Cosmetic, non-colliding."""
    beacon = ET.SubElement(worldbody, "body", {"name": "beacon", "pos": f"{gx} {gy} 0"})
    ET.SubElement(beacon, "geom", dict(_NOCOLLIDE,
        name="beacon_pillar", type="cylinder", size="0.07 0.65",
        pos="0 0 0.65", rgba=DECOR_BEACON_RGBA))
    ET.SubElement(beacon, "geom", dict(_NOCOLLIDE,
        name="beacon_orb", type="sphere", size="0.10",
        pos="0 0 1.45", rgba=DECOR_ORB_RGBA))

def _recolor_scene(root):
    """In-place edit of the base scene's <asset> to greener ground + a warmer
    sky. Applied at generation time (gated by decorations) rather than baked
    into scene.xml, so --no-terrain keeps the original palette."""
    asset = root.find("./asset")
    if asset is None:
        return
    for tex in asset.findall("texture"):
        if tex.get("name") == "groundplane":
            tex.set("rgb1", "0.18 0.44 0.18")
            tex.set("rgb2", "0.12 0.34 0.13")
            tex.set("markrgb", "0.22 0.36 0.16")
            tex.set("mark", "cross")
        elif tex.get("type") == "skybox":
            tex.set("rgb1", "0.45 0.62 0.80")
            tex.set("rgb2", "0.10 0.14 0.28")
    for mat in asset.findall("material"):
        if mat.get("name") == "groundplane":
            mat.set("texrepeat", "8 8")

def _cave_recessions(maze, widths, start_pos, box_size):
  """Map each flanking wall (row,col) -> world-face deltas that NARROW the caves.

  For every narrowed corridor cell (from generate_widths, keyed (col,row)==(x,y)),
  the two perpendicular flank walls grow their INNER face `e` into the corridor
  while the outer face stays fixed, pinching the passage symmetrically about its
  centerline (span 1.0 - 2e). Orientation is re-derived from the maze array itself
  (horizontal <=> both left/right neighbors are open corridor). Returns
  {(row, col): {"x0","x1","y0","y1"}} of signed additive offsets on the wall's
  default full-cell world bounds. Offsets from multiple caves accumulate (a wall
  shared by two parallel caves grows on both faces)."""
  recess = {}
  def _add(cell_rc, key, delta):
    d = recess.setdefault(cell_rc, {"x0": 0.0, "x1": 0.0, "y0": 0.0, "y1": 0.0})
    d[key] += delta
  for (col, row), e in widths.items():
    if e <= 0.0:
      continue
    # Horizontal cave: travel along x, so the OPEN neighbors are left/right.
    horizontal = maze[row][col - 1] == 0 and maze[row][col + 1] == 0
    if horizontal:
      # Flanks are the row-1 (above, world +y) and row+1 (below, world -y) walls.
      _add((row - 1, col), "y0", -e)   # above wall: grow its -y (bottom) face down into corridor
      _add((row + 1, col), "y1", +e)   # below wall: grow its +y (top) face up into corridor
    else:
      # Vertical cave: flanks are col-1 (world -x) and col+1 (world +x) walls.
      _add((row, col - 1), "x1", +e)   # left wall: grow its +x (right) face right into corridor
      _add((row, col + 1), "x0", -e)   # right wall: grow its -x (left) face left into corridor
  return recess

def make_maze_on_mujoco(load_file_path, maze, start_pos, goal_pos, waypoints=None, box_size=[0.5, 0.5, 0.15], maze_rgba="0.5 0.5 0.5 1", goal_rgba="0.6 0.9 0.6 0.5", waypoint_rgba="1 1 0 0.5", terrain=None, widths=None, decorations=False, save_file_path=None):
    tree = ET.parse(load_file_path)
    root = tree.getroot()
    worldbody = root.find("./worldbody")

    # Rimuovi i vecchi elementi del labirinto per rigenerarlo. 'beacon' is the
    # decorative goal-beacon body; wall-attached decorations (mountain_/stone_)
    # ride on the box_* bodies removed here, so they need no separate handling.
    for body in worldbody.findall("body"):
        name = body.get('name')
        if name and ('box_' in name or 'goal' in name or name.startswith('beacon')):
            worldbody.remove(body)
    for geom in list(worldbody.findall("geom")):
        geom_name = geom.get('name') or ""
        if 'waypoint_' in geom_name or geom_name.startswith('terrain_'):
            worldbody.remove(geom)

    # Cosmetic dressing only (see the decoration helpers above): recolor the
    # scene now, and precompute a layout-derived salt for the per-cell props.
    if decorations:
        _recolor_scene(root)
    decor_salt = _decor_salt(maze) if decorations else 0

    # Terrain tiles: one thin box per non-grass open cell, sitting 0.2 mm proud
    # of the floor plane so contacts land on the tile instead of the plane.
    # priority="1" is REQUIRED: snake geoms inherit mu=3 from snake.xml's global
    # default and equal-priority contacts take the elementwise max, so without
    # it ice (mu=1) would resolve to max(1, 3) = 3 and do nothing. With priority
    # 1 the tile's friction wins outright (for both ice and dirt).
    if terrain:
        for (col, row), terrain_type in terrain.items():
            tile = TERRAIN_TILES[terrain_type]
            pos_x = 2 * box_size[0] * (col - start_pos[0])
            pos_y = -2 * box_size[1] * (row - start_pos[1])
            # Tiles stay full-cell: in a cave the flank walls grow inward and simply
            # sit on top of the tile edges, so the snake only ever contacts the tile
            # in the (narrowed) central passage. No tile resize needed.
            ET.SubElement(worldbody, "geom", {
                "name": f"terrain_{terrain_type}_{row}_{col}",
                "type": "box",
                "size": f"{box_size[0]} {box_size[1]} 0.004",
                "pos": f"{pos_x} {pos_y} -0.0038",
                "rgba": tile["rgba"],
                "friction": tile["friction"],
                "priority": "1",
            })

    # Precompute per-wall face offsets that pinch the caves narrower (empty if no widths).
    recess = _cave_recessions(maze, widths, start_pos, box_size) if widths else {}
    tallest_peak = None

    # Crea i muri del labirinto
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j] == 1:
                pos_x = 2 * box_size[0] * (j - start_pos[0])
                pos_y = -2 * box_size[1] * (i - start_pos[1])
                # Default full-cell world bounds; grow inner faces into adjacent caves.
                x0, x1 = pos_x - box_size[0], pos_x + box_size[0]
                y0, y1 = pos_y - box_size[1], pos_y + box_size[1]
                r = recess.get((i, j))
                if r:
                    x0 += r["x0"]
                    x1 += r["x1"]
                    y0 += r["y0"]
                    y1 += r["y1"]
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                sx, sy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
                body = ET.SubElement(worldbody, "body", {"name": f"box_{i}_{j}", "pos": f"{cx} {cy} {box_size[2]}"})
                ET.SubElement(body, "geom", {"name": f"geom_{i}_{j}", "type": "box", "size": f"{sx} {sy} {box_size[2]}", "rgba": maze_rgba})
                # Cosmetic rocky dressing (non-colliding): stones/mountains + blossoms.
                if decorations:
                    n_stones = _add_base_stones(body, i, j, decor_salt, sx, sy, box_size[2])
                    n_mountains, peak_local = _add_mountain_wall_layers(body, maze, i, j, decor_salt, sx, sy, box_size[2])
                    if peak_local is not None:
                        peak_world = (cx + peak_local[0], cy + peak_local[1], box_size[2] + peak_local[2])
                        if tallest_peak is None or peak_world[2] > tallest_peak[2]:
                            tallest_peak = peak_world
                    if n_stones > 0 or n_mountains > 0:
                        _add_cherry_blossom_tree(body, maze, i, j, decor_salt, sx, sy, box_size[2])

    # Decorative waterfall from tallest mountain peak straight down to floor.
    if decorations and tallest_peak is not None and waypoints:
        _add_waterfall_from_peak(worldbody, tallest_peak, decor_salt)

    # Aggiungi i waypoint come sfere visive
    if waypoints:
        for idx, point in enumerate(waypoints):
            if list(point) == start_pos or list(point) == goal_pos:
                continue
            pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
            pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
            ET.SubElement(worldbody, "geom", {"name": f"waypoint_{idx}", "type": "sphere", "size": "0.1", "pos": f"{pos_x} {pos_y} 0.1", "rgba": waypoint_rgba, "contype": "0", "conaffinity": "0"})

    # Crea il goal
    pos_x = 2 * box_size[0] * (goal_pos[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (goal_pos[1] - start_pos[1])
    body = ET.SubElement(worldbody, "body", {"name": "goal", "pos": f"{pos_x} {pos_y} {box_size[2]}"})
    ET.SubElement(body, "geom", {"name": "goal_geom", "type": "box", "size": f"{box_size[0]} {box_size[1]} {box_size[2]}", "contype": "0", "conaffinity": "0", "rgba": goal_rgba})

    # Glowing beacon over the goal so it stands out from the top-down swarm camera.
    if decorations:
        _add_goal_beacon(worldbody, pos_x, pos_y)

    if save_file_path:
        tree.write(save_file_path, encoding="utf-8", xml_declaration=True)

def update_maze(load_file_path, start_pos, goal_pos, waypoints=None, box_size=[0.5, 0.5, 0.15], goal_rgba="0.6 0.9 0.6 0.5", waypoint_rgba="1 1 0 0.5", save_file_path=None):
    tree = ET.parse(load_file_path)
    root = tree.getroot()
    worldbody = root.find("./worldbody")

    # Rimuovi i vecchi elementi del goal e dei waypoint
    for body in worldbody.findall("body"):
        if body.get('name') == 'goal':
            worldbody.remove(body)
            break  # Assumendo un solo goal
            
    for geom in worldbody.findall("geom"):
        if geom.get('name') and 'waypoint_' in geom.get('name'):
            worldbody.remove(geom)

    
    # Aggiungi i waypoint come sfere visive
    if waypoints:
        for idx, point in enumerate(waypoints):
            if list(point) == start_pos or list(point) == goal_pos:
                continue
            pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
            pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
            ET.SubElement(worldbody, "geom", {"name": f"waypoint_{idx}", "type": "sphere", "size": "0.1", "pos": f"{pos_x} {pos_y} 0.1", "rgba": waypoint_rgba, "contype": "0", "conaffinity": "0"})

    # Crea il goal
    pos_x = 2 * box_size[0] * (goal_pos[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (goal_pos[1] - start_pos[1])
    body = ET.SubElement(worldbody, "body", {"name": "goal", "pos": f"{pos_x} {pos_y} {box_size[2]}"})
    ET.SubElement(body, "geom", {"name": "goal_geom", "type": "box", "size": f"{box_size[0]} {box_size[1]} {box_size[2]}", "contype": "0", "conaffinity": "0", "rgba": goal_rgba})

    # Se save_file_path è specificato, sovrascrive il file. Altrimenti, le modifiche sono solo in memoria.
    # Per mantenere lo stesso file, assicurati che save_file_path sia uguale a load_file_path.
    output_path = save_file_path if save_file_path is not None else load_file_path
    tree.write(output_path, encoding="utf-8", xml_declaration=True)