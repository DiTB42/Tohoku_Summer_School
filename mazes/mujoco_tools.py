import xml.etree.ElementTree as ET

import numpy as np


WATERFALL_FLOOR_RGBA = "0.10 0.38 0.62 0.42"
WATERFALL_FLOOR_FRICTION = "4.0 0.2 0.001"
ICE_FLOOR_RGBA = "0.72 0.90 1.00 0.34"
ICE_FLOOR_FRICTION = "0.18 0.00005 0.00001"


def _select_waterfall_points(waypoints, start_pos, goal_pos, rng):
    if not waypoints or len(waypoints) < 5:
        return []

    candidate_points = [
        tuple(point) for point in waypoints[1:-1]
        if list(point) != start_pos and list(point) != goal_pos
    ]
    if not candidate_points:
        return []

    patch_count = min(max(2, len(candidate_points) // 5), len(candidate_points), 6)
    selected_indices = rng.choice(len(candidate_points), size=patch_count, replace=False)
    return [candidate_points[index] for index in sorted(selected_indices)]


def _select_ice_floor_points(maze, start_pos, goal_pos, rng, blocked_points=None):
    blocked = set(blocked_points or [])
    candidates = []
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):
            point = (col, row)
            if maze[row][col] != 0 or list(point) == start_pos or list(point) == goal_pos:
                continue
            if point in blocked:
                continue
            candidates.append(point)

    if not candidates:
        return []

    patch_count = min(max(3, len(candidates) // 14), len(candidates), 8)
    selected_indices = rng.choice(len(candidates), size=patch_count, replace=False)
    return [candidates[index] for index in sorted(selected_indices)]


def _select_narrow_path_cells(waypoints, start_pos, goal_pos, rng, blocked_points=None):
    if not waypoints or len(waypoints) < 5:
        return {}

    blocked = set(blocked_points or [])
    candidates = []
    for idx in range(1, len(waypoints) - 1):
        prev_point = tuple(waypoints[idx - 1])
        point = tuple(waypoints[idx])
        next_point = tuple(waypoints[idx + 1])
        if list(point) == start_pos or list(point) == goal_pos or point in blocked:
            continue
        same_row = prev_point[1] == point[1] == next_point[1]
        same_col = prev_point[0] == point[0] == next_point[0]
        if same_row or same_col:
            orientation = "horizontal" if same_row else "vertical"
            candidates.append((idx, point, orientation))

    if not candidates:
        return {}

    narrow_count = min(max(4, len(candidates) // 2), len(candidates), 12)
    selected_indices = rng.choice(len(candidates), size=narrow_count, replace=False)
    selected_centers = [candidates[index] for index in selected_indices]

    funnel_strengths = {}
    taper_profile = {
        -2: 0.35,
        -1: 0.7,
        0: 1.0,
        1: 0.7,
        2: 0.35,
    }
    for center_idx, _, orientation in selected_centers:
        for offset, strength in taper_profile.items():
            path_idx = center_idx + offset
            if not (1 <= path_idx < len(waypoints) - 1):
                continue

            point = tuple(waypoints[path_idx])
            if list(point) == start_pos or list(point) == goal_pos or point in blocked:
                continue

            prev_point = tuple(waypoints[path_idx - 1])
            next_point = tuple(waypoints[path_idx + 1])
            if orientation == "horizontal":
                if not (prev_point[1] == point[1] == next_point[1]):
                    continue
            else:
                if not (prev_point[0] == point[0] == next_point[0]):
                    continue

            funnel_strengths[point] = max(funnel_strengths.get(point, 0.0), strength)

    return funnel_strengths


def _wall_geom_for_cell(maze, row, col, box_size, narrow_strengths):
    size_x = box_size[0]
    size_y = box_size[1]
    offset_x = 0.0
    offset_y = 0.0
    height = box_size[2]

    neighbors = [
        (-1, 0, 0.0, -1.0),
        (1, 0, 0.0, 1.0),
        (0, -1, -1.0, 0.0),
        (0, 1, 1.0, 0.0),
    ]

    for d_row, d_col, direction_x, direction_y in neighbors:
        nbr_row = row + d_row
        nbr_col = col + d_col
        if not (0 <= nbr_row < maze.shape[0] and 0 <= nbr_col < maze.shape[1]):
            continue
        strength = narrow_strengths.get((nbr_col, nbr_row), 0.0)
        if maze[nbr_row][nbr_col] != 0 or strength <= 0.0:
            continue
        extension_x = (box_size[0] / 3.0) * strength
        extension_y = (box_size[1] / 3.0) * strength
        if d_col != 0:
            size_x += extension_x
            offset_x += direction_x * extension_x
        else:
            size_y += extension_y
            offset_y += direction_y * extension_y

    return size_x, size_y, height, offset_x, offset_y


def _exposed_wall_faces(maze, row, col):
    faces = []
    neighbors = [
        (-1, 0, 0.0, -1.0),
        (1, 0, 0.0, 1.0),
        (0, -1, -1.0, 0.0),
        (0, 1, 1.0, 0.0),
    ]
    for d_row, d_col, direction_x, direction_y in neighbors:
        nbr_row = row + d_row
        nbr_col = col + d_col
        if not (0 <= nbr_row < maze.shape[0] and 0 <= nbr_col < maze.shape[1]):
            faces.append((direction_x, direction_y))
            continue
        if maze[nbr_row][nbr_col] == 0:
            faces.append((direction_x, direction_y))
    return faces


def _should_add_mountain_wall(exposed_faces, seed):
    face_count = len(exposed_faces)
    if face_count == 0:
        return False

    # Keep a mix: plain walls remain common, corners and isolated wall faces
    # are more likely to get mountain treatment.
    if face_count >= 3:
        return True
    if face_count == 2:
        return seed % 4 != 0
    return seed % 3 == 0


def _base_stones_for_cell(row, col, base_size_x, base_size_y, base_height,
                          base_offset_x, base_offset_y):
    seed = (row + 1) * 41719 + (col + 1) * 11317
    rng = np.random.default_rng(seed)
    stone_count = 2 + (seed % 2)
    stones = []
    for stone_index in range(stone_count):
        shape_roll = (seed + stone_index) % 3
        geom_type = "box"
        if shape_roll == 0:
            size_x = base_size_x * (0.16 + 0.16 * rng.random())
            size_y = base_size_y * (0.16 + 0.16 * rng.random())
            size_z = base_height * (0.10 + 0.10 * rng.random())
        elif shape_roll == 1:
            size_x = base_size_x * (0.22 + 0.20 * rng.random())
            size_y = base_size_y * (0.10 + 0.10 * rng.random())
            size_z = base_height * (0.08 + 0.08 * rng.random())
        else:
            size_x = base_size_x * (0.10 + 0.10 * rng.random())
            size_y = base_size_y * (0.22 + 0.20 * rng.random())
            size_z = base_height * (0.08 + 0.08 * rng.random())

        size_str = f"{size_x} {size_y} {size_z}"
        footprint_x = size_x
        footprint_y = size_y
        top_height = size_z

        pos_x = base_offset_x + (rng.random() - 0.5) * max(0.05, base_size_x * 1.35 - footprint_x)
        pos_y = base_offset_y + (rng.random() - 0.5) * max(0.05, base_size_y * 1.35 - footprint_y)
        pos_z = base_height + top_height * (0.90 + 0.12 * stone_index)
        gray = 0.40 + 0.18 * rng.random()
        stones.append({
            "type": geom_type,
            "size": size_str,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z,
            "rgba": f"{gray:.3f} {gray:.3f} {gray + 0.03:.3f} 1",
        })
    return stones


def _add_base_stones(body, row, col, size_x, size_y, size_z, offset_x, offset_y):
    for stone_index, stone in enumerate(
        _base_stones_for_cell(row, col, size_x, size_y, size_z, offset_x, offset_y)
    ):
        ET.SubElement(body, "geom", {
            "name": f"stone_{row}_{col}_{stone_index}",
            "type": stone['type'],
            "size": stone['size'],
            "pos": f"{stone['pos_x']} {stone['pos_y']} {stone['pos_z']}",
            "rgba": stone['rgba'],
            "contype": "0",
            "conaffinity": "0"
        })


def _mountain_layers_for_cell(maze, row, col, base_size_x, base_size_y, base_height,
                              base_offset_x, base_offset_y):
    seed = (row + 1) * 92821 + (col + 1) * 68917
    rng = np.random.default_rng(seed)
    exposed_faces = _exposed_wall_faces(maze, row, col)
    if not exposed_faces:
        return []

    if not _should_add_mountain_wall(exposed_faces, seed):
        return []

    primary_face = exposed_faces[seed % len(exposed_faces)]
    secondary_face = None
    if len(exposed_faces) >= 2 and seed % 3 != 0:
        secondary_face = exposed_faces[(seed // 3) % len(exposed_faces)]
        if secondary_face == primary_face:
            secondary_face = None

    profile_roll = seed % 10
    if profile_roll <= 1:
        layer_count = 1
        height_scale = 0.75
        first_along_range = (0.82, 1.02)
        first_depth_range = (0.72, 0.96)
        taper_range = (0.92, 0.98)
        vertical_spacing = (0.92, 1.00)
    elif profile_roll <= 3:
        layer_count = 2
        height_scale = 0.82
        first_along_range = (0.62, 0.86)
        first_depth_range = (0.46, 0.70)
        taper_range = (0.84, 0.94)
        vertical_spacing = (0.82, 0.92)
    elif profile_roll <= 5:
        layer_count = 3
        height_scale = 0.90
        first_along_range = (0.50, 0.78)
        first_depth_range = (0.36, 0.60)
        taper_range = (0.80, 0.90)
        vertical_spacing = (0.78, 0.88)
    elif profile_roll <= 7:
        layer_count = 4
        height_scale = 0.98
        first_along_range = (0.44, 0.68)
        first_depth_range = (0.30, 0.52)
        taper_range = (0.76, 0.88)
        vertical_spacing = (0.74, 0.84)
    else:
        layer_count = 5
        height_scale = 1.08
        first_along_range = (0.38, 0.60)
        first_depth_range = (0.24, 0.44)
        taper_range = (0.72, 0.84)
        vertical_spacing = (0.70, 0.80)

    layers = []
    current_top_z = base_height
    current_size_x = None
    current_size_y = None
    current_pos_x = None
    current_pos_y = None
    for layer_index in range(layer_count):
        face_x, face_y = primary_face if layer_index == 0 or secondary_face is None else secondary_face
        tangent_x = -face_y
        tangent_y = face_x
        if current_size_x is None:
            along_size = (base_size_y if face_x else base_size_x) * (
                first_along_range[0] + (first_along_range[1] - first_along_range[0]) * rng.random()
            )
            depth_size = (base_size_x if face_x else base_size_y) * (
                first_depth_range[0] + (first_depth_range[1] - first_depth_range[0]) * rng.random()
            )
        else:
            taper = taper_range[0] + (taper_range[1] - taper_range[0]) * rng.random()
            along_size = current_size_y * taper if face_x else current_size_x * taper
            depth_size = current_size_x * (taper - 0.04 * rng.random()) if face_x else current_size_y * (taper - 0.04 * rng.random())
        layer_height = base_height * max(0.24, height_scale - 0.09 * layer_index + 0.10 * rng.random())

        if face_x:
            size_x = depth_size
            size_y = along_size
            if current_pos_x is None:
                edge_x = base_offset_x + face_x * base_size_x
                pos_x = edge_x - face_x * size_x * (0.78 + 0.12 * rng.random())
                pos_y = base_offset_y + tangent_y * (rng.random() - 0.5) * max(0.06, base_size_y - size_y)
            else:
                pos_x = current_pos_x - face_x * size_x * (0.10 + 0.08 * rng.random())
                pos_y = current_pos_y + tangent_y * (rng.random() - 0.5) * max(0.03, size_y * 0.25)
        else:
            size_x = along_size
            size_y = depth_size
            if current_pos_y is None:
                edge_y = base_offset_y + face_y * base_size_y
                pos_x = base_offset_x + tangent_x * (rng.random() - 0.5) * max(0.06, base_size_x - size_x)
                pos_y = edge_y - face_y * size_y * (0.78 + 0.12 * rng.random())
            else:
                pos_x = current_pos_x + tangent_x * (rng.random() - 0.5) * max(0.03, size_x * 0.25)
                pos_y = current_pos_y - face_y * size_y * (0.10 + 0.08 * rng.random())

        if secondary_face is not None and layer_index == layer_count - 1 and layer_count > 1:
            pos_x += secondary_face[0] * size_x * 0.35
            pos_y += secondary_face[1] * size_y * 0.35

        rgba_g = 0.22 + 0.06 * rng.random()
        rgba_b = 0.12 + 0.04 * rng.random()
        layers.append({
            "size_x": size_x,
            "size_y": size_y,
            "size_z": layer_height,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": current_top_z + layer_height,
            "rgba": f"0.31 {rgba_g:.3f} {rgba_b:.3f} 1",
        })
        current_top_z += 2.0 * layer_height * (
            vertical_spacing[0] + (vertical_spacing[1] - vertical_spacing[0]) * rng.random()
        )
        current_size_x = size_x
        current_size_y = size_y
        current_pos_x = pos_x
        current_pos_y = pos_y

    if len(exposed_faces) >= 2 and layer_count >= 2:
        peak_height = base_height * (0.30 + 0.18 * rng.random())
        peak_radius_x = max(base_size_x * 0.10, layers[-1]["size_x"] * (0.62 + 0.12 * rng.random()))
        peak_radius_y = max(base_size_y * 0.10, layers[-1]["size_y"] * (0.62 + 0.12 * rng.random()))
        layers.append({
            "size_x": peak_radius_x,
            "size_y": peak_radius_y,
            "size_z": peak_height,
            "pos_x": layers[-1]["pos_x"] * (0.60 + 0.16 * rng.random()),
            "pos_y": layers[-1]["pos_y"] * (0.60 + 0.16 * rng.random()),
            "pos_z": current_top_z + peak_height,
            "rgba": "0.38 0.29 0.18 1",
        })
    return layers


def _add_mountain_wall_layers(body, maze, row, col, size_x, size_y, size_z, offset_x, offset_y):
    layers = _mountain_layers_for_cell(
        maze, row, col, size_x, size_y, size_z, offset_x, offset_y
    )
    for layer_index, layer in enumerate(layers):
        ET.SubElement(body, "geom", {
            "name": f"mountain_{row}_{col}_{layer_index}",
            "type": "box",
            "size": f"{layer['size_x']} {layer['size_y']} {layer['size_z']}",
            "pos": f"{layer['pos_x']} {layer['pos_y']} {layer['pos_z']}",
            "rgba": layer['rgba'],
            "contype": "0",
            "conaffinity": "0"
        })


def _add_waterfall_floor_layer(worldbody, point, start_pos, box_size, patch_index):
    pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
    ET.SubElement(worldbody, "geom", {
        "name": f"waterfall_floor_{patch_index}",
        "type": "box",
        "size": f"{box_size[0] * 0.48} {box_size[1] * 0.48} 0.004",
        "pos": f"{pos_x} {pos_y} -0.0038",
        "rgba": WATERFALL_FLOOR_RGBA,
        "friction": WATERFALL_FLOOR_FRICTION,
        "priority": "1"
    })


def _add_ice_floor_layer(worldbody, point, start_pos, box_size, patch_index):
    pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
    ET.SubElement(worldbody, "geom", {
        "name": f"ice_floor_{patch_index}",
        "type": "box",
        "size": f"{box_size[0] * 0.48} {box_size[1] * 0.48} 0.0035",
        "pos": f"{pos_x} {pos_y} -0.0033",
        "rgba": ICE_FLOOR_RGBA,
        "friction": ICE_FLOOR_FRICTION,
        "priority": "1"
    })


def _add_waterfall_patch(worldbody, point, start_pos, box_size, patch_index):
    pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
    _add_waterfall_floor_layer(worldbody, point, start_pos, box_size, patch_index)
    patch = ET.SubElement(worldbody, "body", {
        "name": f"waterfall_patch_{patch_index}",
        "pos": f"{pos_x} {pos_y} 0.0"
    })
    ET.SubElement(patch, "geom", {
        "name": f"waterfall_base_{patch_index}",
        "type": "box",
        "size": "0.48 0.48 0.028",
        "pos": "0 0 0.028",
        "rgba": "0.10 0.46 0.96 0.55",
        "contype": "0",
        "conaffinity": "0"
    })
    ET.SubElement(patch, "geom", {
        "name": f"waterfall_core_{patch_index}",
        "type": "box",
        "size": "0.40 0.40 0.016",
        "pos": "0 0 0.055",
        "rgba": "0.22 0.68 1.00 0.42",
        "contype": "0",
        "conaffinity": "0"
    })

    strip_offsets = (-0.24, 0.0, 0.24)
    for strip_index, offset in enumerate(strip_offsets):
        strip = ET.SubElement(patch, "body", {
            "name": f"waterfall_strip_{patch_index}_{strip_index}",
            "pos": f"{offset} 0 0.060"
        })
        ET.SubElement(strip, "joint", {
            "name": f"waterfall_slide_{patch_index}_{strip_index}",
            "type": "slide",
            "axis": "1 0 0",
            "range": "-0.30 0.30",
            "damping": "0"
        })
        ET.SubElement(strip, "geom", {
            "name": f"waterfall_strip_geom_{patch_index}_{strip_index}",
            "type": "box",
            "size": "0.055 0.46 0.020",
            "rgba": "0.96 0.99 1.00 0.78",
            "contype": "0",
            "conaffinity": "0"
        })

    for band_index, offset in enumerate((-0.16, 0.16)):
        band = ET.SubElement(patch, "body", {
            "name": f"waterfall_band_{patch_index}_{band_index}",
            "pos": f"{offset} 0 0.045"
        })
        ET.SubElement(band, "joint", {
            "name": f"waterfall_band_slide_{patch_index}_{band_index}",
            "type": "slide",
            "axis": "1 0 0",
            "range": "-0.24 0.24",
            "damping": "0"
        })
        ET.SubElement(band, "geom", {
            "name": f"waterfall_band_geom_{patch_index}_{band_index}",
            "type": "box",
            "size": "0.085 0.42 0.014",
            "rgba": "0.28 0.78 1.00 0.52",
            "contype": "0",
            "conaffinity": "0"
        })


def _add_waterfall_patches(worldbody, waterfall_points, start_pos, box_size):
    for patch_index, point in enumerate(waterfall_points):
        _add_waterfall_patch(worldbody, point, start_pos, box_size, patch_index)


def _add_ice_floor_patches(worldbody, ice_points, start_pos, box_size):
    for patch_index, point in enumerate(ice_points):
        _add_ice_floor_layer(worldbody, point, start_pos, box_size, patch_index)


def make_maze_on_mujoco(load_file_path, maze, start_pos, goal_pos, waypoints=None,
                        box_size=[0.5, 0.5, 0.15], maze_rgba="0.45 0.29 0.18 1",
                        goal_rgba="0.6 0.9 0.6 0.5", waypoint_rgba="1 1 0 0.5",
                        save_file_path=None):
    tree = ET.parse(load_file_path)
    root = tree.getroot()
    worldbody = root.find("./worldbody")

    # Remove old maze elements so we can rebuild them cleanly.
    for body in list(worldbody.findall("body")):
        body_name = body.get('name') or ""
        if 'box_' in body_name or 'goal' in body_name or 'fake_wall' in body_name or 'waterfall_' in body_name:
            worldbody.remove(body)
    for geom in list(worldbody.findall("geom")):
        geom_name = geom.get('name') or ""
        if 'waypoint_' in geom_name or 'path_narrow_' in geom_name or 'fake_wall_' in geom_name or 'waterfall_' in geom_name or 'corridor_floor_' in geom_name or 'ice_floor_' in geom_name:
            worldbody.remove(geom)

    rng = np.random.default_rng()
    waterfall_points = _select_waterfall_points(waypoints, start_pos, goal_pos, rng)
    ice_points = _select_ice_floor_points(
        maze, start_pos, goal_pos, rng, blocked_points=waterfall_points
    )
    narrow_cells = _select_narrow_path_cells(waypoints, start_pos, goal_pos, rng, blocked_points=waterfall_points)

    # Create the maze walls.
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j] == 1:
                pos_x = 2 * box_size[0] * (j - start_pos[0])
                pos_y = -2 * box_size[1] * (i - start_pos[1])
                size_x, size_y, size_z, offset_x, offset_y = _wall_geom_for_cell(
                    maze, i, j, box_size, narrow_cells
                )
                body = ET.SubElement(worldbody, "body", {
                    "name": f"box_{i}_{j}",
                    "pos": f"{pos_x} {pos_y} {box_size[2]}"
                })
                ET.SubElement(body, "geom", {
                    "name": f"geom_{i}_{j}",
                    "type": "box",
                    "size": f"{size_x} {size_y} {size_z}",
                    "pos": f"{offset_x} {offset_y} 0",
                    "rgba": maze_rgba,
                    "friction": "0.9 0.05 0.01"
                })
                _add_base_stones(body, i, j, size_x, size_y, size_z, offset_x, offset_y)
                _add_mountain_wall_layers(body, maze, i, j, size_x, size_y, size_z, offset_x, offset_y)

    # Add visual waypoint markers.
    if waypoints:
        for idx, point in enumerate(waypoints):
            if list(point) == start_pos or list(point) == goal_pos:
                continue
            pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
            pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
            ET.SubElement(worldbody, "geom", {
                "name": f"waypoint_{idx}",
                "type": "sphere",
                "size": "0.1",
                "pos": f"{pos_x} {pos_y} 0.1",
                "rgba": waypoint_rgba,
                "contype": "0",
                "conaffinity": "0"
            })

    _add_waterfall_patches(worldbody, waterfall_points, start_pos, box_size)
    _add_ice_floor_patches(worldbody, ice_points, start_pos, box_size)

    # Create the goal.
    pos_x = 2 * box_size[0] * (goal_pos[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (goal_pos[1] - start_pos[1])
    body = ET.SubElement(worldbody, "body", {"name": "goal", "pos": f"{pos_x} {pos_y} {box_size[2]}"})
    ET.SubElement(body, "geom", {"name": "goal_geom", "type": "box", "size": f"{box_size[0]} {box_size[1]} {box_size[2]}", "contype": "0", "conaffinity": "0", "rgba": goal_rgba})

    if save_file_path:
        tree.write(save_file_path, encoding="utf-8", xml_declaration=True)


def update_maze(load_file_path, start_pos, goal_pos, waypoints=None, box_size=[0.5, 0.5, 0.15], goal_rgba="0.6 0.9 0.6 0.5", waypoint_rgba="1 1 0 0.5", save_file_path=None):
    tree = ET.parse(load_file_path)
    root = tree.getroot()
    worldbody = root.find("./worldbody")

    # Remove the previous goal and waypoint markers.
    for body in list(worldbody.findall("body")):
        body_name = body.get('name') or ""
        if body_name == 'goal' or 'waterfall_' in body_name:
            worldbody.remove(body)

    for geom in list(worldbody.findall("geom")):
        geom_name = geom.get('name') or ""
        if 'waypoint_' in geom_name or 'waterfall_' in geom_name:
            worldbody.remove(geom)

    # Add visual waypoint markers.
    if waypoints:
        for idx, point in enumerate(waypoints):
            if list(point) == start_pos or list(point) == goal_pos:
                continue
            pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
            pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
            ET.SubElement(worldbody, "geom", {"name": f"waypoint_{idx}", "type": "sphere", "size": "0.1", "pos": f"{pos_x} {pos_y} 0.1", "rgba": waypoint_rgba, "contype": "0", "conaffinity": "0"})

    # Create the goal.
    pos_x = 2 * box_size[0] * (goal_pos[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (goal_pos[1] - start_pos[1])
    body = ET.SubElement(worldbody, "body", {"name": "goal", "pos": f"{pos_x} {pos_y} {box_size[2]}"})
    ET.SubElement(body, "geom", {"name": "goal_geom", "type": "box", "size": f"{box_size[0]} {box_size[1]} {box_size[2]}", "contype": "0", "conaffinity": "0", "rgba": goal_rgba})

    rng = np.random.default_rng()
    waterfall_points = _select_waterfall_points(waypoints, start_pos, goal_pos, rng)
    _add_waterfall_patches(worldbody, waterfall_points, start_pos, box_size)

    output_path = save_file_path if save_file_path is not None else load_file_path
    tree.write(output_path, encoding="utf-8", xml_declaration=True)