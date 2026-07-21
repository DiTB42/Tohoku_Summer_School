import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_FLOOR_RGBA = "0.18 0.48 0.16 0.01"
DEFAULT_FLOOR_FRICTION = "3 0.005 0.0001"
ICE_FLOOR_RGBA = "0.72 0.90 1.00 0.34"
ICE_FLOOR_FRICTION = "0.15 0.00004 0.00001"
WATERFALL_FLOOR_RGBA = "0.08 0.36 0.78 0.36"
WATERFALL_FLOOR_FRICTION = "15.0 0.0375 0.0005"


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

    candidate_set = set(candidates)
    center_count = min(max(2, len(candidates) // 16), max(2, len(candidates) // 5))
    center_indices = rng.choice(len(candidates), size=center_count, replace=False)

    ice_points = set()
    neighbor_groups = [
        ([(-1, 0), (1, 0), (0, -1), (0, 1)], 0.72),
        ([(-1, -1), (-1, 1), (1, -1), (1, 1)], 0.48),
        ([(-2, 0), (2, 0), (0, -2), (0, 2)], 0.30),
        ([(-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)], 0.18),
    ]
    for center_index in np.atleast_1d(center_indices).tolist():
        center = candidates[center_index]
        ice_points.add(center)
        for offsets, probability in neighbor_groups:
            for d_col, d_row in offsets:
                neighbor = (center[0] + d_col, center[1] + d_row)
                if neighbor in candidate_set and rng.random() < probability:
                    ice_points.add(neighbor)

    return sorted(ice_points)


def _add_corridor_floor_tiles(worldbody, maze, start_pos, box_size, waterfall_points, ice_points):
    waterfall_set = set(waterfall_points)
    ice_set = set(ice_points)
    tile_half_x = box_size[0] * 0.5
    tile_half_y = box_size[1] * 0.5
    tile_half_z = 0.001
    tile_pos_z = -0.001

    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):
            if maze[row][col] != 0:
                continue
            pos_x = 2 * box_size[0] * (col - start_pos[0])
            pos_y = -2 * box_size[1] * (row - start_pos[1])
            point = (col, row)
            if point in waterfall_set:
                rgba = WATERFALL_FLOOR_RGBA
                friction = WATERFALL_FLOOR_FRICTION
                name_prefix = "waterfall_floor"
            elif point in ice_set:
                rgba = ICE_FLOOR_RGBA
                friction = ICE_FLOOR_FRICTION
                name_prefix = "ice_floor"
            else:
                rgba = DEFAULT_FLOOR_RGBA
                friction = DEFAULT_FLOOR_FRICTION
                name_prefix = "corridor_floor"
            ET.SubElement(worldbody, "geom", {
                "name": f"{name_prefix}_{row}_{col}",
                "type": "box",
                "size": f"{tile_half_x} {tile_half_y} {tile_half_z}",
                "pos": f"{pos_x} {pos_y} {tile_pos_z}",
                "rgba": rgba,
                "friction": friction,
                "priority": "1"
            })


def _add_waterfall_patch(worldbody, point, start_pos, box_size, patch_index):
    pos_x = 2 * box_size[0] * (point[0] - start_pos[0])
    pos_y = -2 * box_size[1] * (point[1] - start_pos[1])
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
    narrow_cells = _select_narrow_path_cells(waypoints, start_pos, goal_pos, rng, blocked_points=waterfall_points)
    ice_points = _select_ice_floor_points(
        maze, start_pos, goal_pos, rng, blocked_points=waterfall_points
    )

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

    _add_corridor_floor_tiles(worldbody, maze, start_pos, box_size, waterfall_points, ice_points)

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