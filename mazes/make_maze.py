import random
import numpy as np
import heapq
from collections import deque

# --- FUNZIONI DI UTILITÀ PER LA GENERAZIONE DEL LABIRINTO ---

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def create_maze_layout(height, width):
    """Genera la struttura di un nuovo labirinto casuale.

    Even dimensions are normalized UP to the next odd (10 -> 11): with an even
    dim, _carve's `1 <= n + 2*step < dim` bound lets the 2-step carve land on
    the border row/col (index dim-1), leaving the maze with NO outer wall on
    the bottom/right sides. The next odd size has the same junction grid but
    keeps the full perimeter walled."""
    height += 1 - height % 2
    width += 1 - width % 2
    maze = [[1] * width for _ in range(height)]
    maze[1][1] = 0
    
    def _carve(ny, nx):
        array = list(range(4))
        random.shuffle(array)
        dx_gen = [(1, 2), (-1, -2), (0, 0), (0, 0)]
        dy_gen = [(0, 0), (0, 0), (1, 2), (-1, -2)]
        for i in array:
            if not (1 <= ny + dy_gen[i][1] < height and 1 <= nx + dx_gen[i][1] < width):
                continue
            if maze[ny + dy_gen[i][1]][nx + dx_gen[i][1]] == 0:
                continue
            for j in range(2):
                maze[ny + dy_gen[i][j]][nx + dx_gen[i][j]] = 0
            _carve(ny + dy_gen[i][1], nx + dx_gen[i][1])

    _carve(1, 1)
    return maze

def get_valid_spawn_points(maze):
    """Restituisce una lista di tutte le coordinate (x, y) dei corridoi."""
    valid_points = []
    height, width = len(maze), len(maze[0])
    for y in range(height):
        for x in range(width):
            if maze[y][x] == 0:
                valid_points.append((x, y))
    return valid_points

TERRAIN_TYPES = ("grass", "ice", "dirt")
# Per-type probability that a newly reached cell KEEPS its parent's terrain.
# Grass forms large stable patches (0.8); ice and dirt are less persistent
# (0.65), so slippery/rough stretches stay short.
TERRAIN_KEEP_PROB = {"grass": 0.8, "ice": 0.65, "dirt": 0.65}

def generate_terrain(maze, start, rng, keep_prob=None):
    """Assign a terrain type to every open corridor cell via BFS flood-fill.

    The start cell is ALWAYS 'grass'. Each newly discovered cell keeps the
    terrain of the neighbor it was reached from with probability
    `keep_prob[parent_terrain]` (default TERRAIN_KEEP_PROB: grass 0.8, ice and
    dirt 0.65), otherwise it switches uniformly to one of the other two types
    (grass 10%/10%; ice/dirt 17.5%/17.5%), producing spatially correlated
    patches with shorter ice/dirt runs.

    Coordinates are (x, y) == (col, row), matching cell_distances / astar.
    `rng` must be a numpy Generator (pass the env's self.np_random so terrain
    is reproducible from reset(seed=...)); do NOT use the module-level
    `random` that create_maze_layout uses.

    Returns {(x, y): 'grass' | 'ice' | 'dirt'} covering every open cell
    reachable from `start` (all of them, for a perfect maze)."""
    if keep_prob is None:
        keep_prob = TERRAIN_KEEP_PROB
    terrain = {start: "grass"}
    q = deque([start])
    height, width = len(maze), len(maze[0])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0 and (nx, ny) not in terrain:
                parent_terrain = terrain[(x, y)]
                if rng.random() < keep_prob[parent_terrain]:
                    terrain[(nx, ny)] = parent_terrain
                else:
                    others = [t for t in TERRAIN_TYPES if t != parent_terrain]
                    terrain[(nx, ny)] = others[int(rng.integers(0, len(others)))]
                q.append((nx, ny))
    return terrain

# --- CAVE (variable corridor width) GENERATION ---
# A "cave" is a NARROWER section of an otherwise 1-cell corridor (a pinch/funnel,
# like the kb branch). It is realized in mujoco_tools.make_maze_on_mujoco by
# growing the two flanking wall boxes INWARD symmetrically (outer faces fixed) so
# the passage squeezes down while its centerline stays on the integer grid
# ("entrance in the middle"). generate_widths only marks cells; the geometry (and
# the obs) read the per-cell extension `e`.
CAVE_PROB = 0.32                    # per eligible straight cell, chance to START a cave
CAVE_MIN_LEN, CAVE_MAX_LEN = 1, 7   # cave run length in path cells (1 => single-cell caves allowed)
# Extension `e` per side, in world units: each flank wall grows `e` into the
# corridor, so the open span becomes 1.0 - 2e. Each cave draws its OWN peak
# tightness uniformly in [CAVE_MIN_EXT, CAVE_MAX_EXT], so some pinch hard (span
# ~1-2*0.44 = 0.12) and some less (span ~1-2*0.2 = 0.6). Capped at 0.44 (< 0.5,
# so the two flank walls never meet at the centerline): the tightest passage is
# ~0.12 units, still ~3x the snake's ~0.04-unit body width but a hard squeeze
# for its serpentine undulation.
CAVE_MIN_EXT, CAVE_MAX_EXT = 0.2, 0.44
# Multi-cell caves taper to CAVE_MOUTH_FRAC of their peak at the two mouths (a
# single-cell cave is all "center"), keeping the entrance in the middle.
CAVE_MOUTH_FRAC = 0.5

def generate_widths(maze, path, rng, cave_prob=CAVE_PROB,
                    min_len=CAVE_MIN_LEN, max_len=CAVE_MAX_LEN,
                    min_ext=CAVE_MIN_EXT, max_ext=CAVE_MAX_EXT,
                    mouth_frac=CAVE_MOUTH_FRAC):
    """Probabilistically place narrowing "caves" along straight segments of `path`.

    Returns {(x, y): e} == {(col, row): extension} for every narrowed cell, where
    the open corridor span at that cell is 1.0 - 2*e world units. Only cells that
    the snake actually traverses are considered (the A* solution `path`), skipping
    the start (index 0) and goal (index -1).

    A cell is eligible to be narrowed only if it is on a STRAIGHT stretch (no turn
    at it) AND both of its perpendicular neighbors are walls — i.e. a plain 1-wide
    corridor cell, not a turn or a junction. That guarantees there is a wall on each
    side to grow inward and keeps the pinch symmetric about the corridor centerline.

    Each cave is a contiguous run of 1..`max_len` same-orientation eligible cells
    (length drawn per cave, so single-cell pinches are common), with its OWN peak
    tightness drawn uniformly in `[min_ext, max_ext]` (so caves vary — some pinch
    hard, some barely). Multi-cell caves taper from `mouth_frac`*peak at the two
    mouths up to the full peak at the center, so the entrance sits in the middle.

    Coordinates are (x, y) == (col, row), matching astar / generate_terrain. `rng`
    must be a numpy Generator (pass the env's self.np_random for reproducibility);
    do NOT use the module-level `random`."""
    widths = {}
    if not path or len(path) < 3:
        return widths
    height, width = len(maze), len(maze[0])

    def _orientation(i):
        """'horizontal'/'vertical'/None for path cell `i` (None => turn/end)."""
        if i <= 0 or i >= len(path) - 1:
            return None
        (px, py), (cx, cy), (nx, ny) = path[i - 1], path[i], path[i + 1]
        in_d = (cx - px, cy - py)
        out_d = (nx - cx, ny - cy)
        if in_d != out_d:
            return None  # turn
        return "horizontal" if in_d[0] != 0 else "vertical"

    def _eligible(i):
        orient = _orientation(i)
        if orient is None:
            return None
        col, row = path[i]
        if orient == "horizontal":
            ok = maze[row - 1][col] == 1 and maze[row + 1][col] == 1
        else:
            ok = maze[row][col - 1] == 1 and maze[row][col + 1] == 1
        return orient if ok else None

    i = 1
    while i < len(path) - 1:
        orient = _eligible(i)
        if orient is None or rng.random() >= cave_prob:
            i += 1
            continue
        # This cave's own random target length (>=1) and peak tightness, so caves
        # differ in both size and how hard they pinch.
        target = int(rng.integers(min_len, max_len + 1))
        peak = min_ext + (max_ext - min_ext) * rng.random()
        # Greedily take up to `target` consecutive same-orientation eligible cells.
        j = i
        while j < len(path) - 1 and j - i < target and _eligible(j) == orient:
            j += 1
        run = list(range(i, j))
        half = (len(run) - 1) / 2.0
        for k, idx in enumerate(run):
            # Taper: full peak at the center, down to mouth_frac*peak at the two
            # mouths (single-cell cave => all center).
            shape = 1.0 if half == 0 else 1.0 - (1.0 - mouth_frac) * abs(k - half) / half
            widths[path[idx]] = peak * shape
        i = j
    return widths

def cell_distances(maze, start):
    """BFS shortest-path distance (in moves) from `start` to every reachable corridor cell.

    Restituisce {(x, y): dist}. Le coordinate sono (x, y) == (col, row), come per
    astar / get_valid_spawn_points. Poiché il labirinto è "perfetto" (un albero), la
    distanza BFS coincide con la lunghezza del percorso A* fino a quella cella.
    """
    dist = {start: 0}
    q = deque([start])
    height, width = len(maze), len(maze[0])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0 and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                q.append((nx, ny))
    return dist

def astar(maze, start, goal):
    """Algoritmo A* per trovare il percorso più breve."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    height, width = len(maze), len(maze[0])

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from.get(current)
            return path[::-1]
        
        x, y = current
        for dx, dy in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor
            if not (0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0):
                continue
            new_cost = cost + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, new_cost, neighbor))
                came_from[neighbor] = current
    return []