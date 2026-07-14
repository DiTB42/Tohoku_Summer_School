import random
import numpy as np
import heapq

# --- FUNZIONI DI UTILITÀ PER LA GENERAZIONE DEL LABIRINTO ---

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

def create_maze_layout(height, width):
    """Genera la struttura di un nuovo labirinto casuale."""
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