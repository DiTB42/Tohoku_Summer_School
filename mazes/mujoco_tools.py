import xml.etree.ElementTree as ET
import numpy as np

def make_maze_on_mujoco(load_file_path, maze, start_pos, goal_pos, waypoints=None, box_size=[0.5, 0.5, 0.15], maze_rgba="0.5 0.5 0.5 1", goal_rgba="0.6 0.9 0.6 0.5", waypoint_rgba="1 1 0 0.5", save_file_path=None):
  tree = ET.parse(load_file_path)
  root = tree.getroot()
  worldbody = root.find("./worldbody")

  # Rimuovi i vecchi elementi del labirinto per rigenerarlo
  for body in worldbody.findall("body"):
      if body.get('name') and ('box_' in body.get('name') or 'goal' in body.get('name')):
          worldbody.remove(body)
  for geom in worldbody.findall("geom"):
      if geom.get('name') and 'waypoint_' in geom.get('name'):
          worldbody.remove(geom)

  # Crea i muri del labirinto
  for i in range(maze.shape[0]):
    for j in range(maze.shape[1]):
      if maze[i][j] == 1:
        pos_x = 2 * box_size[0] * (j - start_pos[0])
        pos_y = -2 * box_size[1] * (i - start_pos[1])
        body = ET.SubElement(worldbody, "body", {"name": f"box_{i}_{j}", "pos": f"{pos_x} {pos_y} {box_size[2]}"})
        ET.SubElement(body, "geom", {"name": f"geom_{i}_{j}", "type": "box", "size": f"{box_size[0]} {box_size[1]} {box_size[2]}", "rgba": maze_rgba})

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