"""Helper functions for FinalCBS.py"""

import random
from typing import List, Tuple
from hvbta.models import CapabilityProfile

def load_map(map_filename):
    """Function to load ascii maps from the MAPF benchmark .map files"""
    with open(map_filename, 'r') as f:
        _type = f.readline().strip() # Type octile or similar
        height_line = f.readline().strip() # Height of map e.g. "height 45"
        width_line = f.readline().strip() # Width of map e.g. "width 52"
        map_line = f.readline().strip() # delinates start of map with line that says "map"

        height = int(height_line.split()[1])
        width = int(width_line.split()[1])

        grid = []
        for _ in range(height):
            row_data = f.readline().strip()
            if len(row_data) != width:
                raise ValueError(f"Map file error: row length {len(row_data)} != width {width}") # sanity check for malformed map files
            # periods and G's are traversable terrain, everything else will be unpassable, there are 5 types of unpassable terrains, water will be unpassable
            # row = [0 if c == '.' or c == 'G' else 1 for c in row_data] 
            # 1 means blocked, 0 means free
            row = []
            for c in row_data:
                if c in [".", 'G']:
                    row.append(0)
                else:
                    row.append(1)
            grid.append(row)

    return grid

def create_obstacle_list(grid):
    """Create a reusable obstacle list from the grid for fast lookups"""
    obstacle_list = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == 1:
                obstacle_list.append((r, c)) # store all obstacles in a list
    return obstacle_list

def load_scenario(scenario_filename):
    """Function to load scenarios from the MAPF benchmark .scen files, not using .scen files for now, just randomly assigning agents"""
    agents = []
    with open(scenario_filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue # skip comments
            parts = line.strip().split()
            # .scen files are in format "0 mapName.map 5 10 30 35 42" in order these are the bucket, map file name, start row, start col, goal row, goal col, optimal distance
            start_row = int(parts[2])
            start_col = int(parts[3])
            goal_row = int(parts[4])
            goal_col = int(parts[5])
            agents.append((start_row, start_col, goal_row, goal_col))
    return agents

def get_random_free_position(grid, occupied_positions):
    """
    Parameters
    - grid: 2D list of 0/1 cells representing the map free/obstacles
    - occupied_positions: exisiting agent positions, places we want to consider
      blocked when choosing a new position

    Returns: a single (row, col) position for one agent,
            randomly from free cells with value = 0 that are not occupied
    """
    free_cells = []
    # iterate thru entire grid and find the free cells, make a list of them for choosing from
    for r in range(len(grid)):
        # len(grid[0]) assumes all rows are the same length as the first row
        # if not, we can iterate through each row's length
        for c in range(len(grid[r])):
            if grid[r][c] == 0 and (r, c) not in occupied_positions:
                free_cells.append((r, c))

    if not free_cells:
       raise ValueError("No free cells available to place a robot!")
    
    # randomly sample without replacement
    # chosen = random.sample(free_cells, 1)
    # return chosen
    return random.choice(free_cells)  # return a single random free cell from the list

def build_cbs_agents(robots: List[CapabilityProfile], start_positions: dict, goal_positions: dict) -> list[dict]:
    """
    Return a fresh list of {'name','start','goal'} dicts for CBS,
    based on whatever robots are currently marked assigned.
    """
    return [
        {
            'name': r.robot_id,
            'start': start_positions[r.robot_id],
            'goal':  goal_positions[r.robot_id],
        }
        for r in robots
        if r.assigned and r.current_task is not None
    ]