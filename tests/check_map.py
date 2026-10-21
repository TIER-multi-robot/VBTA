import argparse
import os
from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list

def map_info(path):
    full_path = os.path.join("MAPF_benchmark_maps", path)
    grid = load_map(full_path)
    dims = (len(grid), len(grid[0]))
    obs = set(create_obstacle_list(grid))
    map_dict = {"dimensions": dims, "obstacles": list(obs)}
    free_cells = [(r, c) for r in range(dims[0]) for c in range(dims[1]) if (r, c) not in obs]
    print(f"{path}: {dims[0]}x{dims[1]} free={len(free_cells)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to the *.map file")
    args = parser.parse_args()
    map_info(args.path)

if __name__ == "__main__":
    main()