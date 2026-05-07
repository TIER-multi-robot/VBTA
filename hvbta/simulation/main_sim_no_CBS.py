import psutil
import time
import random
import numpy as np
import csv
import os
import json
from typing import List
import copy
from numba import jit
from numba.typed import List
from hvbta.pathfinding.Final_CBS import CBS, Environment
from hvbta.simulation.timestep_no_CBS import simulate_time_step
import hvbta.allocators.voting as V
import hvbta.suitability as S
from hvbta.suitability import (
    calculate_jains_index,
    calculate_threshold_metrics,
    calculate_inequality_metrics,
    calculate_robustness_metrics
)
from hvbta.pathfinding.CBS import load_map, create_obstacle_list, build_cbs_agents
from hvbta.allocators.misc_assignment import add_new_tasks, add_new_robots, remove_random_robots
import hvbta.generation as G
from hvbta.models import CapabilityProfile, TaskDescription
import heapq
import concurrent.futures
import hvbta.allocators.optimizers as O


def compute_all_fairness_metrics(scores):
    """
    Compute all fairness metrics for a list of scores.
    Returns a flat tuple of values in consistent order for CSV output.
    """
    jains = calculate_jains_index(scores)
    threshold = calculate_threshold_metrics(scores)
    inequality = calculate_inequality_metrics(scores)
    robustness = calculate_robustness_metrics(scores)
    
    return (
        jains,
        threshold["below_ge_frac"],
        threshold["below_good_frac"],
        threshold["deficit_all_ge"],
        threshold["deficit_below_ge"],
        threshold["deficit_all_good"],
        threshold["deficit_below_good"],
        inequality["score_range"],
        inequality["min_max_ratio"],
        inequality["gini"],
        inequality["cv"],
        robustness["median"],
        robustness["mean"],
        robustness["med_mean_gap"],
        robustness["iqr"]
    )


# Column names for all fairness metrics (for CSV headers)
FAIRNESS_METRIC_NAMES = [
    "Jains_Index",
    "Below_GE_Frac",
    "Below_Good_Frac",
    "Deficit_All_GE",
    "Deficit_Below_GE",
    "Deficit_All_Good",
    "Deficit_Below_Good",
    "Score_Range",
    "Min_Max_Ratio",
    "Gini",
    "CV",
    "Median",
    "Mean",
    "Med_Mean_Gap",
    "IQR"
]


def func_name(f):
    return getattr(f, "__name__", str(f))

def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)

# In main_sim_no_CBS.py

@jit(nopython=True)
def astar(start, goal, obstacle_array):
    rows, cols = obstacle_array.shape
    
    # Using a typed List and Dict for Numba compatibility
    open_heap = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start)]
    came_from = {start: (-1, -1)} # dummy value for the start node
    gscore = {start: 0}

    while len(open_heap) > 0:
        f, g, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current != (-1, -1):
                path.append(current)
                current = came_from[current]
            return path[::-1] # list of tuples

        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            
            # check that nx and ny are in bounds (0 <= nx < rows) and not on an obstacle in the bool obstacle_array (0 for free, 1 for obstacle)
            if 0 <= nx < rows and 0 <= ny < cols and not obstacle_array[nx, ny]:
                neigh = (nx, ny)
                tentative_g = g + 1
                
                # Replace np.inf (a float) with a very large integer.
                if tentative_g < gscore.get(neigh, 999999999):
                    came_from[neigh] = current
                    gscore[neigh] = tentative_g
                    new_f = tentative_g + (abs(neigh[0] - goal[0]) + abs(neigh[1] - goal[1]))
                    heapq.heappush(open_heap, (new_f, tentative_g, neigh))

    return [(-1, -1)] # no path found, dummy path returned

def _compute_agent_path(args):
    # args: (rid, start, goal, obstacle_array)
    rid, start, goal, obstacle_array = args
    try:
        path = astar(start, goal, obstacle_array)
    except Exception as e:
        print(f"Error in A* for agent {rid}: {e}")
        path = None
    return (rid, path)

def state_check(robots: List[CapabilityProfile]):
    """
    Returns a state of the robots for deciding whether to re-plan
    Ignores anything that will cause constant replanning
    Includes who is planned and to which goals
    """
    active = []
    goals = []
    for r in robots:
        if r.assigned and r.current_task:
            active.append(r.robot_id)
            goals.append((r.robot_id, tuple(r.current_task.location), r.current_task.task_id))
    active_signature = tuple(sorted(active))
    goals_signature = tuple(sorted(goals))
    return active_signature, goals_signature

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def main_simulation(
        output: tuple[list[tuple[int, int]],list[int],list[int]], 
        robots: List[CapabilityProfile], tasks: List[TaskDescription], 
        num_candidates: int, 
        voting_method: callable, 
        grid: List[List[int]], 
        map_dict: dict, 
        suitability_method: callable, 
        suitability_matrix: np.ndarray, 
        max_time_steps: int, 
        add_tasks: bool, 
        add_robots: bool, 
        remove_robots: bool, 
        tasks_to_add: int = 1, 
        robots_to_add: int = 1, 
        robots_to_remove: int = 1,
        robot_generation_strict: bool = True,
        task_generation_strict: bool = True,
        # If True, use ThreadPoolExecutor instead of ProcessPoolExecutor for parallel A*.
        # Threads avoid multiprocessing pickling/Windows spawn issues but may be slower
        # for CPU-bound A* due to the GIL. Default is False (use processes).
        use_threads: bool = True,
        initial_jains_index: float = 0.0,):
    print(f"SUITABILITY METHOD: {suitability_method}")

    voting_methods = {
        V.rank_assignments_borda: "Borda Count",
        V.rank_assignments_approval: "Approval Voting",
        V.rank_assignments_majority_judgment: "Majority Judgment",
        V.rank_assignments_cumulative_voting: "Cumulative Voting",
        V.rank_assignments_condorcet_method: "Condorcet Method",
        V.rank_assignments_range: "Range Voting"
    }
    if voting_method in voting_methods:
        voting_method_name = voting_methods.get(voting_method, "Unknown Method")
    optimization_methods = {
        O.cbba_task_allocation: "CBBA",
        O.ssia_task_allocation: "SSIA",
        O.ilp_task_allocation: "ILP",
        O.jv_task_allocation: "JV"
    }
    if voting_method in optimization_methods:
        optimization_method_name = optimization_methods.get(voting_method, "Unknown Method")

    initial_positions = set()
    robot_max_id = len(robots)+1
    task_max_id = len(tasks)+1
    total_reward = 0.0
    total_success = 0.0
    total_tasks = len(tasks)
    total_reassignment_time = 0.0
    total_reassignment_score = 0.0
    total_reassignments = 0
    total_time_steps = max_time_steps
    reassignment_jains_indices = []  # Track per-event fairness for reassignments
    
    # Track all fairness metrics for reassignments (list of tuples from compute_all_fairness_metrics)
    reassignment_fairness_metrics = []
    
    # LLM suitability cache - reuse matrix as long as all current robots/tasks are in the cache
    # This allows cache reuse even when tasks are completed, only invalidates when NEW entities are added
    llm_cache = {
        "matrix": suitability_matrix if getattr(suitability_method, "_is_llm_batch", False) else None,
        "robot_id_to_idx": {r.robot_id: i for i, r in enumerate(robots)} if getattr(suitability_method, "_is_llm_batch", False) else None,
        "task_id_to_idx": {t.task_id: j for j, t in enumerate(tasks)} if getattr(suitability_method, "_is_llm_batch", False) else None,
    }

    for r in robots:
        # add the robot's initial position to the occupied positions set
        initial_positions.add(r.location)

    occupied_positions = set(initial_positions) # use the occcupied positions as the current positions for CBS, this is just as a occupation check, not a start position

    assigned_pairs = output[0]
    for robot_idx, task_idx in assigned_pairs:
        r = robots[robot_idx]
        t = tasks[task_idx]

        r.current_task = t
        r.assigned = True
        r.tasks_attempted = 1

        t.assigned_robot = r
        t.assigned = True


    assigned_robots = {r.robot_id: r.current_task.task_id for r in
                       robots if r.assigned and r.current_task is not None}
    unassigned_tasks = [t.task_id for t in tasks if not t.assigned]
    unassigned_robots = [r.robot_id for r in robots if not r.assigned]

    start_positions = {
        r.robot_id: r.location
        for r in robots
        if r.assigned and r.current_task is not None
    }
    goal_positions = {
        r.robot_id: r.current_task.location
        for r in robots
        if r.assigned and r.current_task is not None
    }

    agents = build_cbs_agents(robots, start_positions, goal_positions)

    # Build obstacle set and dims
    dims = map_dict['dimension']
    obstacle_array = np.array(grid, dtype=np.bool_)

    # Compute A* path for each assigned agent independently (parallelized)
    # agents is a list of dicts with 'name', 'start', 'goal' per build_cbs_agents output
    args = []
    for a in agents:
        rid = a['name'].robot_id if hasattr(a['name'], 'robot_id') else a['name']
        start = tuple(a['start'])
        goal = tuple(a['goal'])
        args.append((rid, start, goal, obstacle_array))

    solution = {}
    if args:
        max_workers = min(len(args), (os.cpu_count() or 1))
        ExecutorClass = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
        with ExecutorClass(max_workers=max_workers) as ex:
            for rid, path in ex.map(_compute_agent_path, args):
                if path:
                    solution[rid] = path
                else:
                    print(f"Warning: no A* path found for agent {rid}")
                    solution[rid] = None

    # NOTE: add an int that can return avg path length for analysis
    # Assign computed paths back to robot objects
    id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
    for robot_id, path in solution.items():
        if path is None:
            continue
        if robot_id in id_to_index:
            r = robots[id_to_index[robot_id]]
            r.current_path = path
            r.remaining_distance = max(0, len(path) - 1)
    # avg_path_length = np.mean([len(path)-1 for path in solution.values() if path is not None and len(path) > 1])
    valid_lengths = [len(path) - 1 for path in solution.values() if path is not None and len(path) > 1]
    avg_path_length = float(np.mean(valid_lengths)) if valid_lengths else 0.0


    # Initialize planner state variables used later
    previous_active, previous_goals = state_check(robots)
    current_active, current_goals = state_check(robots)
    time_steps_unchanged = 0
    events = {"new_tasks": 0, "new_robots": 0, "completed_tasks": 0}
    idle_steps = {r.robot_id: 0 for r in robots}

    for time_step in range(max_time_steps):

        # before each time step, refresh the unassigned robots and tasks lists
        unassigned_robots = [r.robot_id for r in robots if not r.assigned]
        unassigned_tasks = [t.task_id for t in tasks if not t.assigned]

        # Simulate time step
        completed_this_step, unassigned_count, total_reward, total_success = simulate_time_step(
            robots, tasks, unassigned_robots, unassigned_tasks,
            suitability_method, occupied_positions, start_positions, 
            goal_positions, 1.0, total_reward, total_success
        )

        if len(tasks) == 0:
            print(f"All tasks completed in {time_step + 1} time steps!")
            total_time_steps = time_step + 1
            break
        events["completed_tasks"] += completed_this_step # track number of tasks completed
        should_replan_cbs = completed_this_step > 0

        # Periodically add new tasks and robots
        if add_tasks and time_step + 1 <= 2 and random.random() < 0.5: # add tasks only in the first 2 time steps, and randomly
            print(f"ADDING NEW TASKS AT TIME STEP {time_step + 1}")
            num_of_tasks_added = random.randint(0, tasks_to_add)
            task_max_id, total_tasks = add_new_tasks(
                tasks, unassigned_tasks, task_max_id, 
                num_of_tasks_added, total_tasks, grid, 
                occupied_positions, task_generation_strict
            )
            events["new_tasks"] += num_of_tasks_added # track number of tasks added

        if add_robots and time_step + 1 <= 4 and random.random() < 0.5: # add robots only in the first 4 time steps, and randomly
            print(f"ADDING NEW ROBOTS AT TIME STEP {time_step + 1}")
            num_of_robots_added = random.randint(0, robots_to_add)
            robot_max_id = add_new_robots(
                robots, unassigned_robots, robot_max_id, 
                num_of_robots_added, grid, occupied_positions, 
                robot_generation_strict
            )
            events["new_robots"] += num_of_robots_added # track number of robots added
            for r in robots:
                if r.robot_id not in idle_steps:
                    idle_steps[r.robot_id] = 0

        # Periodically remove robots
        if remove_robots and time_step + 1 <= 4 and random.random() < 0.5: # remove robots only in the first 4 time steps, and randomly
            if len(assigned_robots) > 1: # Otherwise will break CBS, we need at least one agent for things to run smoothly
                print(f"REMOVING RANDOM ROBOTS AT TIME STEP {time_step + 1}")
                remove_random_robots(robots, tasks, unassigned_robots, unassigned_tasks, random.randint(0, robots_to_remove), occupied_positions, start_positions, goal_positions)

        for r in robots:
            if not r.assigned:
                idle_steps[r.robot_id] = idle_steps.get(r.robot_id, 0) + 1 # update idle steps of unassigned robots
            else:
                idle_steps[r.robot_id] = 0
        
        # Update start and goal positions before cbs
        for robot in robots:
            if robot.assigned and robot.current_task:
                start_positions[robot.robot_id] = robot.location
                goal_positions[robot.robot_id] = robot.current_task.location
            else:
                start_positions.pop(robot.robot_id, None)
                goal_positions.pop(robot.robot_id, None)

        # update planning signatures
        current_active, current_goals = state_check(robots)

        # Update assigned robots
        assigned_robots = {r.robot_id: r.current_task.task_id for r in robots if r.assigned and r.current_task}

        # decide to reassign and replan
        should_replan = False
        if unassigned_robots and unassigned_tasks:
            if events["new_tasks"] or events["new_robots"] or events["completed_tasks"]:
                should_replan = True
            elif (current_active != previous_active) or (current_goals != previous_goals):
                should_replan = True

        if should_replan:
            should_replan_cbs = True

        # Reassign unassigned robots to unassigned tasks
        # if should_replan and start_positions and goal_positions:
        if should_replan:

            # Determine what to pass to reassignment functions
            if getattr(sm, "_is_llm_batch", False):
                # Check if all current robots/tasks exist in cached index mappings
                # This allows cache reuse even when tasks are completed (removed)
                # as long as no NEW robots/tasks have been added
                current_robot_ids = {r.robot_id for r in robots}
                current_task_ids = {t.task_id for t in tasks}
                
                cache_valid = (
                    llm_cache["matrix"] is not None and
                    llm_cache["robot_id_to_idx"] is not None and
                    llm_cache["task_id_to_idx"] is not None and
                    current_robot_ids.issubset(llm_cache["robot_id_to_idx"].keys()) and
                    current_task_ids.issubset(llm_cache["task_id_to_idx"].keys())
                )
                
                if not cache_valid:
                    # New robots or tasks added that aren't in cache, need fresh LLM call
                    print(f"LLM cache miss - rebuilding matrix for {len(robots)} robots, {len(tasks)} tasks")
                    suitability_matrix = sm(robots, tasks)
                    llm_cache["matrix"] = suitability_matrix
                    llm_cache["robot_id_to_idx"] = {r.robot_id: i for i, r in enumerate(robots)}
                    llm_cache["task_id_to_idx"] = {t.task_id: j for j, t in enumerate(tasks)}
                
                # Pass matrix tuple for direct array lookup (fast path)
                suitability_source = (llm_cache["matrix"], llm_cache["robot_id_to_idx"], llm_cache["task_id_to_idx"])
            else:
                # Pass scorer function for non-LLM methods
                suitability_source = sm
            

            if voting_method in voting_methods:
                print(f"REASSIGNING WITH VOTING METHOD: {voting_method_name}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = V.reassign_robots_to_tasks(
                    robots, tasks, num_candidates, voting_method, suitability_source,
                    unassigned_robots, unassigned_tasks, start_positions, goal_positions
                )
                # Track all fairness metrics for this reassignment
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))
            elif voting_method in optimization_methods:
                print(f"REASSIGNING WITH OPTIMIZATION METHOD: {optimization_method_name}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = O.reassign_robots_to_tasks_with_method(
                    robots, tasks, num_candidates, voting_method, suitability_source,
                    unassigned_robots, unassigned_tasks, voting_method, start_positions, goal_positions
                )
                # Track all fairness metrics for this reassignment
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))
            total_reassignment_time  += reassign_length
            total_reassignment_score += reassign_score

            # rebuild starts/goals after potential changes from reassignment
            for robot in robots:
                if robot.assigned and robot.current_task:
                    start_positions[robot.robot_id] = tuple(robot.location)
                    goal_positions[robot.robot_id]  = tuple(robot.current_task.location)
                else:
                    start_positions.pop(robot.robot_id, None)
                    goal_positions.pop(robot.robot_id, None)
        
        if should_replan_cbs and start_positions and goal_positions:
            # Build agents and compute A* paths for each independently (fallback to simple assignment)
            agents = build_cbs_agents(robots, start_positions, goal_positions)

            # Compute A* per-agent paths in parallel using ProcessPoolExecutor
            args = []
            for a in agents:
                rid = a['name'].robot_id if hasattr(a['name'], 'robot_id') else a['name']
                start = tuple(a['start'])
                goal = tuple(a['goal'])
                args.append((rid, start, goal, obstacle_array))

            solution = {}
            if args:
                max_workers = min(len(args), (os.cpu_count() or 1))
                ExecutorClass = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor
                with ExecutorClass(max_workers=max_workers) as ex:
                    for rid, path in ex.map(_compute_agent_path, args):
                        if path:
                            solution[rid] = path
                        else:
                            print(f"Warning: no A* path found for agent {rid}")
                            solution[rid] = None

            # Assign computed paths back to robot objects
            id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
            for robot_id, path in solution.items():
                if path is None:
                    continue
                if robot_id in id_to_index:
                    r = robots[id_to_index[robot_id]]
                    r.current_path = path
                    r.remaining_distance = max(0, len(path) - 1)

            previous_active, previous_goals = state_check(robots)  # update to the post-replan state
            events = {k: 0 for k in events}  # reset counters we just consumed

    overall_success_rate = total_success / total_tasks
    avg_reassignment_score = (total_reassignment_score / total_reassignments) if total_reassignments > 0 else 0.0
    avg_reassignment_jains_index = (sum(reassignment_jains_indices) / len(reassignment_jains_indices)) if reassignment_jains_indices else 0.0
    
    # Compute average fairness metrics across reassignments
    if reassignment_fairness_metrics:
        num_metrics = len(FAIRNESS_METRIC_NAMES)
        avg_reassign_metrics = tuple(
            sum(m[i] for m in reassignment_fairness_metrics) / len(reassignment_fairness_metrics)
            for i in range(num_metrics)
        )
    else:
        # Default zeros if no reassignments occurred
        avg_reassign_metrics = tuple(0.0 for _ in FAIRNESS_METRIC_NAMES)
    
    print(f"Voting: Total reward: {total_reward}, Overall success rate: {overall_success_rate:.2%}, Tasks completed: {total_success}, Reassignment Time: {total_reassignment_time}, Reassignment Score: {total_reassignment_score}, \ntotal reassignments: {total_reassignments}, total tasks: {total_tasks}, Total robots: {len(robots)}")
    
    # Return: base metrics + initial_jains_index + avg_reassignment_jains_index + avg reassignment fairness metrics (14 additional)
    return (total_reward, overall_success_rate, total_success, total_reassignment_time, 
            total_reassignment_score, total_reassignments, min(total_time_steps, max_time_steps), 
            avg_reassignment_score, avg_path_length, initial_jains_index, avg_reassignment_jains_index) + avg_reassign_metrics[1:]  # Skip first (Jains) as it's already in avg_reassignment_jains_index


def benchmark_simulation(
        output: tuple[list[tuple[int, int]],list[int],list[int]], 
        robots: List[CapabilityProfile], tasks: List[TaskDescription], 
        num_candidates: int, 
        voting_method: callable, 
        grid: List[List[int]], 
        map_dict: dict, 
        suitability_method: callable, 
        suitability_matrix: np.ndarray, 
        max_time_steps: int, 
        add_tasks: bool, 
        add_robots: bool, 
        remove_robots: bool, 
        tasks_to_add: int = 1, 
        robots_to_add: int = 1, 
        robots_to_remove: int = 1,
        robot_generation_strict: bool = True,
        task_generation_strict: bool = True,
        initial_jains_index: float = 0.0,):
    start_time = time.perf_counter_ns()
    output_tuple = main_simulation(
        output, robots, tasks, num_candidates, voting_method, 
        grid, map_dict, suitability_method, suitability_matrix, 
        max_time_steps, add_tasks, add_robots, remove_robots, 
        tasks_to_add, robots_to_add, robots_to_remove,
        robot_generation_strict, task_generation_strict,
        initial_jains_index=initial_jains_index)
    end_time = time.perf_counter_ns()
    execution_time = end_time - start_time

    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used

    print(f"Simulation completed in {execution_time:.5f} nanoseconds.")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage / (1024 * 1024)} MB")

    return output_tuple + (execution_time, cpu_usage, memory_usage)

if __name__ == "__main__":
        voting_methods = [
            V.rank_assignments_borda, 
            V.rank_assignments_approval, 
            V.rank_assignments_majority_judgment, 
            V.rank_assignments_cumulative_voting, 
            V.rank_assignments_condorcet_method, 
            V.rank_assignments_range
            ]
        voting_names = [func_name(f) for f in voting_methods]
        allocation_methods = [
            O.cbba_task_allocation, 
            O.ssia_task_allocation, 
            O.ilp_task_allocation, 
            O.jv_task_allocation
            ]
        allocation_names = [func_name(f) for f in allocation_methods]
        all_methods = voting_methods + allocation_methods
        suitability_methods = [
            S.evaluate_suitability_balanced, 
            S.evaluate_suitability_loose, 
            S.evaluate_suitability_strict,
            # S.evaluate_suitability_from_names_with_llm
            ]
        small_maps = [
            r"den201d.map", # 37  x 37
            r"den202d.map", # 40  x 39
            r"den404d.map", # 34  x 28
            r"lak101d.map", # 31  x 30
            r"lak102d.map", # 30  x 38
            r"lak105d.map", # 25  x 31
            r"lak107d.map", # 26  x 36
            r"lak108d.map", # 26  x 27
        ]
        medium_maps = [
            r"arena.map",   # 49  x 49
            r"den009d.map", # 34  x 50
            r"den101d.map", # 41  x 73
            r"den204d.map", # 66  x 66
            r"den207d.map", # 50  x 38
            r"den403d.map", # 49  x 74
            r"den405d.map", # 42  x 74
            r"den407d.map", # 57  x 33
            r"den408d.map", # 50  x 34
            r"hrt002d.map", # 50  x 49
            r"isound1.map", # 63  x 55
            r"lak103d.map", # 49  x 49
            r"lak104d.map", # 41  x 41
        ]
        large_maps = [
            r"den001d.map", # 80  x 211
            r"den020d.map", # 118 x 89
            r"den203d.map", # 77  x 93
            r"den206d.map", # 190 x 50
            r"den308d.map", # 88  x 100
            r"den312d.map", # 81  x 65
            r"den900d.map", # 128 x 128
            r"den901d.map", # 128 x 129
            r"den998d.map", # 86  x 62
            r"hrt001d.map", # 112 x 104
            r"lak106d.map", # 113 x 97
            r"lak203d.map", # 146 x 112
            r"lak307d.map", # 84  x 84
            r"ost002d.map", # 145 x 181
        ]
        map_paths = (
        random.sample(small_maps, 1) +
        random.sample(medium_maps, 1) +
        random.sample(large_maps, 1)
        )

        # randomly chosen maps from strict run
        # map_paths = [
        #     r"den020d.map", # 118 x 89
        #     r"den404d.map", # 34  x 28
        #     r"den405d.map", # 42  x 74
        #     r"den408d.map", # 50  x 34
        #     r"isound1.map", # 63  x 55
        #     r"lak102d.map", # 30  x 38
        #     r"lak108d.map", # 26  x 27
        #     r"lak203d.map", # 146 x 112
        #     r"ost002d.map", # 145 x 181
        # ]
        # map_paths = [
        #     r"den020d.map", # 118 x 89
        #     # r"den201d.map", # 37  x 37
        #     # r"arena.map",   # 49  x 49
        #     # r"den204d.map", # 66  x 66
        # ]
        
        # max_time_steps = 500
        robot_sizes = [20]
        task_sizes = [20]
        Run_ID = 1
        num_repetitions = 1
        add_tasks = False
        add_robots = False
        remove_robots = False
        robot_generation_strict = True
        task_generation_strict = True
        map_dir = r"MAPF_benchmark_maps"

        dir_path = os.path.join('hvbta', 'io', 'results')
        os.makedirs(dir_path, exist_ok=True)
        full_paths = [os.path.join(map_dir, m) for m in map_paths]

        # DO SIMULATION FOR ALL MAPS
        for map_file in full_paths:
            grid = load_map(map_file) # 2D list of 0/1 representing the map
            HYPOTENUSE = (len(grid)**2 + len(grid[0])**2) ** 0.5
            dims = (len(grid), len(grid[0])) # dimensions of the map grid
            map_size = "Small" if dims[0] < 40 and dims[1] < 40 else "Medium" if dims[0] < 75 and dims[1] < 75 else "Large"
            obstacles = create_obstacle_list(grid) # list of obstacle coordinates
            map_dict = {
                'dimension': dims,
                'obstacles': obstacles
            }

            results_path  = os.path.join(dir_path, f"Strict_Generation_LLM_simulation_results_{os.path.basename(map_file)}.csv")
            profiles_path = os.path.join(dir_path, f"Strict_Generation_LLM_profiles_{os.path.basename(map_file)}.csv")

            with open(results_path, mode="w", newline='') as file, \
                open(profiles_path, mode="w", newline='') as profile_file:

                writer = csv.writer(file)
                profiles_w = csv.writer(profile_file)

                writer.writerow([
                    'Run ID', 'Method', 'Suitability Method', 'Num Robots', 
                    'Num Tasks', 'Num Candidates', 'Total Score', 
                    'Task Normalized Score', 'Score Density', 'Length',
                    # Initial fairness metrics (15 columns from assignment_infos)
                    'Init_Jains_Index', 'Init_Below_GE_Frac', 'Init_Below_Good_Frac',
                    'Init_Deficit_All_GE', 'Init_Deficit_Below_GE',
                    'Init_Deficit_All_Good', 'Init_Deficit_Below_Good',
                    'Init_Score_Range', 'Init_Min_Max_Ratio',
                    'Init_Gini', 'Init_CV',
                    'Init_Median', 'Init_Mean',
                    'Init_Med_Mean_Gap', 'Init_IQR',
                    # Output from simulation
                    'total_reward', 'overall_success_rate', 'total_success', 
                    'total_reassignment_time', 'total_reassignment_score', 
                    'total_reassignments', 'Total Time Steps', 'Average Reassignment Score',
                    'Average Path Length', 'Initial Jains Index (sim)', 'Avg Reassignment Jains Index',
                    # Additional fairness metrics (14 columns for avg reassignment, excluding Jains which is above)
                    'Avg_Reass_Below_GE_Frac', 'Avg_Reass_Below_Good_Frac',
                    'Avg_Reass_Deficit_All_GE', 'Avg_Reass_Deficit_Below_GE',
                    'Avg_Reass_Deficit_All_Good', 'Avg_Reass_Deficit_Below_Good',
                    'Avg_Reass_Score_Range', 'Avg_Reass_Min_Max_Ratio',
                    'Avg_Reass_Gini', 'Avg_Reass_CV',
                    'Avg_Reass_Median', 'Avg_Reass_Mean',
                    'Avg_Reass_Med_Mean_Gap', 'Avg_Reass_IQR',
                    'Execution Time', 'CPU Usage', 'Memory Usage', 'Map Size'])
                
                profiles_w.writerow([
                'Run_ID', 'Map',
                'Num Robots', 'Num Tasks',
                'Suitability Method',
                'RobotProfiles', 'TaskProfiles'])

                for num_robots in robot_sizes:
                    print(f"\n\n\nSTARTING SIMULATION FOR {num_robots} ROBOTS")

                    for num_tasks in task_sizes:
                        print(f"\n\n\nSTARTING SIMULATION FOR {num_tasks} TASKS")
                        candidate_sizes = [
                            # max(1, max(int(num_robots * 0.75), int(num_tasks * 0.75))),
                            # max(1, max(int(num_robots * 1.0), int(num_tasks * 1.0))),
                            # max(1, min(max(int(num_robots), int(num_tasks)), 75)),
                            # max(1, max(int(num_robots), int(num_tasks))),
                            # 25,
                            50,
                        ]
                        WORKLOAD = max(1.0, num_tasks / num_robots)
                        if WORKLOAD > 10:
                            extender = 1.5
                        elif WORKLOAD > 5:
                            extender = 1.2
                        else:
                            extender = 1.0
                        max_time_steps = max(200, int(HYPOTENUSE * 1.5 * extender)) # allow enough time steps for longest path plus some buffer
                        for nc in candidate_sizes:
                            # print(f"\n\n\nSTARTING SIMULATION FOR {nc} CANDIDATES")

                            for sm in suitability_methods:
                                sm_name = func_name(sm)
                                # print(f"\n\n\nSTARTING SIMULATION FOR SUITABILITY METHOD: {sm_name}")

                                for rep in range(num_repetitions):
                                    print(f"\n\n\nSTARTING SIMULATION REPETITION {rep+1}/{num_repetitions}")
                                    voting_outputs = []
                                    assignment_infos = []

                                    if robot_generation_strict:
                                        robots = [G.generate_random_robot_profile_strict(f"R{idx+1}", grid, set()) for idx in range(num_robots)]
                                        robot_profiles = [r.strict_profile_name for r in robots]
                                    else:
                                        robots = [G.generate_random_robot_profile(f"R{idx+1}", grid, set()) for idx in range(num_robots)]
                                    if task_generation_strict:
                                        tasks = [G.generate_random_task_description_strict(f"T{idx+1}", grid, set(), []) for idx in range(num_tasks)]
                                        task_profiles = [t.strict_profile_name for t in tasks]
                                    else:
                                        tasks = [G.generate_random_task_description(f"T{idx+1}", grid, set(), []) for idx in range(num_tasks)]

                                    pairwise_scorer = sm
                                    if getattr(sm, "_is_llm_batch", False):
                                        suitability_matrix = sm(robots, tasks)
                                        pairwise_scorer = S.make_pairwise_from_batch(lambda *_: suitability_matrix, robots, tasks)
                                    else:
                                        pairwise_scorer = sm
                                        suitability_matrix = S.calculate_suitability_matrix(robots, tasks, pairwise_scorer)

                                    # if both generations are strict we log them to be able to compare later, if one is not theres no point
                                    if robot_generation_strict and task_generation_strict:
                                        profiles_w.writerow([
                                        Run_ID, os.path.basename(map_file),
                                        num_robots, num_tasks,
                                        sm_name,
                                        json.dumps(robot_profiles),
                                        json.dumps(task_profiles)])


                                    if suitability_all_zero(suitability_matrix):
                                        random_assignment = True
                                        # Voting - random fallback
                                        for method_fn, method_name in zip(voting_methods, voting_names):
                                            # print("All suitability scores are zero, randomly assigning tasks to robots.")
                                            output, score, length, per_agent_scores = V.assign_tasks_randomly(robots, tasks, suitability_matrix, nc)
                                            initial_metrics = compute_all_fairness_metrics(per_agent_scores)
                                            initial_jains = initial_metrics[0]
                                            assigned_count = len(output[0]) if output and output[0] else 0
                                            task_normalized_score = (score / assigned_count) if assigned_count > 0 else 0.0
                                            score_density = (score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and score > 0 else 0.0
                                            assignment_infos.append([Run_ID, method_name, sm_name, num_robots, num_tasks, nc, score, task_normalized_score, score_density, length] + list(initial_metrics))
                                            voting_outputs.append((output, initial_jains))
                                        
                                        # Optimization - random fallback
                                        cbba_output, cbba_score, cbba_length, cbba_per_agent_scores = O.assign_tasks_with_method_randomly(O.cbba_task_allocation, suitability_matrix, nc)
                                        cbba_initial_metrics = compute_all_fairness_metrics(cbba_per_agent_scores)
                                        cbba_initial_jains = cbba_initial_metrics[0]
                                        assigned_count = len(cbba_output[0]) if cbba_output and cbba_output[0] else 0 # nuber of pairs in the chosen assignment
                                        task_normalized_score = (cbba_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (cbba_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and cbba_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "cbba_task_allocation", sm_name, num_robots, num_tasks, nc, cbba_score, task_normalized_score, score_density, cbba_length] + list(cbba_initial_metrics))

                                        ssia_output, ssia_score, ssia_length, ssia_per_agent_scores = O.assign_tasks_with_method_randomly(O.ssia_task_allocation, suitability_matrix, nc)
                                        ssia_initial_metrics = compute_all_fairness_metrics(ssia_per_agent_scores)
                                        ssia_initial_jains = ssia_initial_metrics[0]
                                        assigned_count = len(ssia_output[0]) if ssia_output and ssia_output[0] else 0
                                        task_normalized_score = (ssia_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (ssia_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and ssia_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "ssia_task_allocation", sm_name, num_robots, num_tasks, nc, ssia_score, task_normalized_score, score_density, ssia_length] + list(ssia_initial_metrics))

                                        ilp_output, ilp_score, ilp_length, ilp_per_agent_scores = O.assign_tasks_with_method_randomly(O.ilp_task_allocation, suitability_matrix, nc)
                                        ilp_initial_metrics = compute_all_fairness_metrics(ilp_per_agent_scores)
                                        ilp_initial_jains = ilp_initial_metrics[0]
                                        assigned_count = len(ilp_output[0]) if ilp_output and ilp_output[0] else 0
                                        task_normalized_score = (ilp_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (ilp_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and ilp_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "ilp_task_allocation", sm_name, num_robots, num_tasks, nc, ilp_score, task_normalized_score, score_density, ilp_length] + list(ilp_initial_metrics))

                                        jv_output, jv_score, jv_length, jv_per_agent_scores = O.assign_tasks_with_method_randomly(O.jv_task_allocation, suitability_matrix, nc)
                                        jv_initial_metrics = compute_all_fairness_metrics(jv_per_agent_scores)
                                        jv_initial_jains = jv_initial_metrics[0]
                                        assigned_count = len(jv_output[0]) if jv_output and jv_output[0] else 0
                                        task_normalized_score = (jv_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (jv_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and jv_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "jv_task_allocation", sm_name, num_robots, num_tasks, nc, jv_score, task_normalized_score, score_density, jv_length] + list(jv_initial_metrics))

                                        outputs = voting_outputs + [(cbba_output, cbba_initial_jains), (ssia_output, ssia_initial_jains), (ilp_output, ilp_initial_jains), (jv_output, jv_initial_jains)]

                                    else:
                                        # Voting - normal
                                        for method_fn, method_name in zip(voting_methods, voting_names):
                                            output, score, length, per_agent_scores = V.assign_tasks_with_voting(robots, tasks, suitability_matrix, nc, method_fn)
                                            initial_metrics = compute_all_fairness_metrics(per_agent_scores)
                                            initial_jains = initial_metrics[0]
                                            assigned_count = len(output[0]) if output and output[0] else 0 # nuber of pairs in the chosen assignment
                                            task_normalized_score = (score / assigned_count) if assigned_count > 0 else 0.0 # per assigned task normalized score
                                            score_density = (score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and score > 0 else 0.0
                                            assignment_infos.append([Run_ID, method_name, sm_name, num_robots, num_tasks, nc, score, task_normalized_score, score_density, length] + list(initial_metrics))
                                            voting_outputs.append((output, initial_jains))

                                        # Optimization - normal
                                        cbba_output, cbba_score, cbba_length, cbba_per_agent_scores = O.assign_tasks_with_method(O.cbba_task_allocation,suitability_matrix)
                                        cbba_initial_metrics = compute_all_fairness_metrics(cbba_per_agent_scores)
                                        cbba_initial_jains = cbba_initial_metrics[0]
                                        assigned_count = len(cbba_output[0]) if cbba_output and cbba_output[0] else 0 # nuber of pairs in the chosen assignment
                                        task_normalized_score = (cbba_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (cbba_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and cbba_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "cbba_task_allocation", sm_name, num_robots, num_tasks, nc, cbba_score, task_normalized_score, score_density, cbba_length] + list(cbba_initial_metrics))

                                        ssia_output, ssia_score, ssia_length, ssia_per_agent_scores = O.assign_tasks_with_method(O.ssia_task_allocation,suitability_matrix)
                                        ssia_initial_metrics = compute_all_fairness_metrics(ssia_per_agent_scores)
                                        ssia_initial_jains = ssia_initial_metrics[0]
                                        assigned_count = len(ssia_output[0]) if ssia_output and ssia_output[0] else 0
                                        task_normalized_score = (ssia_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (ssia_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and ssia_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "ssia_task_allocation", sm_name, num_robots, num_tasks, nc, ssia_score, task_normalized_score, score_density, ssia_length] + list(ssia_initial_metrics))

                                        ilp_output, ilp_score, ilp_length, ilp_per_agent_scores = O.assign_tasks_with_method(O.ilp_task_allocation,suitability_matrix)
                                        ilp_initial_metrics = compute_all_fairness_metrics(ilp_per_agent_scores)
                                        ilp_initial_jains = ilp_initial_metrics[0]
                                        assigned_count = len(ilp_output[0]) if ilp_output and ilp_output[0] else 0
                                        task_normalized_score = (ilp_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (ilp_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and ilp_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "ilp_task_allocation", sm_name, num_robots, num_tasks, nc, ilp_score, task_normalized_score, score_density, ilp_length] + list(ilp_initial_metrics))

                                        jv_output, jv_score, jv_length, jv_per_agent_scores = O.assign_tasks_with_method(O.jv_task_allocation,suitability_matrix)
                                        jv_initial_metrics = compute_all_fairness_metrics(jv_per_agent_scores)
                                        jv_initial_jains = jv_initial_metrics[0]
                                        assigned_count = len(jv_output[0]) if jv_output and jv_output[0] else 0
                                        task_normalized_score = (jv_score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (jv_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 and jv_score > 0 else 0.0
                                        assignment_infos.append([Run_ID, "jv_task_allocation", sm_name, num_robots, num_tasks, nc, jv_score, task_normalized_score, score_density, jv_length] + list(jv_initial_metrics))

                                        outputs = voting_outputs + [(cbba_output, cbba_initial_jains), (ssia_output, ssia_initial_jains), (ilp_output, ilp_initial_jains), (jv_output, jv_initial_jains)]

                                    for idx, (out_tuple, meth) in enumerate(zip(outputs, all_methods)):
                                        # out_tuple is (output, initial_jains_index) tuple
                                        out, initial_jains = out_tuple
                                        # print(f"\n\n\nRUNNING SIMULATION FOR METHOD: {meth}")
                                        output_tuple = benchmark_simulation(
                                            out, copy.deepcopy(robots), copy.deepcopy(tasks), 
                                            nc, meth, grid, map_dict, sm, suitability_matrix, 
                                            max_time_steps, add_tasks, add_robots, remove_robots, 
                                            10, 10, 10, robot_generation_strict, task_generation_strict,
                                            initial_jains_index=initial_jains)
                                        # write a single combined row: assignment info + benchmark metrics
                                        row_prefix = assignment_infos[idx] if idx < len(assignment_infos) else [Run_ID, func_name(meth), sm_name, num_robots, num_tasks, nc, 0, 0, 0, 0]
                                        writer.writerow(row_prefix + list(output_tuple) + [map_size])
                                    Run_ID += 1