from scipy.optimize import linear_sum_assignment
import numpy as np
from typing import List, Tuple, Callable
import pulp
import time
from hvbta.models import CapabilityProfile, TaskDescription
from .assignments import generate_random_assignments
from hvbta.suitability import calculate_suitability_matrix, calculate_total_suitability
from .misc_assignment import extract_submatrix

def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)

def jv_task_allocation(matrix):
    matrix = np.array(matrix)
    max_val = np.max(matrix)
    
    cost_matrix = max_val - matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignment = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if matrix[r, c] > 0]
    
    return assignment

def cbba_task_allocation(suitability_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Uses a two-phase Consensus-Based Bundle Algorithm (CBBA) for task allocation.
    Each robot is assigned to one task, and each task is assigned to one robot.
    
    Parameters:
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability score of robot i for task j.
    
    Returns:
        final_assignment: A list of (robot, task) pairs representing the final allocation.
    """
    num_robots = len(suitability_matrix)
    num_tasks = len(suitability_matrix[0])

    # Initialize assignment variables
    robot_bundles = [[] for _ in range(num_robots)]  # Bundle of tasks for each robot
    task_bids = [-1] * num_tasks  # Highest bid for each task

    # Phase 1: Bundle Construction
    for robot in range(num_robots):
        # Each robot evaluates each task for inclusion in its bundle
        for task in range(num_tasks):
            bid = suitability_matrix[robot][task]
            if bid > task_bids[task]:
                task_bids[task] = bid
            robot_bundles[robot].append(task)
    
    not_assigned = list(range(num_robots))
    # Phase 2: Conflict Resolution
    updated = True
    while updated:
        updated = False
        
        for robot in range(num_robots):
            if not robot_bundles[robot] or robot not in not_assigned:
                continue  # Skip robots with no tasks in their bundle
                
            # Sort the robot's bundle in descending order of suitability scores
            robot_bundles[robot].sort(key=lambda task: suitability_matrix[robot][task], reverse=True)
            # Iterate through tasks in the robot's bundle
            for task in robot_bundles[robot][:]:
                # If the robot has the highest unique bid for this task, assign it
                highest_bid = max(suitability_matrix[competing_robot][task] for competing_robot in not_assigned)
                if suitability_matrix[robot][task] == highest_bid:
                    updated = True
                    task_bids[task] = suitability_matrix[robot][task]
                    # Clear remaining tasks from this robot's bundle
                    robot_bundles[robot] = [task]
                    not_assigned.remove(robot)
                    for r in range(num_robots):
                        if r != robot and task in robot_bundles[r]:
                            robot_bundles[r].remove(task)
                    break

        if len(not_assigned) == 0:
            updated = False
                    
    final_assignment = [(index, bundle[0]) for index, bundle in enumerate(robot_bundles) if bundle and suitability_matrix[index][bundle[0]] != 0]

    return final_assignment

def ssia_task_allocation(suitability_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Uses the Sequential Single-Item Auction (SSIA) for task allocation.
    
    Parameters:
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability score of robot i for task j.
    
    Returns:
        assignments: A list of (robot, task) pairs representing the final allocation.
    """
    num_robots = len(suitability_matrix)
    num_tasks = len(suitability_matrix[0])
    #**print(f"Suitability Matrix Issue: {suitability_matrix}")
    #**print(f"Number of Robots: {num_robots}, Number of Tasks: {num_tasks}")

    
    # Initialize assignment list to store (robot, task) pairs
    assignments = []
    assigned_robots = set()  # Track robots that have already been assigned
    
    # Auction tasks sequentially
    for task in range(num_tasks):
        # NOTE: I can fix it by making the highest_bid negative infinity but I want to see why the suitability matrix is all negative
        highest_bid = -1
        winning_robot = -1

        # Robots bid for the current task
        for robot in range(num_robots):
            if robot in assigned_robots:
                continue  # Skip if the robot is already assigned
            bid = suitability_matrix[robot][task]
            if bid > highest_bid:
                # check if the bid is positive because we are getting an empty assigned robots dictionary
                #**print(f"Robot {robot} bids {bid} for Task {task}")
                highest_bid = bid
                winning_robot = robot
        
        # Assign the task to the robot with the highest bid if any bid is positive
        # if highest_bid >= 0:
        if winning_robot != -1 and highest_bid > 0:
            assignments.append((winning_robot, task))
            assigned_robots.add(winning_robot)  # Mark this robot as assigned
    
    return assignments

def ilp_task_allocation(suitability_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Uses Integer Linear Programming (ILP) to maximize suitability-based task allocation.
    
    Parameters:
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability score of robot i for task j.
    
    Returns:
        assignment: A list of (robot, task) pairs representing the final assignment.
    """
    num_robots = len(suitability_matrix)
    num_tasks = len(suitability_matrix[0])

    # Define the ILP problem
    problem = pulp.LpProblem("TaskAssignment", pulp.LpMaximize)

    # Define binary decision variables x_ij where x_ij = 1 if robot i is assigned to task j, else 0
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(num_tasks)] for i in range(num_robots)]

    # Objective: Maximize total suitability score
    problem += pulp.lpSum(suitability_matrix[i][j] * x[i][j] for i in range(num_robots) for j in range(num_tasks))

    # Constraint: Each task can be assigned to at most one robot
    for j in range(num_tasks):
        problem += pulp.lpSum(x[i][j] for i in range(num_robots)) <= 1, f"Task_{j}_Assignment"

    # Constraint: Each robot can be assigned to at most one task
    for i in range(num_robots):
        problem += pulp.lpSum(x[i][j] for j in range(num_tasks)) <= 1, f"Robot_{i}_Capacity"

    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Collect the results
    assignment = []
    for i in range(num_robots):
        for j in range(num_tasks):
            if pulp.value(x[i][j]) == 1:
                assignment.append((i, j))

    return assignment

def assign_tasks_with_method(
    allocation_method: Callable[[List[List[float]]], List[Tuple[int, int]]],
    suitability_matrix: List[List[float]]
) -> Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float]:
    """
    Assigns tasks using a specified allocation method and returns the allocation details.
    
    Parameters:
        allocation_method: The function used for task allocation (e.g., `cbba_task_allocation`, `ssia_task_allocation`, `ilp_task_allocation`).
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability score of robot i for task j.
    
    Returns:
        (assignment, unassigned_robots, unassigned_tasks, total_score, allocation_time): A tuple containing:
      1. A list of assigned (robot, task) pairs.
      2. A list of unassigned robot indices.
      3. A list of unassigned task indices.
      4. The total suitability score of the assignment.
      5. The time taken for the allocation (in microseconds).
    """
    num_robots = len(suitability_matrix)
    num_tasks = len(suitability_matrix[0])
    
    # Start timing
    start_time = time.perf_counter_ns()
    
    # Get the assignment using the specified allocation method
    assignment = allocation_method(suitability_matrix)
    
    # End timing
    end_time = time.perf_counter_ns()
    allocation_time = (end_time - start_time) / 1000.0  # Convert nanoseconds to microseconds
    
    # Calculate total suitability score for the assignment
    total_score = calculate_total_suitability(assignment, suitability_matrix)
    
    # Determine unassigned robots and tasks
    assigned_robots = {robot for robot, _ in assignment}
    assigned_tasks = {task for _, task in assignment}
    
    unassigned_robots = [robot for robot in range(num_robots) if robot not in assigned_robots]
    unassigned_tasks = [task for task in range(num_tasks) if task not in assigned_tasks]

    #**print(f"Total Suitability Score: {total_score:.2f}")
    #**print(f"Time taken for allocation: {allocation_time:.2f} microseconds")
    
    # Extract per-agent scores for fairness calculation (only assigned agents)
    per_agent_scores = [float(suitability_matrix[r][t]) for r, t in assignment]
    
    return (assignment, unassigned_robots, unassigned_tasks), total_score, allocation_time, per_agent_scores

def assign_tasks_with_method_randomly(
    allocation_method: Callable[[List[List[float]]], List[Tuple[int, int]]],
    suitability_matrix: List[List[float]],
    num_candidates: int,
) -> Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float]:
    """
    Assigns tasks randomly and returns the allocation details.
    
    Parameters:
        allocation_method: The function used for task allocation (e.g., `cbba_task_allocation`, `ssia_task_allocation`, `ilp_task_allocation`).
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability score of robot i for task j.
    
    Returns:
        (assignment, unassigned_robots, unassigned_tasks, total_score, allocation_time): A tuple containing:
      1. A list of assigned (robot, task) pairs.
      2. A list of unassigned robot indices.
      3. A list of unassigned task indices.
      4. The total suitability score of the assignment.
      5. The time taken for the allocation (in microseconds).
    """
    num_robots = len(suitability_matrix)
    num_tasks = len(suitability_matrix[0])

    random_assignments = generate_random_assignments(num_robots, num_tasks, num_candidates)
    
    # Start timing
    start_time = time.perf_counter_ns()
    
    k = np.random.randint(0, num_candidates)
    final_pairs, final_unr_idx, final_unt_idx = random_assignments[k]
    # final_pairs, final_unr_idx, final_unt_idx = [], [], []
    # for i in range(num_candidates):
    #     pairs, unr_idx, unt_idx = random_assignments[i]
    #     if len(pairs) > len(final_pairs):
    #         final_pairs, final_unr_idx, final_unt_idx = pairs, unr_idx, unt_idx
    
    # End timing
    end_time = time.perf_counter_ns()
    allocation_time = (end_time - start_time) / 1000.0  # Convert nanoseconds to microseconds
    
    total_score = 0.0
    
    assigned_pairs = []
    unassigned_robots = []
    unassigned_tasks = []

    print(f"\n\n\n\nPAIRS: {final_pairs} \n\n\n\n UNR: {final_unr_idx} \n\n\n\n UNT: {final_unt_idx}\n\n\n\n")
    for robot_id, task_id in final_pairs:
        assigned_pairs.append((robot_id, task_id))

    if final_unr_idx is None or final_unt_idx is None:
        assigned_r = {r for r, _ in assigned_pairs}
        assigned_t = {t for _, t in assigned_pairs}
        unassigned_robots = [i for i in range(num_robots) if i not in assigned_r]
        unassigned_tasks = [j for j in range(num_tasks) if j not in assigned_t]
    else:
        unassigned_robots = list(final_unr_idx)
        unassigned_tasks = list(final_unt_idx)

    filtered_best_assignments = (assigned_pairs, unassigned_robots, unassigned_tasks)

    print(f"Best assignment in optimization: {filtered_best_assignments}")
    
    # Per-agent scores are all 0.0 for random assignment (all-zero suitability case)
    per_agent_scores = [0.0] * len(assigned_pairs)
    
    return filtered_best_assignments, total_score, allocation_time, per_agent_scores

def reassign_robots_to_tasks_with_method(
        robots: List[CapabilityProfile], 
        tasks: List[TaskDescription], 
        num_candidates: int, 
        voting_method: str, 
        suitability_source,  # Can be Callable (scorer) OR tuple (matrix, r_idx, t_idx) for LLM
        unassigned_robots: List[str], 
        unassigned_tasks: List[str], 
        allocation_method: Callable[[List[List[float]]], List[Tuple[int, int]]], 
        start_positions: dict, 
        goal_positions: dict,
        map_size: int,
        inertia_threshold: float = 0.1) -> Tuple[dict, List[str], List[str], float, float]:
    """
    Reassigns unassigned robots to unassigned tasks using a specified allocation method.
    Parameters:
        robots: List of all robot profiles.
        tasks: List of all task descriptions.
        num_candidates: Number of candidate assignments to generate.
        voting_method: The name of the voting function to use for ranking assignments.
        suitability_source: Either a callable scorer(robot, task)->float, OR a tuple of
                           (full_matrix, robot_id_to_idx, task_id_to_idx) for direct matrix lookup.
        unassigned_robots: List of unassigned robot IDs.
        unassigned_tasks: List of unassigned task IDs.
        allocation_method: The function used for task allocation (e.g., `cbba_task_allocation`, `ssia_task_allocation`, `ilp_task_allocation`).
        start_positions: Dictionary mapping robot IDs to their start positions.
        goal_positions: Dictionary mapping robot IDs to their goal positions.
        Inertia threshold: minimum improvement in suitability required to steal an already‐assigned task.
        
    Returns:
        return_assignments: Dictionary mapping robot IDs to assigned task IDs.
        unassigned_robots: List of unassigned robot IDs after reassignment.
        unassigned_tasks: List of unassigned task IDs after reassignment.
        score: Total suitability score of the best assignment.
        length: Time taken for the voting process in microseconds.
    """
    urobots = [robot for robot in robots if not robot.assigned]
    utasks  = [task for task in tasks if not task.assigned]

    # Early outs
    if not urobots or not utasks:
        return {}, unassigned_robots, unassigned_tasks, 0.0, 0.0, []

    # Check if suitability_source is a matrix tuple or a callable scorer
    if isinstance(suitability_source, tuple) and len(suitability_source) == 3:
        # Direct matrix lookup path (fast - for LLM)
        full_matrix, robot_id_to_idx, task_id_to_idx = suitability_source
        suitability_matrix = extract_submatrix(full_matrix, urobots, utasks, robot_id_to_idx, task_id_to_idx)
        use_matrix_lookup = True
    else:
        # Callable scorer path (for non-LLM methods)
        suitability_matrix = suitability_source(urobots, utasks, map_size=map_size)
        suitability_matrix = np.clip(suitability_matrix, 0.0, 1.0)
        use_matrix_lookup = False

    if suitability_all_zero(suitability_matrix):
        print("All suitability scores are zero, randomly assigning tasks to robots.")
        output, score, length, per_agent_scores = reassign_robots_to_tasks_randomly_with_method(robots, tasks, num_candidates, unassigned_robots, unassigned_tasks)
    else:
        output, score, length, per_agent_scores = assign_tasks_with_method(allocation_method, suitability_matrix)

    assigned_pairs = output[0]
    return_assignments = {}
    unassigned_robots = [urobots[i].robot_id for i in output[1]]
    unassigned_tasks = [utasks[j].task_id for j in output[2]]

    for (robot_idx, task_idx) in assigned_pairs:
        # print(f"UROBOT: {urobots[robot_idx].robot_id} | UTASK: {utasks[task_idx].task_id}")
        pair_score = suitability_matrix[robot_idx][task_idx]
        r = urobots[robot_idx]
        t = utasks[task_idx]
        r.current_task = t
        r.tasks_attempted += 1
        t.assigned_robot = r
        r.assigned = True
        t.assigned = True
        r.current_task_suitability = pair_score
        t.current_suitability = pair_score
        # update start and goal positions when robots are assigned
        start_positions[r.robot_id] = r.location
        goal_positions[r.robot_id] = t.location
        return_assignments[r.robot_id] = t.task_id
    
    # Check for stealing tasks from already assigned robots
    free_robots = [r for r in robots if not r.assigned]
    if free_robots:
        for task in tasks:
            current = task.assigned_robot
            if current is None:
                continue

            # Use direct matrix lookup if available (LLM path), otherwise use scorer
            if use_matrix_lookup:
                current_suitability = full_matrix[robot_id_to_idx[current.robot_id], task_id_to_idx[task.task_id]]
            else:
                current_suitability = float(suitability_source([current], [task], map_size=map_size)[0, 0])
            # find the best free robot for this task
            best, best_suit = None, current_suitability
            for r in free_robots:
                if use_matrix_lookup:
                    s = full_matrix[robot_id_to_idx[r.robot_id], task_id_to_idx[task.task_id]]
                else:
                    s = float(suitability_source([r], [task], map_size=map_size)[0, 0])
                if s > best_suit:
                    best, best_suit = r, s

            # Inertia check: if the best free robot's suitability is not significantly better, skip stealing
            if best and (best_suit - current_suitability) >= inertia_threshold:
                # unassign the current robot from the task
                current.current_task = None
                current.assigned = False
                current.current_task_suitability = None
                if current.robot_id not in unassigned_robots:
                    unassigned_robots.append(current.robot_id)

                # update the task's assigned robot
                best.current_task = task
                best.assigned = True
                best.tasks_attempted += 1
                best.current_task_suitability = best_suit
                task.assigned_robot = best
                task.current_suitability = best_suit

                start_positions[best.robot_id] = best.location
                goal_positions[best.robot_id] = task.location

                # remove from free list and unassigned robots
                free_robots.remove(best)
                if best.robot_id in unassigned_robots:
                    unassigned_robots.remove(best.robot_id)

    print(f"Reassign Score: {score}, Reassign Length: {length}")

    # Recalculate per_agent_scores after potential stealing - get scores for all assigned robots
    final_per_agent_scores = []
    for r in robots:
        if r.assigned and r.current_task is not None:
            final_per_agent_scores.append(float(r.current_task_suitability))

    return return_assignments, unassigned_robots, unassigned_tasks, score, length, final_per_agent_scores

def reassign_robots_to_tasks_randomly_with_method(
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        num_candidates: int,
        unassigned_robots: List[str],
        unassigned_tasks: List[str],

) ->Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float]:
    """
    Reassigns unassigned robots to unassigned tasks using random assignments.
    for use with the all-zero suitability matrix case.
    
    Parameters:
        robots: List of all robot profiles.
        tasks: List of all task descriptions.
        num_candidates: Number of candidate assignments to generate.
        unassigned_robots: List of unassigned robot IDs.
        unassigned_tasks: List of unassigned task IDs.
    
    Returns:
        return_assignments: Dictionary mapping robot IDs to assigned task IDs.
        unassigned_robots: List of unassigned robot IDs after reassignment.
        unassigned_tasks: List of unassigned task IDs after reassignment.
        score: Total suitability score of the best assignment (always 0.0).
        length: Time taken for the random assignment process in microseconds.
    """
    urobots = [robot for robot in robots if not robot.assigned]
    utasks = [task for task in tasks if not task.assigned]
    num_robots = len(urobots)
    num_tasks = len(utasks)

    random_assignments = generate_random_assignments(num_robots, num_tasks, num_candidates)
    
    start = time.perf_counter_ns()
    # total_scores, assignment_ranking = voting_method(random_assignments, suitability_matrix)
    # assignment ranking is a list of integers usually from 0 to num_candidates-1 that ranks the assignments
    # but here we just pick a random assignment since they are all equally bad
    k = np.random.randint(0, num_candidates)
    final_pairs, final_unr_idx, final_unt_idx = random_assignments[k]
    # for i in range(num_candidates):
    #     pairs, unr_idx, unt_idx = random_assignments[i]
    #     if len(pairs) > len(final_pairs):
    #         final_pairs, final_unr_idx, final_unt_idx = pairs, unr_idx, unt_idx
    #         break
    
    end = time.perf_counter_ns()
    length = (end - start) / 1000.0
    total_scores = 0.0
 
    assigned_pairs = []
    unassigned_robots = []
    unassigned_tasks = []

    print(f"\n\n\n\nPAIRS: {final_pairs} \n\n\n\n UNR: {final_unr_idx} \n\n\n\n UNT: {final_unt_idx}\n\n\n\n")

    for robot_id, task_id in final_pairs:
        assigned_pairs.append((robot_id, task_id))

    if final_unr_idx is None or final_unt_idx is None:
        assigned_r = {r for r, _ in assigned_pairs}
        assigned_t = {t for _, t in assigned_pairs}
        unassigned_robots = [i for i in range(num_robots) if i not in assigned_r]
        unassigned_tasks = [j for j in range(num_tasks) if j not in assigned_t]
    else:
        unassigned_robots = list(final_unr_idx)
        unassigned_tasks = list(final_unt_idx)

    filtered_best_assignments = (assigned_pairs, unassigned_robots, unassigned_tasks)

    # print(f"Best assignment in voting {filtered_best_assignments}")

    # best_score = calculate_total_suitability(filtered_best_assignments[0], suitability_matrix)

    # Per-agent scores are all 0.0 for random assignment (all-zero suitability case)
    per_agent_scores = [0.0] * len(assigned_pairs)

    return filtered_best_assignments, total_scores, length, per_agent_scores
