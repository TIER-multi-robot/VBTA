import psutil
import time
import random
import numpy as np
import csv
import os
from typing import List
import copy
from hvbta.pathfinding.Final_CBS import CBS, Environment
from hvbta.simulation.timestep import simulate_time_step
import hvbta.allocators.voting as V
import hvbta.suitability as S
from hvbta.pathfinding.CBS import load_map, create_obstacle_list, build_cbs_agents
from hvbta.allocators.misc_assignment import add_new_tasks, add_new_robots, remove_random_robots
from hvbta.generation import generate_random_robot_profile_strict, generate_random_task_description_strict
from hvbta.models import CapabilityProfile, TaskDescription
import hvbta.allocators.optimizers as O

def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)

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
        task_generation_strict: bool = True):
    print(f"SUITABILITY METHOD: {suitability_method}")
    # print(f"DEBUG STATEMENT 1")

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

    # print(f"DEBUG STATEMENT 2")

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

    print(f"ROBOTS: {[rob.robot_id for rob in robots]}")
    print(f"TASKS: {[tas.task_id for tas in tasks]}")
    print(f"ASSIGNED PAIRS: {assigned_pairs}")
    print(f"ASSIGNED ROBOTS: {assigned_robots}")
    print(f"UNASSIGNED ROBOTS: {unassigned_robots}")
    print(f"UNASSIGNED TASKS: {unassigned_tasks}")

    # print(f"DEBUG STATEMENT 3")

    agents = build_cbs_agents(robots, start_positions, goal_positions)

    print(f"AGENTS LIST: {agents}")

    # Create the input data dictionary for CBS, this will be passed to the CBS planner
    input_data = {
        'map' : {
            'dimension': map_dict['dimension'],
            'obstacles': map_dict['obstacles']
        },
        'agents': agents,
    }

    # print(f"DEBUG STATEMENT 4")

    env = Environment(
        dimension=map_dict['dimension'],
        agents=input_data['agents'],
        obstacles=map_dict['obstacles'],
    )

    print(f"DEBUG STATEMENT BEFORE TIMESTEPS - ENVIRONMENT CREATED - AGENTS BUILT: {agents}")
    planner = CBS(env)
    res = planner.search()

    # print(f"SOLUTION: {solution}")

    if not res:
        print("CBS failed to find a plan under current constraints.")
        # could possibly fall back on the simple method here if we get a lot of issues
    else:
        solution, nodes_expanded, conflicts = res
        print(f"SOLUTION: {solution}")

        print(f"DEBUG STATEMENT - BEFORE TIMESTESP CBS COMPLETE - NODES EXPANDED: {nodes_expanded}, CONFLICTS: {conflicts}")

        # Iterate through the agents and their schedules
        id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
        for robot_id, schedule in solution.items():
            ridx = id_to_index[robot_id]
            robots[ridx].current_path = [(p['x'], p['y']) for p in schedule]
            robots[ridx].remaining_distance = max(0, len(schedule) - 1)
            print(f"Robot {robot_id} path: {robots[ridx].current_path}")

    # Keep track of previous state to not run CBS when there are no changes
    previous_active, previous_goals = state_check(robots)
    current_active, current_goals = state_check(robots)
 
    # End the simulation if nothing changes for 3 timesteps and CBS stalling (hopefully the tasks are done and the robots arent moving)
    time_steps_unchanged = 0

    events = {
        "new_tasks": 0,
        "new_robots": 0,
        "completed_tasks": 0,
    }
    idle_steps = {r.robot_id: 0 for r in robots} # track idleness of free robots

    for time_step in range(max_time_steps):
        # print(f"DEBUG STATEMENT 7 - TIME STEP {time_step+1}")
        # print(f"\n--- Time Step {time_step + 1} ---")
        # print(f"OCCUPIED POSITIONS: {occupied_positions}")
        print(f"AMOUNT OF ASSIGNED ROBOTS: {len([rob.current_task for rob in robots])}")
        print(f"ASSIGNED ROBOTS: {[rob.assigned for rob in robots].count(True)}")
        print(f"START POSITIONS: {start_positions}")
        print(f"GOAL POSITIONS: {goal_positions}")
        print(f"UNASSIGNED ROBOTS: {unassigned_robots}")
        print(f"UNASSIGNED TASKS: {unassigned_tasks}")
        # print(f"ALL ROBOTS: {len(robots)}")
        # print(f"ALL TASKS: {len(tasks)}")
        print(f"LIST OF ALL ROBOTS: {[rob.robot_id for rob in robots]}")
        print(f"LIST OF ALL TASKS: {[tas.task_id for tas in tasks]}")

        # before each time step, refresh the unassigned robots and tasks lists
        unassigned_robots = [r.robot_id for r in robots if not r.assigned]
        unassigned_tasks = [t.task_id for t in tasks if not t.assigned]

        # Simulate time step
        completed_this_step, unassigned_count, total_reward, total_success = simulate_time_step(
            robots, tasks, unassigned_robots, unassigned_tasks,
            suitability_method, occupied_positions, start_positions, 
            goal_positions, 1.0, total_reward, total_success
        )

        # print(f"DEBUG STATEMENT 8")

        if len(tasks) == 0:
            print(f"All tasks completed in {time_step} time steps!")
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

        # print(f"DEBUG STATEMENT 9")

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

        # print(f"DEBUG STATEMENT 10")

        # Periodically remove robots
        if remove_robots and time_step + 1 <= 4 and random.random() < 0.5: # remove robots only in the first 4 time steps, and randomly
            if len(robots) > 1: # Ensure at least one robot remains in the simulation
                print(f"REMOVING RANDOM ROBOTS AT TIME STEP {time_step + 1}")
                # Keeps track of which robots were removed so their idle state can be cleaned up
                removed_robots = remove_random_robots(robots, tasks, unassigned_robots, unassigned_tasks, random.randint(0, robots_to_remove), occupied_positions, start_positions, goal_positions)
                # Remove stale idle_steps entries for robots that are no longer in the simulation
                for removed_robot in removed_robots: 
                    idle_steps.pop(removed_robot.robot_id, None) 
        # print(f"DEBUG STATEMENT 11")

        for r in robots:
            if not r.assigned:
                idle_steps[r.robot_id] = idle_steps.get(r.robot_id, 0) + 1 # update idle steps of unassigned robots
            else:
                idle_steps[r.robot_id] = 0
        # stalling_robot = any(value >= 5 for value in idle_steps.values()) # if any robot has been idle for 5 or more steps, consider it stalling
        
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

        # print(f"DEBUG STATEMENT 12")

        # Update assigned robots
        assigned_robots = {r.robot_id: r.current_task.task_id for r in robots if r.assigned and r.current_task}

        # decide to reassign and replan
        should_replan = False
        # Allow replanning when either side changes, including dynamic task or robot arrivals
        if unassigned_robots or unassigned_tasks or events["new_tasks"] or events["new_robots"]:
            if events["new_tasks"] or events["new_robots"] or events["completed_tasks"]:
                should_replan = True
            elif (current_active != previous_active) or (current_goals != previous_goals):
                should_replan = True
            # elif stalling_robot:
            #     should_replan = True

        if should_replan:
            should_replan_cbs = True

        # Reassign unassigned robots to unassigned tasks
        if should_replan and start_positions and goal_positions:
            print("State change, re-run CBS...")
            # NOTE: here we need to reassign tasks based on the method used, voting vs optimizer
            if voting_method in voting_methods:
                print(f"REASSIGNING WITH VOTING METHOD: {voting_method_name}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length = V.reassign_robots_to_tasks(
                    robots, tasks, num_candidates, voting_method, suitability_method,
                    unassigned_robots, unassigned_tasks, start_positions, goal_positions
                )
            elif voting_method in optimization_methods:
                print(f"REASSIGNING WITH OPTIMIZATION METHOD: {optimization_method_name}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length = O.reassign_robots_to_tasks_with_method(
                    robots, tasks, num_candidates, voting_method, suitability_method,
                    unassigned_robots, unassigned_tasks, voting_method, start_positions, goal_positions
                )
            total_reassignment_time  += reassign_length
            total_reassignment_score += reassign_score

            # print(f"DEBUG STATEMENT 13")

            # rebuild starts/goals after potential changes from reassignment
            for robot in robots:
                if robot.assigned and robot.current_task:
                    start_positions[robot.robot_id] = tuple(robot.location)
                    goal_positions[robot.robot_id]  = tuple(robot.current_task.location)
                else:
                    start_positions.pop(robot.robot_id, None)
                    goal_positions.pop(robot.robot_id, None)
            print("***************************AFTER REASSIGNMENT***************************")
            print(f"AMOUNT OF ASSIGNED ROBOTS: {len([rob.current_task for rob in robots])}")
            print(f"ASSIGNED ROBOTS: {[rob.assigned for rob in robots].count(True)}")
            print(f"START POSITIONS: {start_positions}")
            print(f"GOAL POSITONS: {goal_positions}")
            print(f"UNASSIGNED ROBOTS: {unassigned_robots}")
            print(f"UNASSIGNED TASKS: {unassigned_tasks}")
            print(f"LIST OF ALL ROBOTS: {[rob.robot_id for rob in robots]}")
            print(f"LIST OF ALL TASKS: {[tas.task_id for tas in tasks]}")
        
        if should_replan_cbs and start_positions and goal_positions:
            agents = build_cbs_agents(robots, start_positions, goal_positions)

            print(f"DEBUG STATEMENT - ENVIRONMENT CREATED - AGENTS BUILT: {agents}")

            # duplicate-start validation
            start_locations = [a['start'] for a in agents]
            if len(start_locations) != len(set(start_locations)):
                print("ERROR: Duplicate start locations found in agent list. Aborting CBS.")
                solution = None
            else:
                env = Environment(dimension=map_dict['dimension'], agents=agents, obstacles=map_dict['obstacles'])
                planner = CBS(env)
                res = planner.search()
                # print(f"CBS COMPLETE. New solution: {solution}")

                print(f"DEBUG STATEMENT - CBS COMPLETE - NODES EXPANDED: {nodes_expanded}, CONFLICTS: {conflicts}")

                if res:
                    solution, nodes_expanded, conflicts = res
                    print(f"CBS COMPLETE. New solution: {solution}")
                    id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
                    for robot_id, schedule in solution.items():
                        ridx = id_to_index[robot_id]
                        r = robots[ridx]
                        r.current_path = [(p['x'], p['y']) for p in schedule]
                        r.remaining_distance = max(0, len(schedule) - 1)

                    previous_active, previous_goals = state_check(robots)  # update to the post-replan state
                    events = {k: 0 for k in events}  # reset counters we just consumed

                    # print(f"DEBUG STATEMENT 16")
                else:
                    print("CBS failed to find a plan under current constraints.")
                    # could possibly fall back on the simple method here if we get a lot of issues
                    # but for now, we will just skip CBS and continue with the simulation
                    print("Skipping CBS...")
                    events = {k: 0 for k in events}  # reset counters even when CBS fails
                    time_steps_unchanged += 1
                    # print(f"DEBUG STATEMENT 17 - TIME STEPS UNCHANGED {time_steps_unchanged}")
                    if time_steps_unchanged >= 3:
                        print("No state change for 3 time steps, ending simulation.")
                        break
        else:
            print("No state change, skip CBS...")
            print(f"ALL ROBOTS: {len(robots)}")
            print(f"ALL TASKS: {len(tasks)}")
            print(f"LIST OF ALL ROBOTS: {[rob.robot_id for rob in robots]}")
            print(f"LIST OF ALL TASKS: {[tas.task_id for tas in tasks]}")
            # print(f"DEBUG STATEMENT 18")

    # print(f"DEBUG STATEMENT 19")
    overall_success_rate = total_success / total_tasks
    print(f"Voting: Total reward: {total_reward}, Overall success rate: {overall_success_rate:.2%}, Tasks completed: {total_success}, Reassignment Time: {total_reassignment_time}, Reassignment Score: {total_reassignment_score}, total reassignments: {total_reassignments}")
    return (total_reward, overall_success_rate, total_success, total_reassignment_time, total_reassignment_score, total_reassignments)
#     for robot in robots:
#         print(f"Robot {robot.robot_id} attempted {robot.tasks_attempted} tasks and successfully completed {robot.tasks_successful} of them.")

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
        task_generation_strict: bool = True):
    start_time = time.time()
    output_tuple = main_simulation(
        output, robots, tasks, num_candidates, voting_method, 
        grid, map_dict, suitability_method, suitability_matrix, 
        max_time_steps, add_tasks, add_robots, remove_robots, 
        tasks_to_add, robots_to_add, robots_to_remove,
        robot_generation_strict, task_generation_strict)
    end_time = time.time()
    execution_time = end_time - start_time

    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used

    print(f"Simulation completed in {execution_time:.2f} seconds.")
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
        voting_names = [
            "rank_assignments_borda", 
            "rank_assignments_approval", 
            "rank_assignments_majority_judgment", 
            "rank_assignments_cumulative_voting", 
            "rank_assignments_condorcet_method", 
            "rank_assignments_range"
            ]
        allocation_methods = [
            O.cbba_task_allocation, 
            O.ssia_task_allocation, 
            O.ilp_task_allocation, 
            O.jv_task_allocation
            ]
        allocation_names = [
            "cbba_task_allocation", 
            "ssia_task_allocation", 
            "ilp_task_allocation", 
            "jv_task_allocation"
            ]
        all_methods = [
            V.rank_assignments_borda,
            V.rank_assignments_approval,
            V.rank_assignments_majority_judgment,
            V.rank_assignments_cumulative_voting,
            V.rank_assignments_condorcet_method,
            V.rank_assignments_range,
            O.cbba_task_allocation,
            O.ssia_task_allocation,
            O.ilp_task_allocation,
            O.jv_task_allocation
            ]
        suitability_methods = [
            S.evaluate_suitability_new, 
            S.evaluate_suitability_loose, 
            S.evaluate_suitability_strict
            ]
        max_time_steps = 100
        robot_sizes = [8, 10, 15]
        # candidate_sizes = [5, 10, 15]
        num_repetitions = 1
        add_tasks = False
        add_robots = False
        remove_robots = False
        robot_generation_strict = True
        task_generation_strict = True
        map_file = r"test_small_open.map"
        # dir_path = r"hvbta\io\results"
        dir_path = os.path.join('hvbta', 'io', 'results')
        grid = load_map(map_file) # 2D list of 0/1 representing the map
        dims = (len(grid), len(grid[0])) # dimensions of the map grid
        obstacles = create_obstacle_list(grid) # list of obstacle coordinates
        map_dict = {
            'dimension': dims,
            'obstacles': obstacles
        }
        with open(os.path.join(dir_path, "simulation_results.csv"), mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Method', 'Suitability Method', 'Num Robots', 
                'Num Tasks', 'Num Candidates', 'Total Score', 
                'Task Normalized Score', 'Score Density', 'Length', 
                'total_reward', 'overall_success_rate', 'total_success', 
                'total_reassignment_time', 'total_reassignment_score', 
                'total_reassignments', 'Execution Time', 'CPU Usage', 'Memory Usage'])
            for num_robots in robot_sizes:
                print(f"\n\n\nSTARTING SIMULATION FOR {num_robots} ROBOTS")
                task_sizes = [
                    # int(num_robots * 0.5),
                    int(num_robots * 0.75),
                    num_robots,
                    int(num_robots * 1.25),
                    # int(num_robots * 1.5)
                ]
                for num_tasks in task_sizes:
                    print(f"\n\n\nSTARTING SIMULATION FOR {num_tasks} TASKS")
                    candidate_sizes = [
                        max(1, int(num_robots * 0.75)),
                            max(1, int(num_tasks * 1)),
                    ]
                    for nc in candidate_sizes:
                        print(f"\n\n\nSTARTING SIMULATION FOR {nc} CANDIDATES")
                        for sm in suitability_methods:
                            print(f"\n\n\nSTARTING SIMULATION FOR SUITABILITY METHOD: {sm}")
                            for rep in range(num_repetitions):
                                print(f"\n\n\nSTARTING SIMULATION REPETITION {rep+1}/{num_repetitions}")
                                voting_outputs = []
                                assignment_infos = []
                                robots = [generate_random_robot_profile_strict(f"R{idx+1}", grid, set()) for idx in range(num_robots)]
                                tasks = [generate_random_task_description_strict(f"T{idx+1}", grid, set(), []) for idx in range(num_tasks)]
                                suitability_matrix = S.calculate_suitability_matrix(robots, tasks, sm)
                                # while suitability_all_zero(suitability_matrix): #change this to just random assignment when there is all zeros
                                #     robots = [generate_random_robot_profile_strict(f"R{idx+1}", grid, set()) for idx in range(num_robots)]
                                #     tasks = [generate_random_task_description_strict(f"T{idx+1}", grid, set(), []) for idx in range(num_tasks)]
                                #     suitability_matrix = S.calculate_suitability_matrix(robots, tasks, sm)
                                if suitability_all_zero(suitability_matrix):
                                    for method_idx in range(len(voting_methods)):
                                        print("All suitability scores are zero, randomly assigning tasks to robots.")
                                        output, score, length = V.assign_tasks_randomly(robots, tasks, suitability_matrix, nc)
                                        assigned_count = len(output[0]) if output and output[0] else 0
                                        task_normalized_score = (score / assigned_count) if assigned_count > 0 else 0.0
                                        score_density = (score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                        # writer.writerow([voting_names[method_idx], sm, num_robots, num_tasks, nc, score, task_normalized_score, score_density, length])
                                        assignment_infos.append([voting_names[method_idx], sm, num_robots, num_tasks, nc, score, task_normalized_score, score_density, length])
                                        voting_outputs.append(output)
                                    cbba_output, cbba_score, cbba_length = O.assign_tasks_with_method_randomly(O.cbba_task_allocation, suitability_matrix, nc)
                                    assigned_count = len(cbba_output[0]) if cbba_output and cbba_output[0] else 0 # nuber of pairs in the chosen assignment
                                    task_normalized_score = (cbba_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (cbba_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["cbba_task_allocation", sm, num_robots, num_tasks, nc, cbba_score, task_normalized_score, score_density, cbba_length])
                                    ssia_output, ssia_score, ssia_length = O.assign_tasks_with_method_randomly(O.ssia_task_allocation, suitability_matrix, nc)
                                    assigned_count = len(ssia_output[0]) if ssia_output and ssia_output[0] else 0
                                    task_normalized_score = (ssia_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (ssia_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["ssia_task_allocation", sm, num_robots, num_tasks, nc, ssia_score, task_normalized_score, score_density, ssia_length])
                                    ilp_output, ilp_score, ilp_length = O.assign_tasks_with_method_randomly(O.ilp_task_allocation, suitability_matrix, nc)
                                    assigned_count = len(ilp_output[0]) if ilp_output and ilp_output[0] else 0
                                    task_normalized_score = (ilp_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (ilp_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["ilp_task_allocation", sm, num_robots, num_tasks, nc, ilp_score, task_normalized_score, score_density, ilp_length])
                                    jv_output, jv_score, jv_length = O.assign_tasks_with_method_randomly(O.jv_task_allocation, suitability_matrix, nc)
                                    assigned_count = len(jv_output[0]) if jv_output and jv_output[0] else 0
                                    task_normalized_score = (jv_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (jv_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["jv_task_allocation", sm, num_robots, num_tasks, nc, jv_score, task_normalized_score, score_density, jv_length])
                                    outputs = voting_outputs + [cbba_output, ssia_output, ilp_output, jv_output]
                                    
                                else:
                                    for method_idx in range(len(voting_methods)):
                                        output, score, length = V.assign_tasks_with_voting(robots, tasks, suitability_matrix, nc, voting_methods[method_idx])
                                        assigned_count = len(output[0]) if output and output[0] else 0 # nuber of pairs in the chosen assignment
                                        task_normalized_score = (score / assigned_count) if assigned_count > 0 else 0.0 # per assigned task normalized score
                                        score_density = (score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                        assignment_infos.append([voting_names[method_idx], sm, num_robots, num_tasks, nc, score, task_normalized_score, score_density, length])
                                        voting_outputs.append(output)
                                    cbba_output, cbba_score, cbba_length = O.assign_tasks_with_method(O.cbba_task_allocation,suitability_matrix)
                                    assigned_count = len(cbba_output[0]) if cbba_output and cbba_output[0] else 0 # nuber of pairs in the chosen assignment
                                    task_normalized_score = (cbba_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (cbba_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["cbba_task_allocation", sm, num_robots, num_tasks, nc, cbba_score, task_normalized_score, score_density, cbba_length])
                                    ssia_output, ssia_score, ssia_length = O.assign_tasks_with_method(O.ssia_task_allocation,suitability_matrix)
                                    assigned_count = len(ssia_output[0]) if ssia_output and ssia_output[0] else 0
                                    task_normalized_score = (ssia_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (ssia_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["ssia_task_allocation", sm, num_robots, num_tasks, nc, ssia_score, task_normalized_score, score_density, ssia_length])
                                    ilp_output, ilp_score, ilp_length = O.assign_tasks_with_method(O.ilp_task_allocation,suitability_matrix)
                                    assigned_count = len(ilp_output[0]) if ilp_output and ilp_output[0] else 0
                                    task_normalized_score = (ilp_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (ilp_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["ilp_task_allocation", sm, num_robots, num_tasks, nc, ilp_score, task_normalized_score, score_density, ilp_length])
                                    jv_output, jv_score, jv_length = O.assign_tasks_with_method(O.jv_task_allocation,suitability_matrix)
                                    assigned_count = len(jv_output[0]) if jv_output and jv_output[0] else 0
                                    task_normalized_score = (jv_score / assigned_count) if assigned_count > 0 else 0.0
                                    score_density = (jv_score / (num_robots * num_tasks)) if (num_robots * num_tasks) > 0 else 0.0
                                    assignment_infos.append(["jv_task_allocation", sm, num_robots, num_tasks, nc, jv_score, task_normalized_score, score_density, jv_length])
                                outputs = voting_outputs + [cbba_output, ssia_output, ilp_output, jv_output]
                                for idx, (out, meth) in enumerate(zip(outputs, all_methods)):
                                    print(f"\n\n\nRUNNING SIMULATION FOR METHOD: {meth}")
                                    output_tuple = benchmark_simulation(
                                        out, copy.deepcopy(robots), copy.deepcopy(tasks), 
                                        nc, meth, grid, map_dict, sm, suitability_matrix, 
                                        max_time_steps, add_tasks, add_robots, remove_robots, 
                                        10, 10, 10, robot_generation_strict, task_generation_strict)
                                    row_prefix = assignment_infos[idx] if idx < len(assignment_infos) else [str(meth), sm, num_robots, num_tasks, nc, 0, 0, 0, 0]
                                    writer.writerow(row_prefix + list(output_tuple))
