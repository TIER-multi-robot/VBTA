import psutil
import time
import random
import numpy as np
import csv
import os
import json
import copy
import datetime
from typing import List

from hvbta.pathfinding.Final_CBS import CBS, Environment
from hvbta.simulation.timestep import simulate_time_step
import hvbta.allocators.voting as V
import hvbta.allocators.optimizers as O
import hvbta.suitability as S
import hvbta.generation as G
from hvbta.metrics import (
    calculate_jains_index,
    calculate_threshold_metrics,
    calculate_inequality_metrics,
    calculate_robustness_metrics,
)
from hvbta.pathfinding.CBS import load_map, create_obstacle_list, build_cbs_agents
from hvbta.allocators.misc_assignment import add_new_tasks, add_new_robots, remove_random_robots, remove_random_tasks
from hvbta.models import CapabilityProfile, TaskDescription


def compute_all_fairness_metrics(scores):
    """Flat tuple of fairness metrics in a consistent order for CSV output."""
    jains = calculate_jains_index(scores)
    threshold = calculate_threshold_metrics(scores)
    inequality = calculate_inequality_metrics(scores)
    robustness = calculate_robustness_metrics(scores)
    return (
        jains,
        threshold["below_ge_frac"], threshold["below_good_frac"],
        threshold["deficit_all_ge"], threshold["deficit_below_ge"],
        threshold["deficit_all_good"], threshold["deficit_below_good"],
        inequality["score_range"], inequality["min_max_ratio"],
        inequality["gini"], inequality["cv"],
        robustness["median"], robustness["mean"],
        robustness["med_mean_gap"], robustness["iqr"],
    )


FAIRNESS_METRIC_NAMES = [
    "Jains_Index",
    "Below_GE_Frac", "Below_Good_Frac",
    "Deficit_All_GE", "Deficit_Below_GE",
    "Deficit_All_Good", "Deficit_Below_Good",
    "Score_Range", "Min_Max_Ratio",
    "Gini", "CV",
    "Median", "Mean",
    "Med_Mean_Gap", "IQR",
]


def func_name(f):
    return getattr(f, "__name__", str(f))


def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)


def state_check(robots: List[CapabilityProfile], unassigned_robots: List[str], unassigned_tasks: List[str]):
    """State signature for deciding whether to re-plan."""
    active, goals = [], []
    for r in robots:
        if r.assigned and r.current_task:
            active.append(r.robot_id)
            goals.append((r.robot_id, tuple(r.current_task.location), r.current_task.task_id))
    return (
        tuple(sorted(active)),
        tuple(sorted(goals)),
        tuple(sorted(unassigned_robots)),
        tuple(sorted(unassigned_tasks)),
    )


def run_cbs(robots, start_positions, goal_positions, map_dict):
    """
    Plan all assigned-robot paths jointly with CBS.

    Unassigned robots are added as stationary CBS agents (start == goal == current
    location). This lets CBS route the moving robots around them; otherwise CBS
    would happily plan paths straight through an idle robot's cell, causing
    undetected physical collisions.

    Writes resulting paths back onto each robot's `current_path` and
    `remaining_distance`. Returns the average moving-agent path length over
    successful agents, or 0.0 if CBS fails / no agents.
    """
    if not start_positions or not goal_positions:
        # Even with no assigned agents, unassigned robots are trivially stationary; nothing to plan.
        return 0.0

    # Augment with stationary agents for unassigned robots so CBS sees them as blockers.
    aug_starts = dict(start_positions)
    aug_goals = dict(goal_positions)
    stationary_ids = set()
    for r in robots:
        if r.robot_id in aug_starts:
            continue  # already planned as a mover
        if getattr(r, "location", None) is None:
            continue
        aug_starts[r.robot_id] = r.location
        aug_goals[r.robot_id] = r.location
        stationary_ids.add(r.robot_id)

    agents = build_cbs_agents(robots, aug_starts, aug_goals)
    if not agents:
        return 0.0

    # duplicate-start guard - CBS rejects this anyway, but fail fast and informative
    start_locations = [a['start'] for a in agents]
    if len(start_locations) != len(set(start_locations)):
        print("ERROR: Duplicate start locations in agent list. Skipping CBS.")
        return 0.0

    env = Environment(
        dimension=map_dict['dimension'],
        agents=agents,
        obstacles=map_dict['obstacles'],
    )
    res = CBS(env).search()
    if not res:
        print("CBS failed to find a plan under current constraints.")
        return 0.0

    solution, _nodes_expanded, _conflicts = res
    id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
    valid_lengths = []
    for robot_id, schedule in solution.items():
        if robot_id not in id_to_index:
            continue
        r = robots[id_to_index[robot_id]]
        # Skip stationary blockers - do not overwrite their (empty) current_path with the trivial wait schedule.
        if robot_id in stationary_ids:
            continue
        r.current_path = [(p['x'], p['y']) for p in schedule]
        r.remaining_distance = max(0, len(schedule) - 1)
        if len(schedule) > 1:
            valid_lengths.append(len(schedule) - 1)
    return float(np.mean(valid_lengths)) if valid_lengths else 0.0


def reassign_robots_to_tasks_direct(
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        bypass_fn,
        unassigned_robots: List[str],
        unassigned_tasks: List[str],
        start_positions: dict,
        goal_positions: dict,
        llm_cache: dict = None):
    """
    Replan via a direct-assignment LLM (e.g. bypass_suitability_from_names_with_llm).
    Mirrors the return shape of V.reassign_robots_to_tasks /
    O.reassign_robots_to_tasks_with_method so the main loop can dispatch uniformly:
        return_assignments, unassigned_robots, unassigned_tasks, score, length, per_agent_scores
    The bypass operates only on the unassigned subset, so it never steals tasks from
    busy robots. If a matrix cache is supplied, per-pair scores are read out of it for
    the fairness metrics; otherwise per-pair scores default to 0.5 sentinel.
    """
    urobots = [r for r in robots if not r.assigned]
    utasks = [t for t in tasks if not t.assigned]
    if not urobots or not utasks:
        return {}, unassigned_robots, unassigned_tasks, 0.0, 0.0, []

    t0 = time.perf_counter_ns()
    (sub_pairs, _unr_sub, _unt_sub), parse_failed = bypass_fn(urobots, utasks)
    length = (time.perf_counter_ns() - t0) / 1000.0

    if parse_failed:
        print("[direct LLM replan] parse failed - leaving unassigned this tick")
        return {}, unassigned_robots, unassigned_tasks, 0.0, length, []

    cache_matrix = llm_cache.get("matrix") if llm_cache else None
    r_to_idx = llm_cache.get("robot_id_to_idx") if llm_cache else None
    t_to_idx = llm_cache.get("task_id_to_idx") if llm_cache else None

    return_assignments = {}
    score = 0.0
    for ri, ti in sub_pairs:
        r = urobots[ri]
        t = utasks[ti]
        pair_score = 0.5
        if (cache_matrix is not None and r_to_idx and t_to_idx
                and r.robot_id in r_to_idx and t.task_id in t_to_idx):
            pair_score = float(cache_matrix[r_to_idx[r.robot_id], t_to_idx[t.task_id]])

        r.current_task = t
        r.tasks_attempted += 1
        r.assigned = True
        t.assigned_robot = r
        t.assigned = True
        r.current_task_suitability = pair_score
        t.current_suitability = pair_score
        start_positions[r.robot_id] = r.location
        goal_positions[r.robot_id] = t.location
        return_assignments[r.robot_id] = t.task_id
        score += pair_score

    new_unassigned_robots = [r.robot_id for r in robots if not r.assigned]
    new_unassigned_tasks = [t.task_id for t in tasks if not t.assigned]
    per_agent_scores = [
        float(r.current_task_suitability)
        for r in robots
        if r.assigned and r.current_task is not None and r.current_task_suitability is not None
    ]
    return return_assignments, new_unassigned_robots, new_unassigned_tasks, score, length, per_agent_scores


def main_simulation(
        output: tuple,
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
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
        initial_jains_index: float = 0.0,
        # Lifelong event rates: per-timestep probability of an arrival/removal check firing.
        # Old <=2 / <=4 hard-caps are gone; set 0.0 to disable a channel.
        task_arrival_rate: float = 0.05,
        robot_arrival_rate: float = 0.02,
        robot_departure_rate: float = 0.02,
        task_cancellation_rate: float = 0.02,
        remove_tasks: bool = False,
        tasks_to_remove: int = 1):
    print(f"SUITABILITY METHOD: {suitability_method}")

    voting_methods = {
        V.rank_assignments_borda: "Borda Count",
        V.rank_assignments_approval: "Approval Voting",
        V.rank_assignments_majority_judgment: "Majority Judgment",
        V.rank_assignments_cumulative_voting: "Cumulative Voting",
        V.rank_assignments_condorcet_method: "Condorcet Method",
        V.rank_assignments_range: "Range Voting",
    }
    optimization_methods = {
        O.cbba_task_allocation: "CBBA",
        O.ssia_task_allocation: "SSIA",
        O.ilp_task_allocation: "ILP",
        O.jv_task_allocation: "JV",
    }
    voting_method_name = voting_methods.get(voting_method, "Unknown Method")
    optimization_method_name = optimization_methods.get(voting_method, "Unknown Method")

    HYPOTENUSE = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    total_reward = 0.0
    total_success = 0.0
    total_tasks = len(tasks)  # tasks_spawned - grows with dynamic additions
    # Set of every task_id that was ever assigned to a robot. Denominator for
    # attempted_completion_rate, which excludes late-arriving tasks that never
    # got a chance to be attempted before max_time_steps.
    tasks_ever_assigned = set()
    total_reassignment_time = 0.0
    total_reassignment_score = 0.0
    total_reassignments = 0
    total_time_steps = max_time_steps
    reassignment_jains_indices = []
    reassignment_fairness_metrics = []
    reassign_score = 0.0
    reassign_length = 0.0

    # LLM suitability cache - reuse matrix unless NEW entities are added
    llm_cache = {
        "matrix": suitability_matrix if getattr(suitability_method, "_is_llm_batch", False) else None,
        "robot_id_to_idx": {r.robot_id: i for i, r in enumerate(robots)} if getattr(suitability_method, "_is_llm_batch", False) else None,
        "task_id_to_idx": {t.task_id: j for j, t in enumerate(tasks)} if getattr(suitability_method, "_is_llm_batch", False) else None,
    }

    occupied_positions = set(r.location for r in robots)

    # apply initial assignment
    assigned_pairs = output[0]
    for robot_idx, task_idx in assigned_pairs:
        r = robots[robot_idx]
        t = tasks[task_idx]
        r.current_task = t
        r.assigned = True
        r.tasks_attempted = 1
        t.assigned_robot = r
        t.assigned = True
        tasks_ever_assigned.add(t.task_id)

    unassigned_tasks = [t.task_id for t in tasks if not t.assigned]
    unassigned_robots = [r.robot_id for r in robots if not r.assigned]
    start_positions = {r.robot_id: r.location for r in robots if r.assigned and r.current_task}
    goal_positions = {r.robot_id: r.current_task.location for r in robots if r.assigned and r.current_task}

    avg_path_length = run_cbs(robots, start_positions, goal_positions, map_dict)

    # create the initial state
    previous_active, previous_goals, previous_unassigned_robots, previous_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)
    events = {"new_tasks": 0, "new_robots": 0, "completed_tasks": 0}
    idle_steps = {r.robot_id: 0 for r in robots}
    robot_max_id = len(robots) + 1
    task_max_id = len(tasks) + 1

    for time_step in range(max_time_steps):
        # update unnasigned robots and tasks
        unassigned_robots = [r.robot_id for r in robots if not r.assigned]
        unassigned_tasks = [t.task_id for t in tasks if not t.assigned]

        # simulate timestep
        completed_this_step, _unassigned_count, total_reward, total_success = simulate_time_step(
            robots, tasks, unassigned_robots, unassigned_tasks,
            suitability_method, occupied_positions, start_positions,
            goal_positions, 1.0, total_reward, total_success,
        )

        # check if no tasks are left
        if len(tasks) == 0:
            print(f"All tasks completed in {time_step + 1} time steps!")
            total_time_steps = time_step + 1
            break

        # update events with any completed tasks this timestep
        events["completed_tasks"] += completed_this_step
        # set the replan_cbs flag
        should_replan_cbs = completed_this_step > 0

        # Lifelong dynamic events: each channel fires with per-timestep probability.
        # add tasks
        if add_tasks and random.random() < task_arrival_rate:
            print(f"ADDING NEW TASKS AT TIME STEP {time_step + 1}")
            n = random.randint(1, max(1, tasks_to_add))
            task_max_id, total_tasks = add_new_tasks(
                tasks, unassigned_tasks, task_max_id, n, total_tasks,
                grid, occupied_positions, task_generation_strict,
            )
            events["new_tasks"] += n

        # add robots
        if add_robots and random.random() < robot_arrival_rate:
            print(f"ADDING NEW ROBOTS AT TIME STEP {time_step + 1}")
            n = random.randint(1, max(1, robots_to_add))
            robot_max_id = add_new_robots(
                robots, unassigned_robots, robot_max_id, n,
                grid, occupied_positions, robot_generation_strict,
            )
            events["new_robots"] += n
            for r in robots:
                idle_steps.setdefault(r.robot_id, 0)

        # remove robots
        if remove_robots and random.random() < robot_departure_rate:
            if len(robots) > 1:
                print(f"REMOVING RANDOM ROBOTS AT TIME STEP {time_step + 1}")
                removed = remove_random_robots(
                    robots, tasks, unassigned_robots, unassigned_tasks,
                    random.randint(1, max(1, robots_to_remove)),
                    occupied_positions, start_positions, goal_positions,
                )
                for removed_robot in removed:
                    idle_steps.pop(removed_robot.robot_id, None)

        # cancel tasks
        if remove_tasks and random.random() < task_cancellation_rate:
            if len(tasks) > 0:
                print(f"CANCELLING RANDOM TASKS AT TIME STEP {time_step + 1}")
                removed_t = remove_random_tasks(
                    tasks, unassigned_tasks,
                    random.randint(1, max(1, tasks_to_remove)),
                    robots, unassigned_robots,
                )
                # Cancellations free their robots via unassign_task_from_robot, so a
                # replan is warranted; treat it as a completion-shaped event.
                events["completed_tasks"] += len(removed_t)

        # increment or reset idle steps for idle robots
        for r in robots:
            if not r.assigned:
                idle_steps[r.robot_id] = idle_steps.get(r.robot_id, 0) + 1
            else:
                idle_steps[r.robot_id] = 0

        # refresh start/goal
        for robot in robots:
            if robot.assigned and robot.current_task:
                start_positions[robot.robot_id] = robot.location
                goal_positions[robot.robot_id] = robot.current_task.location
            else:
                start_positions.pop(robot.robot_id, None)
                goal_positions.pop(robot.robot_id, None)

        # update state
        current_active, current_goals, current_unassigned_robots, current_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)

        # check if there are any new events
        should_replan = False
        if unassigned_robots and unassigned_tasks:
            if events["new_tasks"] or events["new_robots"] or events["completed_tasks"]:
                should_replan = True
            # check if there is a state change
            elif (
                current_active != previous_active
                or current_goals != previous_goals
                or current_unassigned_robots != previous_unassigned_robots
                or current_unassigned_tasks != previous_unassigned_tasks
            ):
                should_replan = True
        if should_replan:
            should_replan_cbs = True

        if should_replan:
            # determine what to pass to the reassignment functions
            if getattr(suitability_method, "_is_llm_batch", False):
                current_robot_ids = {r.robot_id for r in robots}
                current_task_ids = {t.task_id for t in tasks}
                # check the LLM cache is valid
                cache_valid = (
                    llm_cache["matrix"] is not None
                    and llm_cache["robot_id_to_idx"] is not None
                    and llm_cache["task_id_to_idx"] is not None
                    and current_robot_ids.issubset(llm_cache["robot_id_to_idx"].keys())
                    and current_task_ids.issubset(llm_cache["task_id_to_idx"].keys())
                )
                # update it if not
                if not cache_valid:
                    print(f"LLM cache miss - rebuilding matrix for {len(robots)} robots, {len(tasks)} tasks")
                    result = suitability_method(robots, tasks)
                    # Unwrap (matrix, parse_failed) tuples so the cache stores a bare ndarray.
                    suitability_matrix = result[0] if isinstance(result, tuple) else result
                    llm_cache["matrix"] = suitability_matrix
                    llm_cache["robot_id_to_idx"] = {r.robot_id: i for i, r in enumerate(robots)}
                    llm_cache["task_id_to_idx"] = {t.task_id: j for j, t in enumerate(tasks)}
                # use the updated cache as suitability source if LLM is suitibility method
                suitability_source = (llm_cache["matrix"], llm_cache["robot_id_to_idx"], llm_cache["task_id_to_idx"])
            else:
                suitability_source = suitability_method

            # assign with voting
            if voting_method in voting_methods:
                print(f"REASSIGNING WITH VOTING METHOD: {voting_method_name}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = V.reassign_robots_to_tasks(
                    robots, tasks, num_candidates, voting_method, suitability_source,
                    unassigned_robots, unassigned_tasks, start_positions, goal_positions, HYPOTENUSE,
                )
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))
            
            # assign with optimization
            elif voting_method in optimization_methods:
                print(f"REASSIGNING WITH OPTIMIZATION METHOD: {optimization_method_name}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = O.reassign_robots_to_tasks_with_method(
                    robots, tasks, num_candidates, voting_method, suitability_source,
                    unassigned_robots, unassigned_tasks, voting_method, start_positions, goal_positions, HYPOTENUSE,
                )
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))

            # assign with direct LLM (bypass the matrix)
            elif getattr(voting_method, "_is_llm_direct", False):
                print("REASSIGNING WITH DIRECT LLM ASSIGNMENT")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = reassign_robots_to_tasks_direct(
                    robots, tasks, voting_method,
                    unassigned_robots, unassigned_tasks,
                    start_positions, goal_positions,
                    llm_cache=llm_cache,
                )
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))
            total_reassignment_time += reassign_length
            total_reassignment_score += reassign_score

            # update location dicts
            for robot in robots:
                if robot.assigned and robot.current_task:
                    start_positions[robot.robot_id] = tuple(robot.location)
                    goal_positions[robot.robot_id] = tuple(robot.current_task.location)
                else:
                    start_positions.pop(robot.robot_id, None)
                    goal_positions.pop(robot.robot_id, None)

        # Record any task that is currently assigned - this is the "attempted" set.
        # We do this every timestep (cheap) so cancellations of never-attempted tasks
        # do not incorrectly count them.
        for t in tasks:
            if t.assigned:
                tasks_ever_assigned.add(t.task_id)

        # rerun cbs if needed, update state, and clear events
        if should_replan_cbs and start_positions and goal_positions:
            print("\nRERUNNING CBS\n")
            avg_path_length = run_cbs(robots, start_positions, goal_positions, map_dict) or avg_path_length
            previous_active, previous_goals, previous_unassigned_robots, previous_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)
            events = {k: 0 for k in events}

    overall_success_rate = total_success / total_tasks if total_tasks else 0.0
    attempted_completion_rate = total_success / len(tasks_ever_assigned) if tasks_ever_assigned else 0.0
    avg_reassignment_score = (total_reassignment_score / total_reassignments) if total_reassignments > 0 else 0.0
    avg_reassignment_jains_index = (sum(reassignment_jains_indices) / len(reassignment_jains_indices)) if reassignment_jains_indices else 0.0

    if reassignment_fairness_metrics:
        num_metrics = len(FAIRNESS_METRIC_NAMES)
        avg_reassign_metrics = tuple(
            sum(m[i] for m in reassignment_fairness_metrics) / len(reassignment_fairness_metrics)
            for i in range(num_metrics)
        )
    else:
        avg_reassign_metrics = tuple(0.0 for _ in FAIRNESS_METRIC_NAMES)

    print(f"Voting: Total reward: {total_reward}, "
          f"Overall success rate: {overall_success_rate:.2%} ({total_success}/{total_tasks} spawned), "
          f"Attempted completion rate: {attempted_completion_rate:.2%} ({total_success}/{len(tasks_ever_assigned)} attempted), "
          f"Reassignment Time: {total_reassignment_time}, "
          f"Reassignment Score: {total_reassignment_score}, total reassignments: {total_reassignments}, "
          f"Total robots: {len(robots)}")

    return (
        total_reward, overall_success_rate, total_success,
        total_reassignment_time, total_reassignment_score, total_reassignments,
        min(total_time_steps, max_time_steps), avg_reassignment_score, avg_path_length,
        initial_jains_index, avg_reassignment_jains_index,
        attempted_completion_rate, len(tasks_ever_assigned), total_tasks,
    ) + avg_reassign_metrics[1:]  # skip first (Jains) - already exposed above


def benchmark_simulation(
        output: tuple,
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
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
        initial_jains_index: float = 0.0,
        task_arrival_rate: float = 0.05,
        robot_arrival_rate: float = 0.02,
        robot_departure_rate: float = 0.02,
        task_cancellation_rate: float = 0.02,
        remove_tasks: bool = False,
        tasks_to_remove: int = 1):
    start_time = time.perf_counter_ns()
    output_tuple = main_simulation(
        output, robots, tasks, num_candidates, voting_method,
        grid, map_dict, suitability_method, suitability_matrix,
        max_time_steps, add_tasks, add_robots, remove_robots,
        tasks_to_add, robots_to_add, robots_to_remove,
        robot_generation_strict, task_generation_strict,
        initial_jains_index=initial_jains_index,
        task_arrival_rate=task_arrival_rate,
        robot_arrival_rate=robot_arrival_rate,
        robot_departure_rate=robot_departure_rate,
        task_cancellation_rate=task_cancellation_rate,
        remove_tasks=remove_tasks,
        tasks_to_remove=tasks_to_remove,
    )
    execution_time = time.perf_counter_ns() - start_time
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used

    print(f"Simulation completed in {execution_time:.5f} nanoseconds.")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_usage / (1024 * 1024)} MB")

    return output_tuple + (execution_time, cpu_usage, memory_usage)


def _record_assignment(assignment_infos, run_id, method_name, sm_name, num_robots,
                       num_tasks, nc, score, length, per_agent_scores):
    """Compute initial fairness metrics, derived score fields, and append a row."""
    metrics = compute_all_fairness_metrics(per_agent_scores)
    assigned_count = len(per_agent_scores) if per_agent_scores else 0
    task_normalized = (score / assigned_count) if assigned_count > 0 else 0.0
    score_density = (score / (num_robots * num_tasks)) if num_robots * num_tasks > 0 and score > 0 else 0.0
    assignment_infos.append(
        [run_id, method_name, sm_name, num_robots, num_tasks, nc, score,
         task_normalized, score_density, length] + list(metrics)
    )
    return metrics[0]  # initial jains


if __name__ == "__main__":
    voting_methods = [
        V.rank_assignments_borda,
        V.rank_assignments_approval,
        V.rank_assignments_majority_judgment,
        V.rank_assignments_cumulative_voting,
        V.rank_assignments_condorcet_method,
        V.rank_assignments_range,
    ]
    voting_names = [func_name(f) for f in voting_methods]
    optimization_methods = [
        O.cbba_task_allocation,
        O.ssia_task_allocation,
        O.ilp_task_allocation,
        O.jv_task_allocation,
    ]
    optimization_names = [func_name(f) for f in optimization_methods]
    # Direct-assignment methods: the LLM returns (robot_id, task_id) pairs straight up,
    # bypassing the suitability-matrix + voting/optimization stack. Slot here so it
    # benchmarks alongside the others.
    direct_methods = [
        # S.bypass_suitability_from_names_with_llm,
    ]
    direct_names = [func_name(f) for f in direct_methods]
    all_methods = voting_methods + optimization_methods + direct_methods
    suitability_methods = [
        S.evaluate_suitability_balanced,
        S.evaluate_suitability_loose,
        S.evaluate_suitability_strict,
        # S.evaluate_suitability_from_names_with_llm,
    ]

    small_maps = [
        r"den201d.map", r"den202d.map", r"den404d.map", r"lak101d.map",
        r"lak102d.map", r"lak105d.map", r"lak107d.map", r"lak108d.map",
    ]
    medium_maps = [
        r"arena.map", r"den009d.map", r"den101d.map", r"den204d.map",
        r"den207d.map", r"den403d.map", r"den405d.map", r"den407d.map",
        r"den408d.map", r"hrt002d.map", r"isound1.map", r"lak103d.map",
        r"lak104d.map",
    ]
    large_maps = [
        r"den001d.map", r"den020d.map", r"den203d.map", r"den206d.map",
        r"den308d.map", r"den312d.map", r"den900d.map", r"den901d.map",
        r"den998d.map", r"hrt001d.map", r"lak106d.map", r"lak203d.map",
        r"lak307d.map", r"ost002d.map",
    ]
    map_paths = (
        random.sample(small_maps, 1)
        # + random.sample(medium_maps, 1)
        # + random.sample(large_maps, 1)
    )

    robot_sizes = [2]
    task_sizes = [10]
    Run_ID = 1
    num_repetitions = 1
    add_tasks = True
    add_robots = True
    remove_robots = True
    tasks_to_add = 1
    robots_to_add = 1
    robots_to_remove = 1
    robot_generation_strict = True
    task_generation_strict = True
    task_arrival_rate = 0.05
    robot_arrival_rate = 0.02
    robot_departure_rate = 0.02
    task_cancellation_rate = 0.02
    remove_tasks = True
    tasks_to_remove = 1
    map_dir = r"MAPF_benchmark_maps"

    dir_path = os.path.join('hvbta', 'io', 'results')
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(dir_path, exist_ok=True)
    full_paths = [os.path.join(map_dir, m) for m in map_paths]

    for map_file in full_paths:
        grid = load_map(map_file)
        HYPOTENUSE = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
        dims = (len(grid), len(grid[0]))
        map_size = "Small" if dims[0] < 40 and dims[1] < 40 else "Medium" if dims[0] < 75 and dims[1] < 75 else "Large"
        obstacles = create_obstacle_list(grid)
        map_dict = {'dimension': dims, 'obstacles': obstacles}

        results_path = os.path.join(dir_path, f"{stamp}_simulation_{os.path.basename(map_file)}.csv")
        profiles_path = os.path.join(dir_path, f"{stamp}_profiles_{os.path.basename(map_file)}.csv")

        with open(results_path, mode="w", newline='') as file, \
             open(profiles_path, mode="w", newline='') as profile_file:
            writer = csv.writer(file)
            profiles_w = csv.writer(profile_file)

            writer.writerow([
                'Run ID', 'Method', 'Suitability Method', 'Num Robots',
                'Num Tasks', 'Num Candidates', 'Total Score',
                'Task Normalized Score', 'Score Density', 'Length',
                'Init_Jains_Index', 'Init_Below_GE_Frac', 'Init_Below_Good_Frac',
                'Init_Deficit_All_GE', 'Init_Deficit_Below_GE',
                'Init_Deficit_All_Good', 'Init_Deficit_Below_Good',
                'Init_Score_Range', 'Init_Min_Max_Ratio',
                'Init_Gini', 'Init_CV',
                'Init_Median', 'Init_Mean',
                'Init_Med_Mean_Gap', 'Init_IQR',
                'total_reward', 'overall_success_rate', 'total_success',
                'total_reassignment_time', 'total_reassignment_score',
                'total_reassignments', 'Total Time Steps', 'Average Reassignment Score',
                'Average Path Length', 'Initial Jains Index (sim)', 'Avg Reassignment Jains Index',
                # Lifelong-operation metrics inserted here to line up with the
                # new tail of the main_simulation return tuple.
                'Attempted_Completion_Rate', 'Tasks_Attempted', 'Tasks_Spawned',
                'Avg_Reass_Below_GE_Frac', 'Avg_Reass_Below_Good_Frac',
                'Avg_Reass_Deficit_All_GE', 'Avg_Reass_Deficit_Below_GE',
                'Avg_Reass_Deficit_All_Good', 'Avg_Reass_Deficit_Below_Good',
                'Avg_Reass_Score_Range', 'Avg_Reass_Min_Max_Ratio',
                'Avg_Reass_Gini', 'Avg_Reass_CV',
                'Avg_Reass_Median', 'Avg_Reass_Mean',
                'Avg_Reass_Med_Mean_Gap', 'Avg_Reass_IQR',
                'Execution Time', 'CPU Usage', 'Memory Usage', 'Map Size',
            ])
            profiles_w.writerow([
                'Run_ID', 'Map', 'Num Robots', 'Num Tasks',
                'Suitability Method', 'RobotProfiles', 'TaskProfiles',
            ])

            for num_robots in robot_sizes:
                print(f"\n\n\nSTARTING SIMULATION FOR {num_robots} ROBOTS")
                for num_tasks in task_sizes:
                    print(f"\n\n\nSTARTING SIMULATION FOR {num_tasks} TASKS")
                    candidate_sizes = [50]
                    workload = max(1.0, num_tasks / num_robots)
                    extender = 1.5 if workload > 10 else 1.2 if workload > 5 else 1.0
                    max_time_steps = max(200, int(HYPOTENUSE * 1.5 * extender))

                    for nc in candidate_sizes:
                        for sm in suitability_methods:
                            sm_name = func_name(sm)
                            for rep in range(num_repetitions):
                                print(f"\n\n\nSTARTING SIMULATION REPETITION {rep+1}/{num_repetitions}")
                                voting_outputs = []
                                assignment_infos = []

                                # Spawn rule: robots may not overlap each other, tasks may not overlap
                                # each other, but a robot and a task may share a cell.
                                robot_locs = set()
                                robots = []
                                gen_robot = G.generate_random_robot_profile_strict if robot_generation_strict else G.generate_random_robot_profile
                                for idx in range(num_robots):
                                    r = gen_robot(f"R{idx+1}", grid, robot_locs)
                                    robots.append(r)
                                    robot_locs.add(r.location)
                                robot_profiles = [r.strict_profile_name for r in robots] if robot_generation_strict else []

                                task_locs = set()
                                tasks = []
                                gen_task = G.generate_random_task_description_strict if task_generation_strict else G.generate_random_task_description
                                for idx in range(num_tasks):
                                    t = gen_task(f"T{idx+1}", grid, task_locs, [])
                                    tasks.append(t)
                                    task_locs.add(t.location)
                                task_profiles = [t.strict_profile_name for t in tasks] if task_generation_strict else []

                                if getattr(sm, "_is_llm_batch", False):
                                    result = sm(robots, tasks)
                                    # LLM scorers return (matrix, parse_failed); unwrap so the
                                    # matrix is what flows through to llm_cache and downstream
                                    # indexing in reassign_robots_to_tasks_direct.
                                    suitability_matrix = result[0] if isinstance(result, tuple) else result
                                else:
                                    suitability_matrix = S.calculate_suitability_matrix(robots, tasks, sm, HYPOTENUSE)

                                if robot_generation_strict and task_generation_strict:
                                    profiles_w.writerow([
                                        Run_ID, os.path.basename(map_file),
                                        num_robots, num_tasks, sm_name,
                                        json.dumps(robot_profiles), json.dumps(task_profiles),
                                    ])

                                # Direct LLM assignment is independent of the matrix; time it and
                                # score its chosen pairs against the rule-based matrix for parity
                                # with voting/optimization rows.
                                direct_outputs = []
                                for d_fn, d_name in zip(direct_methods, direct_names):
                                    d_start = time.perf_counter_ns()
                                    out, parse_failed = d_fn(robots, tasks)
                                    d_length = (time.perf_counter_ns() - d_start) / 1000.0
                                    if parse_failed:
                                        print(f"[{d_name}] parse failed - empty assignment fallback")
                                    pairs = out[0]
                                    per_agent_direct = [float(suitability_matrix[r][t]) for (r, t) in pairs] if not isinstance(suitability_matrix, tuple) else [0.0] * len(pairs)
                                    d_score = sum(per_agent_direct)
                                    initial_jains = _record_assignment(assignment_infos, Run_ID, d_name, sm_name, num_robots, num_tasks, nc, d_score, d_length, per_agent_direct)
                                    direct_outputs.append((out, initial_jains))

                                if suitability_all_zero(suitability_matrix):
                                    # voting - random fallback
                                    for method_fn, method_name in zip(voting_methods, voting_names):
                                        output, score, length, per_agent = V.assign_tasks_randomly(robots, tasks, suitability_matrix, nc)
                                        initial_jains = _record_assignment(assignment_infos, Run_ID, method_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                        voting_outputs.append((output, initial_jains))
                                    # optimization - random fallback
                                    opt_outputs = []
                                    for opt_fn, opt_name in zip(optimization_methods, optimization_names):
                                        out, score, length, per_agent = O.assign_tasks_with_method_randomly(opt_fn, suitability_matrix, nc)
                                        initial_jains = _record_assignment(assignment_infos, Run_ID, opt_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                        opt_outputs.append((out, initial_jains))
                                    outputs = voting_outputs + opt_outputs + direct_outputs
                                else:
                                    # voting - normal
                                    for method_fn, method_name in zip(voting_methods, voting_names):
                                        output, score, length, per_agent = V.assign_tasks_with_voting(robots, tasks, suitability_matrix, nc, method_fn)
                                        initial_jains = _record_assignment(assignment_infos, Run_ID, method_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                        voting_outputs.append((output, initial_jains))
                                    # optimization - normal
                                    opt_outputs = []
                                    for opt_fn, opt_name in zip(optimization_methods, optimization_names):
                                        out, score, length, per_agent = O.assign_tasks_with_method(opt_fn, suitability_matrix)
                                        initial_jains = _record_assignment(assignment_infos, Run_ID, opt_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                        opt_outputs.append((out, initial_jains))
                                    outputs = voting_outputs + opt_outputs + direct_outputs

                                for idx, (out_tuple, meth) in enumerate(zip(outputs, all_methods)):
                                    out, initial_jains = out_tuple
                                    output_tuple = benchmark_simulation(
                                        output=out, 
                                        robots=copy.deepcopy(robots), 
                                        tasks=copy.deepcopy(tasks),
                                        num_candidates=nc, 
                                        voting_method=meth, 
                                        grid=grid, 
                                        map_dict=map_dict, 
                                        suitability_method=sm, 
                                        suitability_matrix=suitability_matrix, 
                                        max_time_steps=max_time_steps, 
                                        add_tasks=add_tasks, 
                                        add_robots=add_robots, 
                                        remove_robots=remove_robots,
                                        tasks_to_add=tasks_to_add, 
                                        robots_to_add=robots_to_add, 
                                        robots_to_remove=robots_to_remove,
                                        robot_generation_strict=robot_generation_strict, 
                                        task_generation_strict=task_generation_strict,
                                        initial_jains_index=initial_jains, 
                                        task_arrival_rate=task_arrival_rate,
                                        robot_arrival_rate=robot_arrival_rate, 
                                        robot_departure_rate=robot_departure_rate,
                                        task_cancellation_rate=task_cancellation_rate, 
                                        remove_tasks=remove_tasks, 
                                        tasks_to_remove=tasks_to_remove
                                    )
                                    row_prefix = assignment_infos[idx] if idx < len(assignment_infos) else [Run_ID, func_name(meth), sm_name, num_robots, num_tasks, nc, 0, 0, 0, 0]
                                    writer.writerow(row_prefix + list(output_tuple) + [map_size])
                                Run_ID += 1
