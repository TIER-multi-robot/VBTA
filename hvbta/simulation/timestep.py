import random
from typing import List, Tuple
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.allocators.misc_assignment import unassign_task_from_robot
from hvbta.suitability import navigation_suitability


def simulate_time_step(
    robots: List[CapabilityProfile],
    tasks: List[TaskDescription],
    unassigned_robots: List[str],
    unassigned_tasks: List[str],
    suitability_method: str,
    occupied_locations: set,
    start_positions: dict,
    goal_positions: dict,
    time_step: float = 1.0,
    total_reward: float = 0.0,
    total_success: int = 0
) -> Tuple[int, int, float, int]:
    """
    Simulates a single time step, updating robot positions, task progress, and handling failures.

    Parameters:
        robots: List of all robots.
        tasks: List of all tasks.
        time_step: The time increment for the simulation step.
        total_reward: Accumulated reward from successfully completed tasks.

    Returns:
        (tasks_completed, count, total_reward, total_success): A count of unassigned robots and the updated total reward.
    """
    unassigned_count = 0
    tasks_completed = 0

    for robot in robots:
        if robot.assigned and robot.current_task and robot.current_path:
            task = robot.current_task

            # walk one step along the precomputed path (built by CBS upstream)
            if robot.current_path and len(robot.current_path) > 1:
                next_position = robot.current_path[1]
                occupied_locations.discard(robot.location)
                robot.location = next_position
                start_positions[robot.robot_id] = next_position
                occupied_locations.add(next_position)
                robot.current_path.pop(0)
                robot.remaining_distance = max(0, len(robot.current_path) - 1)

                if robot.remaining_distance <= 1:
                    robot.remaining_distance = 0

                if robot.battery_life == 0:
                    unassign_task_from_robot(robot, task, unassigned_robots=unassigned_robots, unassigned_tasks=unassigned_tasks)
                    continue

            # at task location -> work on it
            if robot.remaining_distance == 0:
                required_caps = getattr(task, "required_capabilities", None) or []
                payload_required = None
                for req in required_caps:
                    if isinstance(req, str) and "payload" in req.lower():
                        try:
                            payload_required = float(req.split(">=")[-1].strip())
                        except ValueError:
                            payload_required = None
                tools_needed = getattr(task, "tools_needed", None) or []
                sensors_required = tools_needed[0] if len(tools_needed) > 0 and tools_needed[0] else []
                manipulators_required = tools_needed[1] if len(tools_needed) > 1 and tools_needed[1] else []

                suitability = getattr(robot, "current_task_suitability", 0.5)
                if suitability is None:
                    suitability = 0.5
                suitability = max(0.0, min(1.0, suitability))
                speed_factor = 0.5 + 1.5 * suitability  # [0.5, 2.0]

                robot.location = task.location
                start_positions[robot.robot_id] = robot.location
                robot.battery_life -= time_step
                task.time_left -= time_step * speed_factor

                # task completion
                if task.time_left <= 0:
                    total_reward += task.reward
                    total_success += 1
                    task.assigned = False
                    task.assigned_robot = None
                    robot.current_task = None
                    robot.tasks_successful += 1
                    robot.assigned = False
                    robot.time_on_task = 0
                    robot.current_path = []
                    robot.remaining_distance = 0
                    robot.current_task_suitability = None
                    task.current_suitability = None
                    rid = robot.robot_id
                    if rid not in unassigned_robots:
                        unassigned_robots.append(rid)
                    tasks_completed += 1
                    try:
                        tasks.remove(task)
                    except ValueError:
                        pass

                # battery exhausted while working
                if robot.battery_life <= 0:
                    if getattr(task, "reset_progress", False):
                        task.time_left = task.time_to_complete
                    else:
                        task.time_to_complete = task.time_left
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )

                # safety-compliance stochastic failure
                if task.performance_metrics == "safety compliance":
                    if robot.safety_features:
                        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
                        safety_score = matched_safety / max(1, len(task.safety_protocols))
                        if safety_score < 0.75 and (random.random() > 0.75):
                            unassign_task_from_robot(
                                robot, task,
                                unassigned_robots=unassigned_robots,
                                unassigned_tasks=unassigned_tasks,
                            )
                    else:
                        unassign_task_from_robot(
                            robot, task,
                            unassigned_robots=unassigned_robots,
                            unassigned_tasks=unassigned_tasks,
                        )

                # runtime capability re-checks
                if payload_required is not None and robot.payload_capacity < payload_required:
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    continue

                if manipulators_required and not any(tool in robot.manipulators for tool in manipulators_required):
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    continue

                nav_score = navigation_suitability(
                    robot.mobility_type, robot.size, robot.sensor_range,
                    task.navigation_constraints or [],
                )
                if nav_score == 0.0:
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    continue

        elif not robot.assigned:
            unassigned_count += 1

    return tasks_completed, unassigned_count, total_reward, total_success
