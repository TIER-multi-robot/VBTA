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
    unassigned_count = 0  # Count of unassigned robots
    tasks_completed = 0  # Count of tasks completed in this time step

    # Iterate through all robots to update their positions and tasks
    for robot in robots:
        if robot.assigned and robot.current_task and robot.current_path:  # Check that all assigned robots have a task
            # Get the assigned task
            task = robot.current_task

            # check if there is more path to traverse for the robot
            if robot.current_path and len(robot.current_path) > 1:
                next_position = robot.current_path[1] # gives (x, y) coordinate of next step in path
                occupied_locations.discard(robot.location) # remove current location from occupied set
                robot.location = next_position # update location
                # start position for this robot should be replaced, if not then must index by ID
                start_positions[robot.robot_id] = next_position
                occupied_locations.add(next_position) # update occupied set with robots new current location
                robot.current_path.pop(0) # Move the robot one space by removing the first element from the robots path
                robot.remaining_distance = max(0, len(robot.current_path) - 1) # recalculate the remaining distance and new location

                if robot.remaining_distance <= 1:
                    # The robot has reached the task
                    robot.remaining_distance = 0
                    # this wont work here because once the robot reaches the task it will just keep resetting the time on task, I put it down below after completing the task
                    # robot.time_on_task = 0  # Reset time on task so it can begin work (time on task is a counter for how long it takes to complete a task)


                if robot.battery_life == 0: 
                    # print(f"Robot {robot.robot_id} failed to reach task {task.task_id} due to mechanical failure.")
                    # get task id and task index
                    unassign_task_from_robot(robot, task, unassigned_robots=unassigned_robots, unassigned_tasks=unassigned_tasks)
                    continue

            # If the robot is at the task, increment time on task
            if robot.remaining_distance == 0:
                required_caps = getattr(task, "required_capabilities", None) or []
                # print(f"Required capabilities {required_caps}")
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
                # print(f"Tools needed {tools_needed}")
                # print(f"Sensors required {sensors_required} Robot sensors {robot.sensors}")
                # print(f"Manipulators required {manipulators_required} Robot Manipulators {robot.manipulators}")
                # print(f"Payload required {payload_required} Robot payload {robot.payload_capacity}")

                suitability = getattr(robot, "current_task_suitability", 0.5)
                if suitability is None:
                    suitability = 0.5
                
                suitability = max(0.0, min(1.0, suitability))  # Clamp suitability between 0 and 1
                # print(f"Suitability {suitability}")
                speed_factor = 0.5 + 1.5 * suitability  # Speed factor between 0.5 (half speed for 0 suitability) and 2.0 (double speed for 1 suitability)
                #robot.time_on_task += time_step
                robot.location = task.location  # Ensure robot is at the task location
                start_positions[robot.robot_id] = robot.location  # Update start position to task location
                robot.battery_life -= time_step
                task.time_left -= time_step * speed_factor  # Task progresses faster with higher suitability
                # suitability = globals()[suitability_method](robot, task)
                # failure_probability = 1 / (100 * (suitability + 1))  # Higher suitability, lower failure rate
                # Check if the task is completed
                if task.time_left <= 0:
                    # Mark task as completed
                    total_reward += task.reward
                    total_success += 1
                    task.assigned = False
                    task.assigned_robot = None
                    robot.current_task = None
                    robot.tasks_successful += 1
                    robot.assigned = False
                    robot.time_on_task = 0  # Reset time on task so it can begin work (time on task is a counter for how long it takes to complete a task)
                    robot.current_path = []
                    robot.remaining_distance = 0
                    robot.current_task_suitability = None
                    task.current_suitability = None
                    # move it to unassigned robots list with check
                    rid = robot.robot_id
                    if rid not in unassigned_robots:
                        unassigned_robots.append(rid)
                    # print(f"ROBOT {robot.robot_id} COMPLETED TASK {task.task_id}")
                    tasks_completed += 1
                    try:
                        tasks.remove(task)
                    except ValueError:
                        pass
                if robot.battery_life <= 0:
                    if getattr(task, "reset_progress", False): # Only resets progress for certain tasks
                        task.time_left = task.time_to_complete
                    else:
                        task.time_to_complete = task.time_left
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots, 
                        unassigned_tasks=unassigned_tasks
                    )

                if task.performance_metrics == "safety compliance":
                    if robot.safety_features:
                        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
                        safety_score = matched_safety/max(1, len(task.safety_protocols))
                        if safety_score < 0.75 and (random.random() > 0.75):
                            unassign_task_from_robot(
                                robot, task, 
                                unassigned_robots=unassigned_robots, 
                                unassigned_tasks=unassigned_tasks
                            )
                    else:
                        unassign_task_from_robot(
                            robot, task, 
                            unassigned_robots=unassigned_robots, 
                            unassigned_tasks=unassigned_tasks
                        )

                if payload_required is not None and robot.payload_capacity < payload_required:
                    # print(f"Unassign due to payload triggered.")
                    unassign_task_from_robot(
                        robot, task, 
                        unassigned_robots=unassigned_robots, 
                        unassigned_tasks=unassigned_tasks
                    )
                    continue
                
                # if sensors_required and not any(sensor in robot.sensors for sensor in sensors_required):
                #     # print(f"Unassign due to sensor requirements.")
                #     unassign_task_from_robot(
                #         robot, task, 
                #         unassigned_robots=unassigned_robots, 
                #         unassigned_tasks=unassigned_tasks
                #     )
                #     continue

                if manipulators_required and not any(tool in robot.manipulators for tool in manipulators_required):
                    # print(f"Unassign due to manipulator requirements.")
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    continue
                
                nav_score = navigation_suitability(robot.mobility_type, robot.size, robot.sensor_range, task.navigation_constraints or [])
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