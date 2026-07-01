import random
from typing import List, Tuple
from hvbta.models import CapabilityProfile, TaskDescription
import hvbta.generation as G
import numpy as np

def extract_submatrix(full_matrix, robots_subset, tasks_subset, robot_id_to_idx, task_id_to_idx):
    """
    Extract a submatrix from the full suitability matrix for given robot/task subsets.
    Uses pre-computed index mappings for O(1) lookups instead of scorer function calls.
    
    Parameters:
        full_matrix: The full NxM suitability matrix
        robots_subset: List of robot objects to include
        tasks_subset: List of task objects to include
        robot_id_to_idx: Dict mapping robot_id -> row index in full_matrix
        task_id_to_idx: Dict mapping task_id -> column index in full_matrix
    
    Returns:
        Submatrix of shape (len(robots_subset), len(tasks_subset))
    """
    sub_r_indices = [robot_id_to_idx[r.robot_id] for r in robots_subset]
    sub_t_indices = [task_id_to_idx[t.task_id] for t in tasks_subset]
    return full_matrix[np.ix_(sub_r_indices, sub_t_indices)]

def unassign_task_from_robot(robot: CapabilityProfile, task: TaskDescription, unassigned_robots: List[str], unassigned_tasks: List[str]):
    # task_id = robot.current_task.task_id
    # t_index = [task.task_id for task in tasks].index(task_id)
    # unassign task
    task.assigned_robot = None
    task.assigned = False
    robot.assigned = False
    task.current_suitability = None
    robot.current_task_suitability = None

    # move it to unassigned tasks list with check
    tid = task.task_id
    if tid not in unassigned_tasks:
        unassigned_tasks.append(tid)
    # unassign robot
    robot.current_task = None
    robot.current_path = []
    robot.remaining_distance = 0
# move it to unassigned robots list with check
    rid = robot.robot_id
    if rid not in unassigned_robots:
        unassigned_robots.append(rid)
        
def add_new_tasks(tasks: List[TaskDescription], unassigned_tasks: List[str], task_max_id: int, new_task_count: int, total_tasks: int, grid: List[List[int]], occupied_locations: set, strict: bool) -> Tuple[int, int]:
    """
    This function generates new unassigned tasks with random descriptions and adds them to the tasks list.
    It also updates the unassigned tasks list and occupied locations set.
    
    Parameters:
        tasks: List of existing task descriptions.
        unassigned_tasks: List of unassigned task IDs.
        task_max_id: The current maximum task ID to ensure unique IDs for new tasks.
        new_task_count: The number of new tasks to add.
        grid: The grid representing the environment where tasks are located.
        occupied_locations: A set of currently occupied locations to avoid conflicts.
        
    Returns:
        task_max_id: The updated maximum task ID after adding new tasks.
        total_tasks: The updated total number of tasks in the system.
    """
    for _ in range(new_task_count):
        task_id = f"T{task_max_id}"
        task_max_id += 1
        total_tasks += 1
        if strict:
            new_task = G.generate_random_task_description_strict(task_id, grid, occupied_locations, tasks)
        else:
            new_task = G.generate_random_task_description(task_id, grid, occupied_locations, tasks)
        tasks.append(new_task)
        unassigned_tasks.append(task_id)
    return task_max_id, total_tasks

def add_new_robots(robots: List[CapabilityProfile], unassigned_robots: List[str], robot_max_id: int, new_robot_count: int, grid: List[List[int]], occupied_locations: set, strict: bool) -> int:
    """
    This function generates new unassigned robots with random profiles and adds them to the robots list.
    It also updates the unassigned robots list and occupied locations set.
    
    Parameters:
        robots: List of existing robot profiles.
        unassigned_robots: List of unassigned robot IDs.
        robot_max_id: The current maximum robot ID to ensure unique IDs for new robots.
        new_robot_count: The number of new robots to add.
        grid: The grid representing the environment where robots operate.
        occupied_locations: A set of currently occupied locations to avoid conflicts.
        
    Returns:
        robot_max_id: The updated maximum robot ID after adding new robots.
        """
    for _ in range(new_robot_count):
        robot_id = f"R{robot_max_id}"
        robot_max_id += 1
        if strict:
            new_robot = G.generate_random_robot_profile_strict(robot_id, grid, occupied_locations)
        else:
            new_robot = G.generate_random_robot_profile(robot_id, grid, occupied_locations)
        robots.append(new_robot)
        unassigned_robots.append(robot_id)
        occupied_locations.add(new_robot.location)
        # NOTE: do not update start_positions until robots are given a task because they will be in a new location each time they are assigned a task if they get reassigned and CBS recalculates paths
        # NOTE: may need to make robot.current_path an empty list when unassigned to prevent unintentional movement
    return robot_max_id

def remove_random_tasks(
        tasks: List[TaskDescription],
        unassigned_tasks: List[str],
        count: int,
        robots: List[CapabilityProfile],
        unassigned_robots: List[str]) -> List[TaskDescription]:
    """
    Cancels up to `count` random tasks. If a task is currently assigned, the
    responsible robot is freed via unassign_task_from_robot (clearing its path)
    so CBS re-routes it on the next replan.

    Parameters:
        tasks: full task list (mutated - removed tasks are dropped).
        unassigned_tasks: list of pending task IDs (mutated).
        count: upper bound on the number of tasks to cancel. Actual count is
               clamped to len(tasks).
        robots: full robot list (used only for consistency; task's assigned_robot
                pointer is authoritative for the unassign path).
        unassigned_robots: list of robot IDs currently free (mutated when a
                           robot is freed by cancelling its task).

    Returns:
        List of tasks that were removed (for caller cleanup / logging).
    """
    if not tasks or count <= 0:
        return []
    n = min(count, len(tasks))
    tasks_to_remove = random.sample(tasks, n)
    removed = []
    for task in tasks_to_remove:
        assigned_robot = task.assigned_robot
        if assigned_robot is not None:
            unassign_task_from_robot(assigned_robot, task, unassigned_robots, unassigned_tasks)
        if task.task_id in unassigned_tasks:
            unassigned_tasks.remove(task.task_id)
        try:
            tasks.remove(task)
        except ValueError:
            pass
        removed.append(task)
    return removed


def remove_random_robots(robots: List[CapabilityProfile], tasks: List[TaskDescription], unassigned_robots: List[str], unassigned_tasks: List[str], count: int, occupied_locations: set, start_positions: dict, goal_positions: dict):
    """
    This function selects a specified number of robots to remove from the robots list.
    It updates the unassigned robots and tasks lists, as well as the occupied locations set.
    
    Parameters:
        robots: List of existing robot profiles.
        tasks: List of existing task descriptions.
        unassigned_robots: List of unassigned robot IDs.
        unassigned_tasks: List of unassigned task IDs.
        count: The number of robots to remove.
        occupied_locations: A set of currently occupied locations to update.
        start_positions: A dictionary mapping robot IDs to their start positions.
        goal_positions: A dictionary mapping robot IDs to their goal positions.
        
    Returns:
        List of robots that were removed.
    """    
    robots_to_remove = random.sample(robots, min(count, len(robots) - 1))  # Make sure theres always at least one robot so CBS doesnt break
    removed_robots = [] #stores and keeps track of every robot that gets removed during the function


    for robot_to_remove in robots_to_remove:
        occupied_locations.discard(robot_to_remove.location)

        if not robot_to_remove.assigned and robot_to_remove.current_task == None:
            if robot_to_remove.robot_id in unassigned_robots:
                # If the robot is unassigned, just remove it from the list
                unassigned_robots.remove(robot_to_remove.robot_id)
        else:
            task_id = robot_to_remove.current_task.task_id
            t_index = [task.task_id for task in tasks].index(task_id)
            tasks[t_index].assigned_robot = None
            tasks[t_index].assigned = False
            if task_id not in unassigned_tasks:
            # If the task is not already in the unassigned tasks list, add it
                unassigned_tasks.append(task_id)
            # in the case an assigned robot is removed we must update the start_positions and goal_positions lists to remove both locations so they are no longer used in the CBS pathfinding
            start_positions.pop(robot_to_remove.robot_id, None)
            goal_positions.pop(robot_to_remove.robot_id, None)
            # check if the robot is in the unassigned robots list and remove it
            if robot_to_remove.robot_id in unassigned_robots:
                # If the robot is unassigned, just remove it from the list
                unassigned_robots.remove(robot_to_remove.robot_id)
        robots.remove(robot_to_remove)
#         print(f"Robot {robot_to_remove.robot_id} left the system. It attempted {robot_to_remove.tasks_attempted} tasks and successfully completed {robot_to_remove.tasks_successful} of them.")
        removed_robots.append(robot_to_remove)  #Track removed robot so caller can clean up idle_steps
    return removed_robots #Return removed robots so callers can clean up related state
