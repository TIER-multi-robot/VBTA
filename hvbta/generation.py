import numpy as np
import random
from typing import List
from hvbta.pathfinding.CBS import get_random_free_position
from . import config as C
from .models import (CapabilityProfile, TaskDescription, AutonomyLevel, TaskTypes, 
                     Priorities, PerformanceMetrics)

from .profiles import STRICT_ROBOT_PROFILES, STRICT_TASK_PROFILES

def _sample(items: List[str], k_min: int = 0, k_max: int = None) -> List[str]:
    """Sample between k_min and k_max items from the list."""
    if not items:
        return []
    if k_max is None:
        k_max = len(items)
    k_min = max(0, min(k_min, len(items)))
    k_max = max(k_min, min(k_max, len(items)))
    k = random.randint(k_min, k_max)
    return random.sample(items, k)

def get_unique_task_location(tasks, grid, occupied_locations, max_attempts=100):
    """
    Finds a unique location for a new task that does not overlap with existing tasks or occupied robot locations.
    Occupied locations only include robot locations, not task locations so we check both with this function.
    Args:
        tasks: List of existing TaskDescription objects to check against.
        grid: The grid representing the environment, used to find free positions.
        occupied_locations: A set of tuples representing locations occupied by robots.
        max_attempts: Maximum number of attempts to find a unique location before giving up.

    Returns:
        location: A unique (x, y, z) location tuple for the new task.
    """
    attempts = 0
    task_locations = {task.location for task in tasks}
    all_occupied_locations = occupied_locations | task_locations  # Combine occupied locations with existing task locations
    location = get_random_free_position(grid, all_occupied_locations)
    # Ensure the location is unique and not occupied by any robot or existing task
    while any(np.allclose(location, occ_loc) for occ_loc in all_occupied_locations):
        attempts += 1
        if attempts >= max_attempts:
            raise Exception(f"Could not find a unique task location after {max_attempts} attempts")
        location = get_random_free_position(grid, all_occupied_locations)
    return location

def generate_random_robot_profile(robot_id: str, grid: List[List[int]], occupied_locations: set) -> CapabilityProfile:
    """Random robot with capability masking but all fields are always present.
    Lists are [] never None. Numeric fields exist even if 0."""
    HYPOTENUSE = (len(grid)**2 + len(grid[0])**2) ** 0.5

    # Base capabilities
    # I decided everything is going to have at least one random capability to increase suitability scores, the mask only
    # removes up to 3 capability groups but nothing that will cause immediate suitability zeroing
    mobility_type = random.choice(C.MOBILITY_TYPES)
    payload_capacity = round(random.uniform(20.0, 50.0), 1)
    reach = 0.0 if mobility_type in {"aerial"} else round(random.uniform(3.0, 10.0), 1)
    # battery_life = round(random.uniform(max(len(grid), len(grid[0])), max(len(grid), len(grid[0]))*3), 1) # takes into account the size of the map
    battery_life = max(500, round(random.uniform(HYPOTENUSE, HYPOTENUSE*10), 1)) # takes into account the size of the map
    size = (round(random.uniform(1.0, 5.0), 2), round(random.uniform(1.0, 5.0), 2), round(random.uniform(1.0, 5.0), 2))  # (length, width, height)
    environmental_resistance = _sample(C.ENV_RESISTANCES, 1)
    sensors = _sample(C.SENSORS, 1)  # should not be empty
    sensor_range = round(random.uniform(20.0, 50.0), 1) if sensors else 0.0
    manipulators = _sample(C.MANIPULATORS, 1) # should not be empty
    communication_protocols = _sample(C.COMM_PROTOCOLS, 1)  # should not be empty
    special_functions = _sample(C.SPECIAL_FUNCS, 1)
    safety_features = _sample(C.SAFETY_FEATS, 1)
    processing_power = round(random.uniform(5.0, 15.0), 1)
    autonomy_level = random.choice(list(AutonomyLevel)).value
    adaptability = bool(random.getrandbits(1))

    # randomly mask 0-3 capability groups but keep types intact for suitability calculations
    groups = ["environmental_resistance", "special_functions", "safety_features"]
    to_mask = _sample(groups, 0, min(3, len(groups)))
    if "environmental_resistance" in to_mask: environmental_resistance = []
    if "special_functions" in to_mask: special_functions = []
    if "safety_features" in to_mask: safety_features = []

    return CapabilityProfile(
        robot_id=robot_id,
        mobility_type=mobility_type,
        max_speed=1.0, # CBS simplification
        payload_capacity=payload_capacity,
        reach=reach,
        battery_life=battery_life,
        size=size,
        environmental_resistance=environmental_resistance, # list
        sensors=sensors,
        sensor_range=sensor_range, # float (0 if no sensors)
        manipulators=manipulators,
        communication_protocols=communication_protocols,
        processing_power=processing_power,
        autonomy_level=autonomy_level,
        special_functions=special_functions,
        safety_features=safety_features,
        adaptability=adaptability,
        location=get_random_free_position(grid, occupied_locations),
        current_task=None,
        remaining_distance=0.0,
        time_on_task=0,
        tasks_attempted=0,
        tasks_successful=0,
        current_path=[],     # CBS path
        assigned=False
    )

def generate_random_task_description(task_id: str, grid: List[List[int]], occupied_locations: set, tasks) -> TaskDescription:
    task_type = random.choice(list(TaskTypes)).value
    # Map of task_type -> (tools_needed list, one required_cap string)
    task_domains = {
        "delivery": (["GPS"], "payload capacity >= 1.0"),
        "inspection": (["camera"], random.choice(["reach >= 1.5", "reach >= 3.0"])),
        "cleaning": (["gripper"], "payload capacity >= 5.0"),
        "monitoring": (["camera"], random.choice(["payload capacity >= 1.0", "payload capacity >= 5.0", "payload capacity >= 10.0"])),
        "maintenance": (["welding tool"], random.choice(["reach >= 1.5", "reach >= 3.0"])),
        "assembly": (["drill"], random.choice(["reach >= 1.5", "reach >= 3.0"])),
        "surveying": (["LiDAR"], random.choice(["payload capacity >= 1.0", "payload capacity >= 5.0"])),
        "data collection": (["temperature sensor"], random.choice(["payload capacity >= 1.0", "payload capacity >= 5.0"])),
        "assistance": (["microphone"], random.choice(["payload capacity >= 1.0", "payload capacity >= 5.0"]))
    }
    tools_needed, req_cap = task_domains.get(task_type, ([], None))
    required_caps = [req_cap] if req_cap else []

    # random tasks can have zero capability constraints to increase suitability scores
    nav = _sample(C.NAV_CONSTRAINTS, 0)
    env = _sample(C.ENV_CONDITIONS, 0)
    safety = _sample(C.SAFETY_FEATS, 0)
    comm = _sample(C.COMM_PROTOCOLS, 0)
    perf = random.choice(list(PerformanceMetrics)).value

    duration = round(random.uniform(3.0, 7.0), 1)
    difficulty = round(random.uniform(1.0, 10.0), 1)
    reward = round(random.uniform(1.0, 10.0), 1)

    return TaskDescription(
        task_id=task_id,
        task_type=task_type,
        objective=f"Perform {task_type} task",
        priority_level=random.choice(list(Priorities)).value,
        reward=reward,
        difficulty=difficulty,
        location=get_unique_task_location(tasks, grid, occupied_locations),
        navigation_constraints=nav,
        required_capabilities=required_caps, # ALWAYS a list
        time_window=(f"{random.randint(8, 17)}:00", f"{random.randint(18, 23)}:00"),
        environmental_conditions=env,
        dependencies=[],
        tools_needed=list(tools_needed), # ALWAYS a list
        communication_requirements=comm,
        safety_protocols=safety,
        performance_metrics=perf, # keep single string for consistency
        success_criteria="Task completed within time window",
        assigned_robot=None,
        time_to_complete=duration,
        assigned=False
    )

def generate_random_robot_profile_strict(robot_id: str, grid: List[List[int]], occupied_locations: set, choice = -1) -> CapabilityProfile:
    HYPOTENUSE = (len(grid)**2 + len(grid[0])**2) ** 0.5

    if choice == -1:
        p = random.choice(STRICT_ROBOT_PROFILES)
    else: 
        p = STRICT_ROBOT_PROFILES[choice]

    return CapabilityProfile(
        robot_id=robot_id,
        mobility_type=p["mobility_type"],
        max_speed=1.0,
        payload_capacity=float(p["payload_capacity"]),
        reach=float(p["reach"]),
        # battery_life=float(p["battery_life"]),
        battery_life=max(500, round(random.uniform(HYPOTENUSE, HYPOTENUSE*10), 1)), # takes into account the size of the map
        size=tuple(p["size"]),
        environmental_resistance=list(p["environmental_resistance"]),
        sensors=list(p["sensors"]),
        sensor_range=float(p["sensor_range"]),
        manipulators=list(p["manipulators"]),
        communication_protocols=list(p["communication_protocols"]),
        processing_power=float(p["processing_power"]),
        autonomy_level=AutonomyLevel(p["autonomy_level"]),
        special_functions=list(p["special_functions"]),
        safety_features=list(p["safety_features"]),
        adaptability=bool(p["adaptability"]),
        location=get_random_free_position(grid, occupied_locations),
        current_task=None,
        remaining_distance=0.0,
        time_on_task=0,
        tasks_attempted=0,
        tasks_successful=0,
        current_path=[],
        assigned=False,
        strict_profile_name=p.get("name", "")
    )

def generate_random_task_description_strict(task_id: str, grid: List[List[int]], occupied_locations: set, tasks, choice = -1) -> TaskDescription:
    PROFILE_POOL = _sample(STRICT_TASK_PROFILES[0:2], 1, 1) + _sample(STRICT_TASK_PROFILES[2:4], 1, 1) + STRICT_TASK_PROFILES[4:]

    if choice == -1:
        p = random.choice(PROFILE_POOL)
    else: 
        p = STRICT_TASK_PROFILES[choice]

    return TaskDescription(
        task_id=task_id,
        task_type=p["task_type"],
        objective=f"Perform {p['task_type']} task",
        priority_level=p["priority_level"],
        reward=float(p["reward"]),
        difficulty=float(p["difficulty"]),
        location=get_unique_task_location(tasks, grid, occupied_locations),
        navigation_constraints=list(p.get("navigation_constraints", [])),
        required_capabilities=p.get("required_capabilities", {}),  # list of strings
        time_window=(f"{random.randint(8, 17)}:00", f"{random.randint(18, 23)}:00"),
        environmental_conditions=list(p.get("environmental_conditions", [])),
        dependencies=[],
        sensors_needed=list(p.get("sensors_needed")),
        manipulators_needed=list(p.get("manipulators_needed")),
        communication_requirements=[c for c in p.get("communication_requirements", []) if c in C.COMM_PROTOCOLS],
        safety_protocols=list(p.get("safety_protocols", [])),
        performance_metrics=p.get("performance_metric", "completion rate"),
        success_criteria="Task completed within time window",
        assigned_robot=None,
        time_to_complete=float(p.get("duration", 10.0)),
        assigned=False,
        strict_profile_name=p.get("nl_description", p.get("task_type", ""))
    )
