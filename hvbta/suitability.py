import os
import logging
import datetime
import numpy as np
import json, re, ast
from enum import Enum
from pathlib import Path
from llama_cpp import Llama
from dotenv import load_dotenv
from numpy.typing import NDArray
from typing import List, Tuple, Callable
from dataclasses import is_dataclass, asdict
from .models import CapabilityProfile, TaskDescription
# from prompt_toolkit import prompt
# # from transformers import pipeline
# # from inspect import signature, Parameter

_LLAMA_MODEL = None
_LLAMA_REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
_LLAMA_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

def _get_llama():
    """
    Lazy-initialize a llama.cpp Llama instance with full GPU offload.
    On an RTX 4080 Super (16 GB), a Q4_K_M 8B model (~4.9 GB) fits entirely
    in VRAM with n_gpu_layers=-1, which gives the fastest per-call latency
    without HTTP or process overhead. The singleton pattern avoids reloading
    the weights on every suitability evaluation.

    For context window, 4k-8k takes about 5-6 GB of VRAM, every +8k context
    after that adds about 1-2GB more VRAM.

    on a 4080 super
    40k = 12.7GB
    32k = 11.7GB
    16k = 9.6GB
    """
    global _LLAMA_MODEL
    if _LLAMA_MODEL is None:
        _LLAMA_MODEL = Llama.from_pretrained(
            repo_id=_LLAMA_REPO_ID,
            filename=_LLAMA_FILENAME,
            n_gpu_layers=-1,
            n_ctx=16384,
            n_batch=512,
            flash_attn=True,
            verbose=False,
        )
    return _LLAMA_MODEL

load_dotenv(dotenv_path=Path("C:\\Users\\owner\\Documents\\PhD\\TierLab\\VBTA - Original Commented\\.env"))

ScoreFn = Callable[[CapabilityProfile, TaskDescription], float]

def navigation_suitability(
        robot_mobility_type: str, 
        robot_size: Tuple[float, float, float], 
        robot_sensor_range: float, 
        task_constraints: List[str]) -> float:
    """
    Evaluates the suitability of a robot for navigating a task environment based on mobility type, size, and navigation constraints.
    
    Parameters:
        robot_mobility_type: The mobility type of the robot (e.g., "wheeled", "tracked", "legged", "aerial", "hovering", "climbing").
        robot_size: A tuple representing the robot's dimensions (length, width, height).
        robot_sensor_range: A float representing the effective sensor range.
        task_constraints: A list of navigation constraints for the task environment.
    
    Returns:
        A float score [0, 1] representing the suitability for navigation. Returns 0 if there is a critical mismatch that prevents navigation.
    """

    # Initialize the score
    score = 0.0

    # Define size thresholds for narrow spaces, low ceilings, etc.
    narrow_space_threshold = 2.0  # Width limit for narrow spaces
    low_ceiling_threshold = 2.0   # Height limit for low ceilings

    for constraint in task_constraints:
        
        # ---- SIZE ---- #
        # Constraint: Elevator access
        if constraint == "elevator":
            score += 1.0

        # Constraint: Narrow spaces
        elif constraint == "narrow spaces":
            if robot_size[1] <= narrow_space_threshold:
                score += 1.0
            else:
                return 0.0  

        # Constraint: Low ceilings
        elif constraint == "low ceilings":
            if robot_size[2] <= low_ceiling_threshold:
                score += 1.0
            else:
                return 0.0  
            
        # Constraint: Crowded environments
        elif constraint == "crowded":
            if robot_size[0] <= 1.0 and robot_size[1] <= 1.0:
                score += 1.0
            else:
                score += 0.0

        # ---- SENSOR ---- #
        # Constraint: Low visibility
        elif constraint == "low visibility":
            if robot_sensor_range >= 20:
                score += 1.0
            elif robot_sensor_range >= 15:
                score += 0.5
            else:
                score += 0.0

        # ---- MOBILITY TYPE ---- #
        # Constraint: Stairs
        elif constraint == "stairs":
            if robot_mobility_type in ["legged", "aerial"]:
                score += 1.0
            else:
                return 0.0  
            
        # Constraint: Smooth floors
        elif constraint == "smooth surfaces":
            if robot_mobility_type in ["wheeled", "tracked"]:
                score += 1.0

        # Constraint: Uneven floors
        elif constraint == "uneven floors":
            if robot_mobility_type in ["legged", "tracked"]:
                score += 1.0

        # Constraint: Slippery surfaces
        elif constraint == "slippery":
            if robot_mobility_type in ["wheeled", "aerial"]:
                score += 1.0
            else:
                return 0.0

        # Constraint: Windy conditions
        elif constraint == "windy":
            if robot_mobility_type == "aerial":
                return 0.0
            else:
                score += 1.0

    # Final suitability score between [0, 1] (0 if any constraint returns 0)
    return max(0.0, score / max(len(task_constraints), 1.0))

def evaluate_suitability_balanced(
        robots: list[CapabilityProfile], 
        tasks: list[TaskDescription], 
        *, map_size: int) -> np.ndarray:
    """
    Batch suitability score helper
    """
    M = np.zeros((len(robots), len(tasks)))
    for i, r in enumerate(robots):
        for j, t in enumerate(tasks):
            M[i, j] = _balanced_pair(r, t, map_size=map_size)
    return M

def _balanced_pair(robot: CapabilityProfile, task: TaskDescription, map_size: int) -> float:
    """
    Evaluates the suitability of a robot for a given task.
    A higher score indicates better suitability.
    Strict parameters:
    - Battery
    - Payload
    - Reach
    - Manipulators
    Everything else is loose based on what gets checked in the simulation!
    
    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.
    
    Returns:
        score: A float score representing the suitability of the robot for the task. A score of 0 indicates the robot cannot perform the task.
    """
    score = 0.0
    total_weight = 0.0

    weights = {
        "payload": 3.0,
        "manipulators": 4.0,
        "sensors": 3.0,
        "communication": 0.5,
        "safety": 1.0,
        "environmental": 1.0,
        "reach": 2.0,
        "sensor_range": 1.0,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "navigation": 2.0,
    }

    # ---- Payload ---- #
    total_weight += weights["payload"]
    if robot.payload_capacity < task.required_capabilities["payload"]:
        return 0.0
    else:
        score += weights["payload"]

    # ---- Reach ---- #
    total_weight += weights["reach"]
    if robot.reach < task.required_capabilities["reach"]:
        return 0.0
    else:
        score += weights["reach"]

    # ---- Manipulators ---- #
    if task.manipulators_needed:
        total_weight += weights["manipulators"]
        if not any(manip in robot.manipulators for manip in task.manipulators_needed):
            return 0.0
        manipulator_matched = sum(manip in robot.manipulators for manip in task.manipulators_needed)
        manipulator_score = manipulator_matched / len(task.manipulators_needed)
        score += weights["manipulators"] * manipulator_score
    
    # ---- Sensors ---- #
    if task.sensors_needed:
        total_weight += weights["sensors"]
        sensors_matched = sum(sensor in robot.sensors for sensor in task.sensors_needed)
        sensor_score = sensors_matched / len(task.sensors_needed)
        score += weights["sensors"] * sensor_score

    # ---- Communications ---- #
    if task.communication_requirements:
        total_weight += weights["communication"]
        matched_communications = sum(comm in robot.communication_protocols for comm in task.communication_requirements)
        communication_score = matched_communications / len(task.communication_requirements)
        score += weights["communication"] * communication_score

    # ---- Safety ---- #
    if task.safety_protocols:
        total_weight += weights["safety"]
        matched_safety = sum(safe in robot.safety_features for safe in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        score += weights["safety"] * safety_score

    # ---- Environmental ---- #
    if task.environmental_conditions:
        total_weight += weights["environmental"]
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        score += weights["environmental"] * environmental_score

    # ---- Navigation ---- #
    if task.navigation_constraints:
        total_weight += weights["navigation"]
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, robot.sensor_range, task.navigation_constraints)
        score += weights["navigation"] * navigation_score

    # ---- Sensor range ---- #
    total_weight += weights["sensor_range"]
    if robot.current_path and len(robot.current_path) > 0:
        distance_to_task = len(robot.current_path) - 1
    else:
        # manhattan distance fallback if path does not exist yet
        distance_to_task = (task.location[0] - robot.location[0]) + (task.location[1] - robot.location[1])
    if robot.sensor_range >= distance_to_task:
        sensor_range_score = 1.0
    elif robot.sensor_range >= distance_to_task / 2.0:
        sensor_range_score = 0.5
    else:
        sensor_range_score = 0.0
    score += weights["sensor_range"] * sensor_range_score

    # ---- Battery ---- #
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = 1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete) else 0.5
    score += weights["battery_duration"] * battery_score

    # Special functions 
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "utilities": ["precise alignment", "balance control"],
        "debris": ["balance control", "object recognition"],
        "delivery": ["object recognition"],
        "assembly": ["object recognition", "precise alignment"],
        "excavate": [ "terrain leveling", "precise alignment"],
        "item elevation": ["precise alignment", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # ---- Processing power ---- #
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 7.0:
            proc_score = 1.0
        elif robot.processing_power >= 5.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 4.0:
            proc_score = 1.0
        else:
            proc_score = 0.5
    score += weights["processing_power"] * proc_score

    # Normalize 
    if total_weight > 0:
        final_score = score / total_weight
        # ---- Proximity ---- #
        if map_size and map_size > 0:
            final_score *= 1.0 / (1.0 + distance_to_task / map_size)   # in (0, 1]
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_loose(
        robots: list[CapabilityProfile],
        tasks: list[TaskDescription],
        *, map_size: int) -> np.ndarray:
    """
    Batch suitability score helper
    """
    M = np.zeros((len(robots), len(tasks)))
    for i, r in enumerate(robots):
        for j, t in enumerate(tasks):
            M[i, j] = _loose_pair(r, t, map_size=map_size)
    return M

def _loose_pair(robot: CapabilityProfile, task: TaskDescription, map_size: int) -> float:
    """
    Evaluates the suitability of a robot for a given task.
    A higher score indicates better suitability.
    
    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.
    
    Returns:
        score: A float score representing the suitability of the robot for the task. A score of 0 indicates the robot cannot perform the task.
    """
    score = 0.0
    total_weight = 0.0  # for normalization

    weights = {
        "payload": 3.0,
        "manipulators": 4.0,
        "sensors": 3.0,
        "communication": 0.5,
        "safety": 1.0,
        "environmental": 1.0,
        "reach": 2.0,
        "sensor_range": 1.0,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "navigation": 2.0,
    }
    
    # ---- Payload ---- #
    total_weight += weights["payload"]
    if robot.payload_capacity < task.required_capabilities["payload"]:
        score += 0.0
    else:
        score += weights["payload"]

    # ---- Reach ---- #
    total_weight += weights["reach"]
    if robot.reach < task.required_capabilities["reach"]:
        score += 0.0
    else:
        score += weights["reach"]

    # ---- Manipulators ---- #
    if task.manipulators_needed:
        total_weight += weights["manipulators"]
        manipulator_matched = sum(manip in robot.manipulators for manip in task.manipulators_needed)
        manipulator_score = manipulator_matched / len(task.manipulators_needed)
        score += weights["manipulators"] * manipulator_score

    # ---- Sensors ---- #
    if task.sensors_needed:
        total_weight += weights["sensors"]
        sensors_matched = sum(sensor in robot.sensors for sensor in task.sensors_needed)
        sensor_score = sensors_matched / len(task.sensors_needed)
        score += weights["sensors"] * sensor_score

    # ---- Communications ---- #
    if task.communication_requirements:
        total_weight += weights["communication"]
        matched_communications = sum(comm in robot.communication_protocols for comm in task.communication_requirements)
        communication_score = matched_communications / len(task.communication_requirements)
        score += weights["communication"] * communication_score

    # ---- Safety ---- #
    if task.safety_protocols:
        total_weight += weights["safety"]
        matched_safety = sum(safe in robot.safety_features for safe in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        score += weights["safety"] * safety_score

    # ---- Environmental ---- #
    if task.environmental_conditions:
        total_weight += weights["environmental"]
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        score += weights["environmental"] * environmental_score

    # ---- Navigation ---- #
    if task.navigation_constraints:
        total_weight += weights["navigation"]
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, robot.sensor_range, task.navigation_constraints)
        score += weights["navigation"] * navigation_score

    # ---- Sensor range ---- #
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    if robot.sensor_range >= distance_to_task:
        sensor_range_score = 1.0
    elif robot.sensor_range >= distance_to_task / 2.0:
        sensor_range_score = 0.5
    else:
        sensor_range_score = 0.0
    score += weights["sensor_range"] * sensor_range_score

    # ---- Battery ---- #
    total_weight += weights["battery_duration"]
    battery_score = 1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete) else 0.5
    score += weights["battery_duration"] * battery_score

    # Special functions 
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "utilities": ["precise alignment", "balance control"],
        "debris": ["balance control", "object recognition"],
        "delivery": ["object recognition"],
        "assembly": ["object recognition", "precise alignment"],
        "excavate": [ "terrain leveling", "precise alignment"],
        "item elevation": ["precise alignment", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # ---- Processing power ---- #
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 7.0:
            proc_score = 1.0
        elif robot.processing_power >= 5.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 4.0:
            proc_score = 1.0
        else:
            proc_score = 0.5
    score += weights["processing_power"] * proc_score

    # Normalize
    if total_weight > 0:
        final_score = score / total_weight
        # ---- Proximity ---- #
        if map_size and map_size > 0:
            final_score *= 1.0 / (1.0 + distance_to_task / map_size)   # in (0, 1]
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_strict(
        robots: list[CapabilityProfile],
        tasks: list[TaskDescription],
        *, map_size: int) -> np.ndarray:
    """
    Batch suitability score helper
    """
    M = np.zeros((len(robots), len(tasks)))
    for i, r in enumerate(robots):
        for j, t in enumerate(tasks):
            M[i, j] = _strict_pair(r, t, map_size=map_size)
    return M

def _strict_pair(robot: CapabilityProfile, task: TaskDescription, map_size: int) -> float:
    """
    Evaluates the suitability of a robot for a given task.
    A higher score indicates better suitability.
    
    Parameters:
        robot: The CapabilityProfile of the robot.
        task: The TaskDescription of the task.
    
    Returns:
        score: A float score representing the suitability of the robot for the task. A score of 0 indicates the robot cannot perform the task.
    """
    score = 0.0
    total_weight = 0.0

    weights = {
        "payload": 3.0,
        "manipulators": 4.0,
        "sensors": 3.0,
        "communication": 0.5,
        "safety": 1.0,
        "environmental": 1.0,
        "reach": 2.0,
        "sensor_range": 1.0,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "navigation": 2.0,
    }

    # ---- Payload ---- #
    total_weight += weights["payload"]
    if robot.payload_capacity < task.required_capabilities["payload"]:
        return 0.0
    else:
        score += weights["payload"]

    # ---- Reach ---- #
    total_weight += weights["reach"]
    if robot.reach < task.required_capabilities["reach"]:
        return 0.0
    else:
        score += weights["reach"]

    # ---- Manipulators ---- #
    if task.manipulators_needed:
        total_weight += weights["manipulators"]
        if not any(manip in robot.manipulators for manip in task.manipulators_needed):
            return 0.0
        manipulator_matched = sum(manip in robot.manipulators for manip in task.manipulators_needed)
        manipulator_score = manipulator_matched / len(task.manipulators_needed)
        score += weights["manipulators"] * manipulator_score

    # ---- Sensors ---- #
    if task.sensors_needed:
        total_weight += weights["sensors"]
        sensor_matched = sum(sensor in robot.sensors for sensor in task.sensors_needed)
        sensor_score = sensor_matched / len(task.sensors_needed)
        score += weights["sensors"] * sensor_score

    # ---- Communications ---- #
    if task.communication_requirements:
        total_weight += weights["communication"]
        if not any(comm in robot.communication_protocols for comm in task.communication_requirements):
            return 0.0
        matched_communications = sum(comm in robot.communication_protocols for comm in task.communication_requirements)
        communication_score = matched_communications / len(task.communication_requirements)
        score += weights["communication"] * communication_score

    # ---- Safety ---- #
    if task.safety_protocols:
        total_weight += weights["safety"]
        if not any(safe in robot.safety_features for safe in task.safety_protocols):
            return 0.0
        matched_safety = sum(safe in robot.safety_features for safe in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        score += weights["safety"] * safety_score

    # ---- Environmental ---- #
    if task.environmental_conditions:
        total_weight += weights["environmental"]
        if not any(condition in robot.environmental_resistance for condition in task.environmental_conditions):
            return 0.0
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        score += weights["environmental"] * environmental_score

    # ---- Navigation ---- #
    if task.navigation_constraints:
        total_weight += weights["navigation"]
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, robot.sensor_range, task.navigation_constraints)
        if navigation_score == 0:
            return 0.0
        score += weights["navigation"] * navigation_score

    # ---- Sensor range ---- #
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    if robot.sensor_range >= distance_to_task:
        sensor_range_score = 1.0
    elif robot.sensor_range >= distance_to_task / 2.0:
        sensor_range_score = 0.5
    else:
        sensor_range_score = 0.0
    score += weights["sensor_range"] * sensor_range_score

    # ---- Battery ---- #
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = 1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete) else 0.5
    score += weights["battery_duration"] * battery_score

    # ---- Special functions ---- #
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "utilities": ["precise alignment", "balance control"],
        "debris": ["balance control", "object recognition"],
        "delivery": ["object recognition"],
        "assembly": ["object recognition", "precise alignment"],
        "excavate": [ "terrain leveling", "precise alignment"],
        "item elevation": ["precise alignment", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # ---- Processing power ---- #
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 7.0:
            proc_score = 1.0
        elif robot.processing_power >= 5.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 4.0:
            proc_score = 1.0
        else:
            proc_score = 0.5
    score += weights["processing_power"] * proc_score

    # Normalize
    if total_weight > 0:
        final_score = score / total_weight
        # ---- Proximity ---- #
        if map_size and map_size > 0:
            final_score *= 1.0 / (1.0 + distance_to_task / map_size)   # in (0, 1]
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def calculate_total_suitability(assignment: List[Tuple[int, int]], suitability_matrix: List[List[float]]) -> float:
    """
    Calculates the total suitability score for a given assignment. (lookup table)
    
    Parameters:
        assignment: A list of (robot, task) pairs representing the assignment.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        total_suitability: The total suitability score for the assignment.
    """
    total_suitability = 0.0
    
    # Sum the suitability ratings for each robot-task pair in the assignment
    for robot, task in assignment:
        total_suitability += suitability_matrix[robot][task]
    
    return total_suitability

def check_zero_suitability(assignment: List[Tuple[int, int]], suitability_matrix: List[List[float]]) -> bool:
    """
    Checks if any robot-task pair in the assignment has a suitability rating of 0.
    
    Parameters:
        assignment: A list of (robot, task) pairs representing the assignment.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        Bool: True if any robot-task pair in the assignment has a suitability of 0, otherwise False.
    """
    for robot, task in assignment:
        if suitability_matrix[robot][task] == 0:
            return True  # Found a zero suitability rating
    
    return False  # No zero suitability ratings found

def calculate_suitability_matrix(
    robots: List, 
    tasks: List, 
    scorer: Callable[..., object],
    map_size: int
) -> NDArray[np.float64]:
    """
    Compute suitability matrix using either:
      • a rules based scorer (Balanced, Loose, Strict)
      • an LLM based scorer (Llama, Mixtral)
    """
    M = scorer(robots, tasks, map_size)
    return np.clip(M, 0.0, 1.0)

def _to_jsonable(x):
    """Recursively convert to JSON-safe Python objects."""
    if is_dataclass(x):
        x = asdict(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, Enum):
        return x.name
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x

def _extract_output_matrix_text(raw: str) -> str | None:
    """
    Find the first bracketed matrix *after* the token 'OUTPUT'.
    Returns the substring like "[ [...], [...], ... ]" or None.
    """
    if not isinstance(raw, str):
        return None

    # Find OUTPUT token (case-insensitive, as a whole word)
    m = re.search(r'\bOUTPUT\b', raw, flags=re.IGNORECASE)
    start = m.end() if m else 0

    # Find the first '[' after OUTPUT
    first_lb = raw.find('[', start)
    if first_lb == -1:
        return None

    # Balanced bracket scan to find the matching closing ']'
    depth = 0
    for i in range(first_lb, len(raw)):
        ch = raw[i]
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return raw[first_lb:i+1]

    return None

def _clean_for_json(s: str) -> str:
    """
    Make the matrix string JSON-friendly:
    - replace single quotes with double
    - drop trailing commas before ] or }
    - replace NaN/Infinity with 0
    - strip stray semicolons
    """
    s = s.strip()
    # If it looks like Python-lists with single quotes, normalize
    if "'" in s and '"' not in s:
        s = s.replace("'", '"')

    # Remove trailing commas ", ]" -> "]"
    s = re.sub(r',\s*]', ']', s)
    s = re.sub(r',\s*}', '}', s)

    # Replace non-JSON tokens
    s = re.sub(r'\bNaN\b', '0', s, flags=re.IGNORECASE)
    s = re.sub(r'\bInfinity\b', '0', s, flags=re.IGNORECASE)
    s = s.replace(';', ',')

    return s

def _parse_output_matrix(raw_text: str, nR: int, nT: int) -> Tuple[np.ndarray | None, bool]:
    """
    Parse the OUTPUT matrix into an (nR, nT) float array.
    Returns None if impossible to parse.
    """
    dir_path = os.path.join('hvbta', 'io', 'logging')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(dir_path, "logging_", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), encoding='utf-8', level=logging.DEBUG)

    parse_failed = True
    block = _extract_output_matrix_text(raw_text)
    if not block:
        logger.debug("RAW TEXT NOT EXTRACTED TO MATRIX \n\n\n")
        logger.debug(raw_text)
        return None, parse_failed

    cleaned = _clean_for_json(block)

    data = None
    # Try JSON first
    try:
        data = json.loads(cleaned)
    except Exception:
        # Fall back to Python literal eval
        try:
            data = ast.literal_eval(cleaned)
        except Exception:
            logger.debug("JSON AND AST EVAL BOTH FAILED \n\n\n")
            logger.debug(cleaned)
            return None, parse_failed

    try:
        arr = np.array(data, dtype=float)
    except Exception:
        logger.debug("NP ARRAY FAILED, DATA MAY CONTAIN NON NUMERIC VALUES \n\n\n")
        logger.debug(data)
        return None, parse_failed

    # Fix common shape slips
    if arr.ndim == 1:
        # If one row was returned but we expect one robot, accept as (1, nT)
        if nR == 1 and arr.size == nT:
            arr = arr.reshape(1, nT)
        # If one column returned but expect one task, accept as (nR, 1)
        elif nT == 1 and arr.size == nR:
            arr = arr.reshape(nR, 1)

    # If shape is close, pad/truncate (optional; or reject)
    if arr.ndim == 2:
        r, c = arr.shape
        # pad rows
        if r < nR:
            pad = np.zeros((nR - r, min(c, nT)), dtype=float)
            arr = np.vstack([arr[:, :min(c, nT)], pad])
            r, c = arr.shape
        # pad cols
        if c < nT:
            pad = np.zeros((min(r, nR), nT - c), dtype=float)
            arr = np.hstack([arr[:min(r, nR), :], pad])

        # finally enforce exact size
        arr = arr[:nR, :nT]
    else:
        logger.debug("NP ARRAY INCORRECT DIMENSIONS OR UNEXPECTED SIZE\n")
        logger.debug(f"EXPECTED SHAPE: {nR} by {nT}\n")
        logger.debug(f"ACTUAL SHAPE: {arr.shape}\n")
        
        return None, parse_failed

    # clip to [0,1]
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape != (nR, nT):
        logger.debug("NP ARRAY INCORRECT DIMENSIONS OR UNEXPECTED SIZE\n")
        logger.debug(f"EXPECTED SHAPE: {nR} by {nT}\n")
        logger.debug(f"ACTUAL SHAPE: {arr.shape}\n")
        return None, parse_failed
    parse_failed = False
    return arr, parse_failed

def _parse_output_matrix_bypass(raw_text: str) -> np.ndarray | None:
    """Parse LLM output to get tuple of 3 lists to represent assignments
    Tuple[List[Tuple[int, int]], List[int], List[int]]
    Each list represents:
        - assigned pairs of robots and tasks
        - unassigned robots
        - unassigned tasks
    all represented with their robot and task IDs"""
    # TODO: FINISH THIS FUNCTION, DIRECT ASSIGNMENT WITH LLM
    pass


def build_name_only_prompt(robots_json: str, tasks_json: str) -> str:
    return f"""
You score robot–task suitability using ONLY each task's name and a short description (if provided).
Infer typical requirements from the task name (and description), then score how suitable each robot is.
Do not include code fences, explanations, or any text outside of the score matrix. Do not include comments.
The matrix must be exactly N rows (robots) by M columns (tasks).

INPUT (JSON)
Robots:
{robots_json}

Tasks (only names + optional nl_description):
{tasks_json}

Scoring principles (follow closely)
- No hard-fail zeros unless the robot is blatantly incapable for the *typical* demands implied by the task name.
- Infer likely needs from the name/description, e.g.:
  · "item elevation" → hoisting/cable hoist, long reach, stable placement.
  · "excavate" → hydraulic bucket, traction (tracked), high payload, rugged sensors.
  · "lay bricks" → dispenser/gripper, precise alignment, moderate reach, stable placement.
  · "scaffold" → work near frames at height, drill/gripper, precise alignment, narrow access.
  · "delivery" → mobility/endurance, small payload capacity, navigation through corridors.
  · "utilities" → gripper operation of valves/switches, precise manipulation; or bucket variant for bulk.
  · "debris" → gripper variant (pick-and-place) vs bucket variant (bulk scooping).
- Consider: manipulators, mobility type, payload capacity, reach, sensor suite, comms/safety, autonomy level, processing power, battery/endurance.
- Use soft weights; if a capability is somewhat relevant but not mandatory, give partial credit rather than 0.
- Calibrate to avoid trivial 0.0/1.0: clamp to [0.05, 0.99] unless obviously perfect (1.0) or clearly mismatched (~0.0).

Output format (STRICT)
Return ONLY:
OUTPUT
[
  [r1_t1, r1_t2, ..., r1_tM],
  [r2_t1, r2_t2, ..., r2_tM],
  ...
]
with floats in [0,1]. No extra text after the matrix.
"""

# --- Minimal serializers so the prompt stays short & cheap ---
def _robot_min_view(r: CapabilityProfile) -> dict:
    return dict(
        robot_id=r.robot_id,
        mobility_type=r.mobility_type,
        manipulators=r.manipulators,
        payload_capacity=r.payload_capacity,
        reach=r.reach,
        sensors=r.sensors,
        sensor_range=r.sensor_range,
        communication_protocols=r.communication_protocols,
        safety_features=r.safety_features,
        special_functions=r.special_functions,
        processing_power=r.processing_power,
        autonomy_level=r.autonomy_level,
        battery_life=r.battery_life,
    )

def _task_name_view(t: TaskDescription) -> dict:
    # Try to read nl_description if present; otherwise just pass the task_type
    d = {
        "task_id": t.task_id,
        "task_type": t.task_type,
    }
    # using getattr for the safe access with defaults
    desc = getattr(t, "nl_description", None) or getattr(t, "strict_profile_name", None)
    if isinstance(desc, str):
        d["nl_description"] = desc
    return d

def evaluate_suitability_from_names_with_llm(robots: List[CapabilityProfile], tasks: List[TaskDescription], model=_LLAMA_FILENAME) -> np.ndarray:
    """
    Evaluate suitability of robots for tasks using a local GGUF-quantized Llama model
    via llama.cpp with full GPU offload. Returns an (R, T) float array in [0,1].
    Parameters:
        robots: List of CapabilityProfile objects.
        tasks: List of TaskDescription objects.
        model: The GGUF filename to load (unused at call time; model is held in the
               module-level singleton initialized by _get_llama()).
    Returns:
        M: An (R, T) numpy array of float suitability scores in [0,1].
    """
    evaluate_suitability_from_names_with_llm._is_llm_batch = True
    llm = _get_llama()

    R = [_to_jsonable(_robot_min_view(r)) for r in robots]
    T = [_to_jsonable(_task_name_view(t)) for t in tasks]

    text_prompt = build_name_only_prompt(
        robots_json=json.dumps(R, ensure_ascii=False),
        tasks_json=json.dumps(T, ensure_ascii=False)
    )

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a careful assistant that outputs only the requested format."},
            {"role": "user", "content": text_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    content = resp["choices"][0]["message"]["content"]

    M, M.parse_failed = _parse_output_matrix(content, nR=len(robots), nT=len(tasks))
    if M is None:
        M = np.full((len(robots), len(tasks)), 0.0, dtype=float)
    return M

def bypass_suitability_from_names_with_llm(robots, tasks, model=_LLAMA_FILENAME) -> np.ndarray:
    """
    One-shot assignment variant using the local GGUF-quantized Llama model via
    llama.cpp with full GPU offload. Returns an (R, T) float array in [0,1].
    Parameters:
        robots: List of CapabilityProfile objects.
        tasks: List of TaskDescription objects.
        model: The GGUF filename (held in the module-level singleton).
    Returns:
        M: An (R, T) numpy array of float suitability scores in [0,1].
    """
    evaluate_suitability_from_names_with_llm._is_llm_batch = True
    llm = _get_llama()

    R = [_to_jsonable(_robot_min_view(r)) for r in robots]
    T = [_to_jsonable(_task_name_view(t)) for t in tasks]

    example_prompt = (
        "Assign each task to the single most suitable robot.\n"
        "Reply with ONLY a JSON object, no explanation:\n"
        '{"assignments": [{"task_id": "<id>", "robot_id": "<id>"}, ...]}\n\n'
        "Robots:\n[{\"id\": \"r0\", \"name\": \"HeavyLifter\"}, {\"id\": \"r1\", \"name\": \"SwiftDrone\"}]\n\n"
        "Tasks:\n[{\"id\": \"t0\", \"name\": \"lift_crate\"}, {\"id\": \"t1\", \"name\": \"aerial_survey\"}]"
    )
    example_answer = '{"assignments": [{"task_id": "t0", "robot_id": "r0"}, {"task_id": "t1", "robot_id": "r1"}]}'

    real_prompt = (
        "Assign each task to the single most suitable robot.\n"
        "Reply with ONLY a JSON object, no explanation:\n"
        '{"assignments": [{"task_id": "<id>", "robot_id": "<id>"}, ...]}\n\n'
        f"Robots:\n{json.dumps(R, ensure_ascii=False)}\n\n"
        f"Tasks:\n{json.dumps(T, ensure_ascii=False)}"
    )

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a robot task scheduler. Reply with a single JSON object and nothing else."},
            {"role": "user",      "content": example_prompt},
            {"role": "assistant", "content": example_answer},
            {"role": "user",      "content": real_prompt},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    content = resp["choices"][0]["message"]["content"]

    M, M.parse_failed = _parse_output_matrix(content, nR=len(robots), nT=len(tasks))
    if M is None:
        M = np.full((len(robots), len(tasks)), 0.0, dtype=float)
    return M