import json, re, ast
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Union
from transformers import pipeline
from llama_cpp import Llama
# from prompt_toolkit import prompt
from .models import CapabilityProfile, TaskDescription
# from inspect import signature, Parameter
from enum import Enum
from dataclasses import is_dataclass, asdict

_LLAMA_MODEL = None
_LLAMA_REPO_ID = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
_LLAMA_FILENAME = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

MAP_SCALE = 1.0

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

def evaluate_suitability_new(robot: CapabilityProfile, task: TaskDescription) -> float:
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
        "proximity": 1.0,
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
        proximity_factor = weights["proximity"] / (1E-5 + distance_to_task / MAP_SCALE)
        final_score *= proximity_factor
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_loose(robot: CapabilityProfile, task: TaskDescription) -> float:
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
        "proximity": 1.0,
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
        proximity_factor = weights["proximity"] / (1E-5 + distance_to_task / MAP_SCALE)
        final_score *= proximity_factor
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_strict(robot: CapabilityProfile, task: TaskDescription) -> float:
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
        "proximity": 1.0,
        # "autonomy_match": 0.5,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        # "adaptability": 0.5,
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
        proximity_factor = weights["proximity"] / (1E-5 + distance_to_task / MAP_SCALE)
        final_score *= proximity_factor
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

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

def _parse_output_matrix(raw_text: str, nR: int, nT: int) -> np.ndarray | None:
    """
    Parse the OUTPUT matrix into an (nR, nT) float array.
    Returns None if impossible to parse.
    """
    block = _extract_output_matrix_text(raw_text)
    if not block:
        return None

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
            return None

    try:
        arr = np.array(data, dtype=float)
    except Exception:
        return None

    # Fix common shape slips
    if arr.ndim == 1:
        # If one row was returned but we expect one robot, accept as (1, nT)
        if nR == 1 and arr.size == nT:
            arr = arr.reshape(1, nT)
        # If one column returned but expect one task, accept as (nR, 1)
        elif nT == 1 and arr.size == nR:
            arr = arr.reshape(nR, 1)

    # If shape is close, pad/truncate (optional; or you can reject)
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
        return None

    # clip to [0,1]
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape != (nR, nT):
        return None
    return arr

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

    M = _parse_output_matrix(content, nR=len(robots), nT=len(tasks))
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

    M = _parse_output_matrix(content, nR=len(robots), nT=len(tasks))
    if M is None:
        M = np.full((len(robots), len(tasks)), 0.0, dtype=float)
    return M


    # # Parse the OUTPUT matrix
    # m = re.search(r"OUTPUT\s*\[\s*(.*)\s*\]\s*$", content, re.S)
    # if not m:
    #     raise ValueError("LLM did not return a valid OUTPUT matrix.")
    # body = "[" + m.group(1).strip() + "]"
    # M = np.array(json.loads(body), dtype=float)

    # # Safety shape check
    # if M.shape != (len(R), len(T)):
    #     raise ValueError(f"Matrix shape {M.shape} does not match robots={len(R)} tasks={len(T)}.")

    # return M


def make_pairwise_from_batch(batch_fn, robots_all, tasks_all):
    """
    Wrap a batch scorer f(robots, tasks)->matrix into a pairwise scorer g(robot, task)->float.
    Caches the matrix so we only call the LLM once per run.
    Pre-computes the matrix immediately to avoid repeated checks.
    """
    # Pre-compute the matrix immediately instead of lazy evaluation
    M = batch_fn(robots_all, tasks_all)
    if not isinstance(M, np.ndarray):
        M = np.array(M, dtype=float)
    else:
        M = M.astype(float)
    
    # Pre-build index dictionaries
    r_index = {r.robot_id: i for i, r in enumerate(robots_all)}
    t_index = {t.task_id: j for j, t in enumerate(tasks_all)}

    def g(robot, task):
        return M[r_index[robot.robot_id], t_index[task.task_id]]
    
    # Attach the matrix for direct access if needed
    g._matrix = M
    g._r_index = r_index
    g._t_index = t_index
    
    return g


def calculate_total_suitability(assignment: List[Tuple[int, int]], suitability_matrix: List[List[float]]) -> float:
    """
    Calculates the total suitability score for a given assignment.
    
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

# def calculate_suitability_matrix(robots: List[CapabilityProfile], tasks: List[TaskDescription], scorer: ScoreFn) -> np.ndarray:
#     """
#     Calculates the suitability matrix for the given robots and tasks.
    
#     Parameters:
#         robots: List of robot profiles.
#         tasks: List of task descriptions.
#         suitability_method: The name of the suitability evaluation function.
    
#     Returns:
#         suitability_matrix: A 2D numpy array representing the suitability scores of each robot-task pair.
#     """
#     suitability_matrix = np.zeros((len(robots), len(tasks)), dtype=float)

#     # Evaluate suitability of each robot for each task
#     for i, robot in enumerate(robots):
#         for j, task in enumerate(tasks):
#             # suitability_score = globals()[suitability_method](robot, task)
#             # suitability_matrix[i][j] = suitability_score
#             suitability_matrix[i, j] = scorer(robot, task)
            
#     return suitability_matrix

def calculate_suitability_matrix(
    robots: List, 
    tasks: List, 
    scorer: Callable[..., object]
) -> NDArray[np.float64]:
    """
    Compute suitability matrix using either:
      • a batch scorer: scorer(robots, tasks) -> ndarray[float] (R x T)
      • a pairwise scorer: scorer(robot, task) -> float

    The function auto-detects which kind you passed by first attempting a batch call.
    If that fails or doesn't return the right shape, it falls back to pairwise.
    """
    R, T = len(robots), len(tasks)
    expected_shape = (R, T)

    # --- Try batch mode first (this is ideal for LLM-based scoring) ---
    try:
        maybe_matrix = scorer(robots, tasks)  # if scorer is batch, this should work
        if isinstance(maybe_matrix, np.ndarray) and maybe_matrix.shape == expected_shape:
            M = maybe_matrix.astype(float, copy=False)
            return np.clip(M, 0.0, 1.0)
    except Exception:
        # Not a batch scorer (or it failed) → fall back to pairwise
        pass

    # --- Pairwise fallback ---
    M = np.zeros(expected_shape, dtype=float)
    for i, r in enumerate(robots):
        for j, t in enumerate(tasks):
            try:
                M[i, j] = float(scorer(r, t))
            except Exception:
                M[i, j] = 0.0
    return np.clip(M, 0.0, 1.0)


def calculate_jains_index(scores: List[float]) -> float:
    """
    Calculate Jain's Fairness Index for a list of allocation scores.
    
    Jain's Index formula: J = (sum(x_i))^2 / (n * sum(x_i^2))
    
    Properties:
    - Returns 1.0 for perfectly fair allocations (all scores equal)
    - Returns 1/n for maximally unfair allocations (one agent gets everything)
    - Range: [1/n, 1]
    
    Parameters:
        scores: List of individual agent allocation scores (suitability scores).
                Only includes assigned agents (unassigned agents are excluded).
    
    Returns:
        Jain's fairness index value in range [0, 1], or 1.0 if empty/all zeros.
    """
    if not scores or len(scores) == 0:
        return 1.0  # No allocations = trivially fair
    
    n = len(scores)
    sum_scores = sum(scores)
    sum_sq_scores = sum(s ** 2 for s in scores)
    
    if sum_sq_scores == 0:
        return 1.0  # All zeros = perfectly "fair" (everyone got equally nothing)
    
    return (sum_scores ** 2) / (n * sum_sq_scores)


def calculate_threshold_metrics(scores: List[float]) -> dict:
    """
    Calculate threshold fairness metrics for Approval voting analysis.
    
    Uses two thresholds:
    - GOOD_ENOUGH (0.5): Minimum acceptable suitability
    - GOOD (0.7): Desirable suitability level
    
    Parameters:
        scores: List of individual agent allocation scores.
    
    Returns:
        Dictionary with:
        - below_ge_frac: Fraction of agents below "good enough" threshold (0.5)
        - below_good_frac: Fraction of agents below "good" threshold (0.7)
        - deficit_all_ge: Mean deficit from 0.5 threshold, averaged over ALL agents
        - deficit_below_ge: Mean deficit from 0.5 threshold, averaged over those BELOW only
        - deficit_all_good: Mean deficit from 0.7 threshold, averaged over ALL agents  
        - deficit_below_good: Mean deficit from 0.7 threshold, averaged over those BELOW only
    """
    THRESHOLD_GOOD_ENOUGH = 0.5
    THRESHOLD_GOOD = 0.7
    
    if not scores or len(scores) == 0:
        return {
            "below_ge_frac": 0.0,
            "below_good_frac": 0.0,
            "deficit_all_ge": 0.0,
            "deficit_below_ge": 0.0,
            "deficit_all_good": 0.0,
            "deficit_below_good": 0.0
        }
    
    n = len(scores)
    
    # Good Enough threshold (0.5)
    below_ge = [s for s in scores if s < THRESHOLD_GOOD_ENOUGH]
    below_ge_count = len(below_ge)
    below_ge_frac = below_ge_count / n
    deficit_all_ge = sum(max(0, THRESHOLD_GOOD_ENOUGH - s) for s in scores) / n
    deficit_below_ge = sum(max(0, THRESHOLD_GOOD_ENOUGH - s) for s in below_ge) / below_ge_count if below_ge_count > 0 else 0.0
    
    # Good threshold (0.7)
    below_good = [s for s in scores if s < THRESHOLD_GOOD]
    below_good_count = len(below_good)
    below_good_frac = below_good_count / n
    deficit_all_good = sum(max(0, THRESHOLD_GOOD - s) for s in scores) / n
    deficit_below_good = sum(max(0, THRESHOLD_GOOD - s) for s in below_good) / below_good_count if below_good_count > 0 else 0.0
    
    return {
        "below_ge_frac": below_ge_frac,
        "below_good_frac": below_good_frac,
        "deficit_all_ge": deficit_all_ge,
        "deficit_below_ge": deficit_below_ge,
        "deficit_all_good": deficit_all_good,
        "deficit_below_good": deficit_below_good
    }


def calculate_inequality_metrics(scores: List[float]) -> dict:
    """
    Calculate min-max and inequality fairness metrics for Majority Judgment analysis.
    
    Uses O(n log n) Gini coefficient calculation via sorted cumulative sum.
    
    Parameters:
        scores: List of individual agent allocation scores.
    
    Returns:
        Dictionary with:
        - score_range: max - min score (spread between best and worst)
        - min_max_ratio: min/max ratio (closer to 1 = fairer)
        - gini: Gini coefficient [0=perfect equality, 1=max inequality]
        - cv: Coefficient of variation (std/mean, normalized dispersion)
    """
    if not scores or len(scores) == 0:
        return {
            "score_range": 0.0,
            "min_max_ratio": 1.0,
            "gini": 0.0,
            "cv": 0.0
        }
    
    n = len(scores)
    min_s = min(scores)
    max_s = max(scores)
    mean_s = sum(scores) / n
    
    # Range
    score_range = max_s - min_s
    
    # Min/Max ratio (1.0 if max is 0 to avoid division by zero)
    min_max_ratio = min_s / max_s if max_s > 0 else 1.0
    
    # Gini coefficient - O(n log n) via sorted cumulative sum
    # Formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    # where x_i are sorted in ascending order and i is 1-indexed
    sorted_scores = sorted(scores)
    cumsum = sum((i + 1) * s for i, s in enumerate(sorted_scores))
    total = sum(sorted_scores)
    if total > 0:
        gini = (2 * cumsum) / (n * total) - (n + 1) / n
        gini = max(0.0, gini)  # Ensure non-negative due to floating point
    else:
        gini = 0.0
    
    # Coefficient of variation
    if mean_s > 0:
        variance = sum((s - mean_s) ** 2 for s in scores) / n
        std_dev = variance ** 0.5
        cv = std_dev / mean_s
    else:
        cv = 0.0
    
    return {
        "score_range": score_range,
        "min_max_ratio": min_max_ratio,
        "gini": gini,
        "cv": cv
    }


def calculate_robustness_metrics(scores: List[float]) -> dict:
    """
    Calculate outlier robustness metrics for comparing median vs mean behavior.
    
    These metrics help demonstrate why Majority Judgment (median-based) is more
    robust than mean-based methods to extreme outlier scores.
    
    Parameters:
        scores: List of individual agent allocation scores.
    
    Returns:
        Dictionary with:
        - median: Median score (resistant to outliers)
        - mean: Mean score (sensitive to outliers)
        - med_mean_gap: |median - mean| (large gap indicates skewed distribution)
        - iqr: Interquartile range (Q3 - Q1, robust spread measure)
    """
    if not scores or len(scores) == 0:
        return {
            "median": 0.0,
            "mean": 0.0,
            "med_mean_gap": 0.0,
            "iqr": 0.0
        }
    
    n = len(scores)
    sorted_scores = sorted(scores)
    
    # Mean
    mean_s = sum(scores) / n
    
    # Median
    if n % 2 == 1:
        median_s = sorted_scores[n // 2]
    else:
        median_s = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    
    # Median-Mean gap
    med_mean_gap = abs(median_s - mean_s)
    
    # Interquartile Range (IQR = Q3 - Q1)
    def percentile(data, p):
        """Calculate percentile using linear interpolation."""
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])
    
    q1 = percentile(sorted_scores, 25)
    q3 = percentile(sorted_scores, 75)
    iqr = q3 - q1
    
    return {
        "median": median_s,
        "mean": mean_s,
        "med_mean_gap": med_mean_gap,
        "iqr": iqr
    }
