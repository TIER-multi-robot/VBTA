import os, json, re, ast
from dotenv import load_dotenv
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable, Union
from huggingface_hub import InferenceClient
# from prompt_toolkit import prompt
from .models import CapabilityProfile, TaskDescription
# from inspect import signature, Parameter
from enum import Enum
from dataclasses import is_dataclass, asdict

load_dotenv()

ScoreFn = Callable[[CapabilityProfile, TaskDescription], float]

def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)

def _split_tool_requirements(tools_needed):
    """
    Normalise TaskDescription.tools_needed into:
      - sensors: flat list of sensor ids
      - manipulator_groups: list of AND-requirement groups, each a list of tool ids

    Supports both the legacy flat list (['LiDAR', 'camera', ...]) and the newer
    [[sensors...], [manipulator group], ...] format used by strict profiles.
    """
    if not tools_needed:
        return [], []

    # Canonical case: [[sensor list], [manip group], ...]
    if isinstance(tools_needed, (list, tuple)) and tools_needed:
        first = tools_needed[0]
        if isinstance(first, (list, tuple)):
            sensors = [str(tool) for tool in first if tool]
            manip_groups = [
                [str(tool) for tool in group if tool]
                for group in tools_needed[1:]
                if isinstance(group, (list, tuple)) and group
            ]
            return sensors, manip_groups

        # Legacy case: flat list of sensor strings
        if all(isinstance(item, str) for item in tools_needed):
            return [str(item) for item in tools_needed if item], []

    # Fallback – treat anything string-ish as a sensor requirement
    return [str(item) for item in tools_needed if isinstance(item, str)], []

def navigation_suitability(robot_mobility_type: str, robot_size: Tuple[float, float, float], task_constraints: List[str]) -> float:
    """
    Evaluates the suitability of a robot for navigating a task environment based on mobility type, size, and navigation constraints.
    
    Parameters:
        robot_mobility_type: The mobility type of the robot (e.g., "wheeled", "tracked", "legged", "aerial", "hovering", "climbing").
        robot_size: A tuple representing the robot's dimensions (length, width, height).
        task_constraints: A list of navigation constraints for the task environment.
    
    Returns:
        A float score representing the suitability for navigation. Returns 0 if there is a critical mismatch that prevents navigation.
    """

    # Initialize the score
    score = 0.0

    # Define size thresholds for narrow spaces, low ceilings, etc.
    narrow_space_threshold = 2.0  # Width limit for narrow spaces
    low_ceiling_threshold = 2.0   # Height limit for low ceilings

    # Handle each task constraint based on mobility type
    for constraint in task_constraints:
        # Constraint: Elevator access
        if constraint == "elevator":
            score += 1.0

        # Constraint: Stairs
        elif constraint == "stairs":
            if robot_mobility_type in ["legged", "aerial", "hovering", "climbing"]:
                score += 1.0
            else:
                return 0.0

        # Constraint: Shelves
        elif constraint == "shelves":
            if robot_size[2] < low_ceiling_threshold or robot_mobility_type in ["aerial", "climbing", "hovering"]:
                score += 1.0  # Only smaller robots can access shelves effectively

        # Constraint: No loud noises allowed
        elif constraint == "no loud noises allowed":
            if robot_mobility_type in ["legged", "hovering"]:
                score += 1.0  # Quieter mobility types

        # Constraint: Narrow spaces
        elif constraint == "narrow spaces":
            if robot_size[1] <= narrow_space_threshold:
                score += 1.0
            else:
                return 0.0  # Larger robots cannot pass through narrow spaces

        # Constraint: Low ceilings
        elif constraint == "low ceilings":
            if robot_size[2] <= low_ceiling_threshold:
                score += 1.0
            else:
                return 0.0  # Tall robots cannot navigate in areas with low ceilings

        # Constraint: Uneven floors
        elif constraint == "uneven floors":
            if robot_mobility_type in ["tracked", "legged", "climbing", "hovering", "aerial"]:
                score += 1.0  # These types handle uneven floors well

        # Constraint: Low visibility
        elif constraint == "low visibility":
            if robot_mobility_type in ["wheeled", "tracked", "legged"]:
                score += 1.0  # Infrared or LiDAR-equipped robots are suitable
            else:
                score += 0.5

        # Constraint: Slippery surfaces
        elif constraint == "slippery":
            if robot_mobility_type in ["tracked", "hovering", "aerial"]:
                score += 1.0  # Hovering and tracked types handle slippery surfaces better
            elif robot_mobility_type in ["wheeled", "legged"]:
                return 0.0  # Wheeled and legged robots are unsuitable on slippery floors

        # Constraint: Crowded environments
        elif constraint == "crowded":
            if robot_size[0] <= 1.0 and robot_size[1] <= 1.0:
                score += 1.0  # Smaller robots are more suitable in crowded environments
            else:
                score += 0.5  # Larger robots get a lower score

        # Constraint: Loose debris
        elif constraint == "loose debris":
            if robot_mobility_type in ["aerial", "hovering", "tracked"]:
                score += 1.0  # Aerial and hovering robots handle debris better
            elif robot_mobility_type == "legged":
                return 0.0  # Legged robots are unsuitable
            else:
                score += 0.5

        # Constraint: No-fly zone
        elif constraint == "no-fly zone":
            if robot_mobility_type == "aerial":
                return 0.0  # Aerial robots cannot navigate in no-fly zones
            else:
                score += 0.5  # Other mobility types are unaffected

        # Constraint: Windy conditions
        elif constraint == "windy":
            if robot_mobility_type == "aerial":
                return 0.0  # Aerial robots struggle in windy conditions
            else:
                score += 0.5  # All other types are more stable in wind

        # Constraint: Dense obstructions (e.g., tree branches, hanging cables)
        elif constraint == "dense obstructions":
            if robot_mobility_type in ["aerial", "legged"]:
                return 0.0  # Aerial and legged robots are unsuitable in dense obstruction areas
            else:
                score += 0.5  # Other types may navigate dense areas on the ground

        # Constraint: Smooth floors
        elif constraint == "smooth surfaces":
            if robot_mobility_type == "climbing":
                return 0.0  # Climbing robots are less suited for smooth surfaces
            else:
                score += 0.5  # Other types may navigate dense areas on the ground

    # Final suitability score (0 if any constraint returns 0)
    return score if score > 0 else 0.0

def evaluate_suitability_new(robot: CapabilityProfile, task: TaskDescription) -> float:
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
        "autonomy_match": 0.5,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "adaptability": 0.5,
        "navigation": 2.0,
    }

    # Payload
    total_weight += weights["payload"]
    if any("payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        return 0.0
    else:
        score += weights["payload"]

    # Reach 
    total_weight += weights["reach"]
    if any("reach" in req and robot.reach < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        score += 0.0
    else:
        score += weights["reach"]

    # # Manipulators (tools_needed[1] is manipulators list)
    # # Check if there are any tool requirements for the task
    # if task.tools_needed:
    #     # A tool requirement exists, so add its weight to the total possible score
    #     total_weight += weights["manipulators"]
        
    #     group_scores = []
    #     # Iterate through each required tool group (the "AND" part)
    #     for required_group in task.tools_needed:
    #         # Skip any malformed or empty tool groups
    #         if not required_group:
    #             continue

    #         # 1. First, check for basic suitability (the "OR" part)
    #         #    Does the robot have at least one of the tools in this group?
    #         if not any(tool in robot.manipulators for tool in required_group):
    #             return 0.0  # If any AND group is not met, the robot is unsuitable.

    #         # 2. If suitable, then calculate the scalar score for this group
    #         #    This measures redundancy/completeness.
    #         matched_count = sum(tool in robot.manipulators for tool in required_group)
    #         group_score = matched_count / len(required_group)
    #         group_scores.append(group_score)

    #     # 3. If the loop completed, all groups were satisfied.
    #     #    Average the scores of all groups to get the final tool_score.
    #     if group_scores:
    #         tool_score = sum(group_scores) / len(group_scores)
    #         score += weights["manipulators"] * tool_score
    # # if task.tools_needed:
    # #     if ("cable hoist" in task.tools_needed[1] and "cable hoist" not in robot.manipulators and robot.mobility_type in ["hovering", "aerial"]):
    # #         task.tools_needed[1].remove("cable hoist")
    #     # matched_tools = sum(tool in robot.manipulators for tool in task.tools_needed[1])
    #     # tool_score = matched_tools / len(task.tools_needed[1])
    #     # if tool_score != 1:
    #     #     return 0.0
    #     # score += weights["manipulators"] * tool_score

    # # Sensors (tools_needed[0] is sensors list)
    # total_weight += weights["sensors"]
    # if task.tools_needed:
    #     matched_tools = sum(tool in robot.sensors for tool in task.tools_needed[0])
    #     tool_score = matched_tools / len(task.tools_needed[0])
    #     score += weights["sensors"] * tool_score

    sensor_requirements, manipulator_groups = _split_tool_requirements(task.tools_needed)

    # Manipulators – every AND-group must be satisfied by at least one manipulator the robot carries
    if manipulator_groups:
        total_weight += weights["manipulators"]
        group_scores = []
        robot_manips = set(robot.manipulators or [])
        for required_group in manipulator_groups:
            if not any(tool in robot_manips for tool in required_group):
                return 0.0  # hard requirement missing
            matched = sum(tool in robot_manips for tool in required_group)
            group_scores.append(matched / len(required_group))
        if group_scores:
            score += weights["manipulators"] * (sum(group_scores) / len(group_scores))

    # Sensors – treat legacy flat lists as pure sensor requirements
    if sensor_requirements:
        total_weight += weights["sensors"]
        robot_sensors = set(robot.sensors or [])
        matched = sum(tool in robot_sensors for tool in sensor_requirements)
        score += weights["sensors"] * (matched / len(sensor_requirements))

    # Communication 
    total_weight += weights["communication"]
    if task.communication_requirements:
        matched_comm = sum(proto in robot.communication_protocols for proto in task.communication_requirements)
        comm_score = matched_comm / len(task.communication_requirements)
        score += weights["communication"] * comm_score

    # Safety 
    total_weight += weights["safety"]
    if robot.safety_features and task.safety_protocols:
        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        score += weights["safety"] * safety_score

    # Environmental 
    total_weight += weights["environmental"]
    if robot.environmental_resistance and task.environmental_conditions:
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        score += weights["environmental"] * environmental_score

    # Navigation 
    total_weight += weights["navigation"]
    if task.navigation_constraints:
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_score == 0:
            return 0.0
        score += weights["navigation"] * navigation_score

    # Sensor range 
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    sensor_score = 1.0 if robot.sensor_range >= distance_to_task else \
                   0.5 if robot.sensor_range >= distance_to_task / 2 else 0.0
    score += weights["sensor_range"] * sensor_score

    # Proximity 
    total_weight += weights["proximity"]
    if distance_to_task < 20.0:
        score += weights["proximity"]
    elif distance_to_task < 50.0:
        score += weights["proximity"] * 0.5

    # Autonomy 
    total_weight += weights["autonomy_match"]
    autonomy_score = 0.0
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        autonomy_score = 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        autonomy_score = 0.5
    score += weights["autonomy_match"] * autonomy_score

    # Battery 
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = 1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete) else 0.5
    score += weights["battery_duration"] * battery_score

    # Special functions 
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "assembly": ["object recognition", "object tracking", "precise alignment"],
        "utilities": ["percise alignment", "balance control"],
        "excavate": [ "terrain leveling", "object recognition", "precise alignment"],
        "debris": ["balance control", "object recognition"],
        "level": ["terrain leveling", "object recognition"],
        "item elevation": ["precise alignment", "object tracking", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
        "remove scaffold": ["object recognition", "object tracking", "precise alignment"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # Processing power 
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 5.0:
            proc_score = 1.0
        elif robot.processing_power >= 3.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 3.0:
            proc_score = 1.0
        else:
            proc_score = 0.75
    elif task.difficulty > 2:
        if robot.processing_power >= 1.5:
            proc_score = 1.0
        else:
            proc_score = 0.75
    score += weights["processing_power"] * proc_score

    # Adaptability 
    total_weight += weights["adaptability"]
    score += weights["adaptability"] * (1.0 if robot.adaptability else 0.0)

    # Reward/Difficulty ratio 
    reward_score = (task.reward / max(task.difficulty, 1.0))
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    score += priority_multiplier * (reward_score / (reward_score + 10.0))  # squash into (0,1)

    # Normalize 
    if total_weight > 0:
        final_score = score / total_weight
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
        "autonomy_match": 0.5,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "adaptability": 0.5,
        "navigation": 2.0,
    }

    # Payload
    total_weight += weights["payload"]
    if any("payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        score += 0.0
    else:
        score += weights["payload"]

    # Reach
    total_weight += weights["reach"]
    if any("reach" in req and robot.reach < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        score += 0.0
    else:
        score += weights["reach"]

    # # Manipulators (tools_needed[1] is manipulators list)
    # # Manipulators
    # if task.tools_needed:
    #     total_weight += weights["manipulators"]
        
    #     group_scores = []
    #     # Iterate through each required tool group (the "AND" part)
    #     for required_group in task.tools_needed:
    #         if not required_group:
    #             continue

    #         # 1. First, check if the robot has at least one tool for this group (the "OR" part)
    #         if not any(tool in robot.manipulators for tool in required_group):
    #             return 0.0  # If any AND group is not met, the robot is unsuitable.

    #         # 2. If the group is met, calculate the scalar score for redundancy
    #         matched_count = sum(tool in robot.manipulators for tool in required_group)
    #         group_score = matched_count / len(required_group)
    #         group_scores.append(group_score)

    #     # 3. If the loop completed, average the scores to get the final tool_score
    #     if group_scores:
    #         tool_score = sum(group_scores) / len(group_scores)
    #         score += weights["manipulators"] * tool_score
    # # total_weight += weights["manipulators"]
    # # if task.tools_needed:
    # #     if ("cable hoist" in task.tools_needed[1] and "cable hoist" not in robot.manipulators and robot.mobility_type in ["hovering", "aerial"]):
    # #         task.tools_needed[1].remove("cable hoist")
    # #     matched_tools = sum(tool in robot.manipulators for tool in task.tools_needed[1])
    # #     tool_score = matched_tools / len(task.tools_needed[1])
    # #     score += weights["manipulators"] * tool_score

    # # Sensors (tools_needed[0] is sensors list)
    # total_weight += weights["sensors"]
    # if task.tools_needed:
    #     matched_tools = sum(tool in robot.sensors for tool in task.tools_needed[0])
    #     tool_score = matched_tools / len(task.tools_needed[0])
    #     score += weights["sensors"] * tool_score

    sensor_requirements, manipulator_groups = _split_tool_requirements(task.tools_needed)

    # Manipulators – every AND-group must be satisfied by at least one manipulator the robot carries
    if manipulator_groups:
        total_weight += weights["manipulators"]
        group_scores = []
        robot_manips = set(robot.manipulators or [])
        for required_group in manipulator_groups:
            if not any(tool in robot_manips for tool in required_group):
                score += 0.0  # hard requirement missing
            matched = sum(tool in robot_manips for tool in required_group)
            group_scores.append(matched / len(required_group))
        if group_scores:
            score += weights["manipulators"] * (sum(group_scores) / len(group_scores))

    # Sensors – treat legacy flat lists as pure sensor requirements
    if sensor_requirements:
        total_weight += weights["sensors"]
        robot_sensors = set(robot.sensors or [])
        matched = sum(tool in robot_sensors for tool in sensor_requirements)
        score += weights["sensors"] * (matched / len(sensor_requirements))

    # Communication
    total_weight += weights["communication"]
    if task.communication_requirements:
        matched_comm = sum(proto in robot.communication_protocols for proto in task.communication_requirements)
        comm_score = matched_comm / len(task.communication_requirements)
        score += weights["communication"] * comm_score

    # Safety
    total_weight += weights["safety"]
    if robot.safety_features and task.safety_protocols:
        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        score += weights["safety"] * safety_score

    # Environmental
    total_weight += weights["environmental"]
    if robot.environmental_resistance and task.environmental_conditions:
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        score += weights["environmental"] * environmental_score

    # Navigation
    total_weight += weights["navigation"]
    if task.navigation_constraints:
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        score += weights["navigation"] * navigation_score

    # Sensor range
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    sensor_score = 1.0 if robot.sensor_range >= distance_to_task else \
                   0.5 if robot.sensor_range >= distance_to_task / 2 else 0.0
    score += weights["sensor_range"] * sensor_score

    # Proximity
    total_weight += weights["proximity"]
    if distance_to_task < 20.0:
        score += weights["proximity"]
    elif distance_to_task < 50.0:
        score += weights["proximity"] * 0.5

    # Autonomy
    total_weight += weights["autonomy_match"]
    autonomy_score = 0.0
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        autonomy_score = 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        autonomy_score = 0.5
    score += weights["autonomy_match"] * autonomy_score

    # Battery
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = (1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete)
                    else 0 if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life
                    else 0.5)
    score += weights["battery_duration"] * battery_score

    # Special functions
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "assembly": ["object recognition", "object tracking", "precise alignment"],
        "utilities": ["percise alignment", "balance control"],
        "excavate": [ "terrain leveling", "object recognition", "precise alignment"],
        "debris": ["balance control", "object recognition"],
        "level": ["terrain leveling", "object recognition"],
        "item elevation": ["precise alignment", "object tracking", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
        "remove scaffold": ["object recognition", "object tracking", "precise alignment"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # Processing power
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 5.0:
            proc_score = 1.0
        elif robot.processing_power >= 3.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 3.0:
            proc_score = 1.0
        else:
            proc_score = 0.75
    elif task.difficulty > 2:
        if robot.processing_power >= 1.5:
            proc_score = 1.0
        else:
            proc_score = 0.75
    score += weights["processing_power"] * proc_score

    # Adaptability
    total_weight += weights["adaptability"]
    score += weights["adaptability"] * (1.0 if robot.adaptability else 0.0)

    # Reward/Difficulty ratio
    reward_score = (task.reward / max(task.difficulty, 1.0))
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    score += priority_multiplier * (reward_score / (reward_score + 10.0))  # squash into (0,1)

    # Normalize
    if total_weight > 0:
        final_score = score / total_weight
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
        "autonomy_match": 0.5,
        "battery_duration": 2.0,
        "special_functions": 2.0,
        "processing_power": 1.0,
        "adaptability": 0.5,
        "navigation": 2.0,
    }

    # Payload
    total_weight += weights["payload"]
    if any("payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        return 0.0
    else:
        score += weights["payload"]

    # Reach
    total_weight += weights["reach"]
    if any("reach" in req and robot.reach < float(req.split(">= ")[-1]) for req in task.required_capabilities):
        return 0.0
    else:
        score += weights["reach"]

    # # Manipulators (tools_needed[1] is manipulators list)
    # # Manipulators
    # if task.tools_needed:
    #     total_weight += weights["manipulators"]
        
    #     # Iterate through each required tool group (the "AND" part)
    #     for required_group in task.tools_needed:
    #         # Check if the robot has at least one tool from the current group (the "OR" part)
    #         has_matching_tool = any(tool in robot.manipulators for tool in required_group)
            
    #         # If a required group is not satisfied, the robot is completely unsuitable.
    #         if not has_matching_tool:
    #             return 0.0 # Exit immediately with a score of 0

    #     # If the loop completes, it means all tool groups were satisfied.
    #     # Award the full score for this category.
    #     score += weights["manipulators"]
    # # total_weight += weights["manipulators"]
    # # if task.tools_needed:
    # #     if ("cable hoist" in task.tools_needed[1] and "cable hoist" not in robot.manipulators and robot.mobility_type in ["hovering", "aerial"]):
    # #         task.tools_needed[1].remove("cable hoist")
    # #     matched_tools = sum(tool in robot.manipulators for tool in task.tools_needed[1])
    # #     tool_score = matched_tools / len(task.tools_needed[1])
    # #     if tool_score != 1:
    # #         return 0.0
    # #     score += weights["manipulators"] * tool_score

    # # Sensors (tools_needed[0] is sensors list)
    # total_weight += weights["sensors"]
    # if task.tools_needed:
    #     matched_tools = sum(tool in robot.sensors for tool in task.tools_needed[0])
    #     tool_score = matched_tools / len(task.tools_needed[0])
    #     if tool_score != 1:
    #         return 0.0
    #     score += weights["sensors"] * tool_score

    sensor_requirements, manipulator_groups = _split_tool_requirements(task.tools_needed)

    # Manipulators – every AND-group must be satisfied by at least one manipulator the robot carries
    if manipulator_groups:
        total_weight += weights["manipulators"]
        group_scores = []
        robot_manips = set(robot.manipulators or [])
        for required_group in manipulator_groups:
            if not any(tool in robot_manips for tool in required_group):
                return 0.0  # hard requirement missing
            matched = sum(tool in robot_manips for tool in required_group)
            group_scores.append(matched / len(required_group))
        if group_scores:
            score += weights["manipulators"] * (sum(group_scores) / len(group_scores))

    # Sensors – treat legacy flat lists as pure sensor requirements
    if sensor_requirements:
        total_weight += weights["sensors"]
        robot_sensors = set(robot.sensors or [])
        matched = sum(tool in robot_sensors for tool in sensor_requirements)
        score += weights["sensors"] * (matched / len(sensor_requirements))

    # Communication
    total_weight += weights["communication"]
    if task.communication_requirements:
        matched_comm = sum(proto in robot.communication_protocols for proto in task.communication_requirements)
        comm_score = matched_comm / len(task.communication_requirements)
        if comm_score != 1:
            return 0.0
        score += weights["communication"] * comm_score

    # Safety
    total_weight += weights["safety"]
    if robot.safety_features and task.safety_protocols:
        matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
        safety_score = matched_safety / len(task.safety_protocols)
        if safety_score != 1:
            return 0.0
        score += weights["safety"] * safety_score

    # Environmental
    total_weight += weights["environmental"]
    if robot.environmental_resistance and task.environmental_conditions:
        matched_environmental = sum(condition in robot.environmental_resistance for condition in task.environmental_conditions)
        environmental_score = matched_environmental / len(task.environmental_conditions)
        if environmental_score != 1:
            return 0.0
        score += weights["environmental"] * environmental_score

    # Navigation
    total_weight += weights["navigation"]
    if task.navigation_constraints:
        navigation_score = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_score == 0:
            return 0.0
        score += weights["navigation"] * navigation_score

    # Sensor range
    total_weight += weights["sensor_range"]
    distance_to_task = len(robot.current_path) - 1
    sensor_score = 1.0 if robot.sensor_range >= distance_to_task else \
                   0.5 if robot.sensor_range >= distance_to_task / 2 else 0.0
    score += weights["sensor_range"] * sensor_score

    # Proximity
    total_weight += weights["proximity"]
    if distance_to_task < 20.0:
        score += weights["proximity"]
    elif distance_to_task < 50.0:
        score += weights["proximity"] * 0.5

    # Autonomy
    total_weight += weights["autonomy_match"]
    autonomy_score = 0.0
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        autonomy_score = 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        autonomy_score = 0.5
    score += weights["autonomy_match"] * autonomy_score

    # Battery
    total_weight += weights["battery_duration"]
    if ((distance_to_task / robot.max_speed) + task.time_to_complete) > robot.battery_life:
        return 0.0
    battery_score = 1.0 if robot.battery_life >= 2 * ((distance_to_task / robot.max_speed) + task.time_to_complete) else 0.5
    score += weights["battery_duration"] * battery_score

    # Special functions
    total_weight += weights["special_functions"]
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "assembly": ["object recognition", "object tracking", "precise alignment"],
        "utilities": ["percise alignment", "balance control"],
        "excavate": [ "terrain leveling", "object recognition", "precise alignment"],
        "debris": ["balance control", "object recognition"],
        "level": ["terrain leveling", "object recognition"],
        "item elevation": ["precise alignment", "object tracking", "balance control"],
        "lay bricks": ["object recognition", "precise alignment"],
        "scaffold": ["precise alignment", "balance control"],
        "remove scaffold": ["object recognition", "object tracking", "precise alignment"],
    }
    required_functions = task_function_mapping[task.task_type]
    matched_special_functions = sum(special in robot.special_functions for special in required_functions)
    special_functions_score = matched_special_functions / len(required_functions)
    score += weights["special_functions"] * special_functions_score

    # Processing power
    total_weight += weights["processing_power"]
    proc_score = 0.0
    if task.difficulty > 7:
        if robot.processing_power >= 5.0:
            proc_score = 1.0
        elif robot.processing_power >= 3.0:
            proc_score = 0.75
        else:
            proc_score = 0.5
    elif task.difficulty > 4:
        if robot.processing_power >= 3.0:
            proc_score = 1.0
        else:
            proc_score = 0.75
    elif task.difficulty > 2:
        if robot.processing_power >= 1.5:
            proc_score = 1.0
        else:
            proc_score = 0.75
    score += weights["processing_power"] * proc_score

    # Adaptability
    total_weight += weights["adaptability"]
    score += weights["adaptability"] * (1.0 if robot.adaptability else 0.0)

    # Reward/Difficulty ratio
    reward_score = (task.reward / max(task.difficulty, 1.0))
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    score += priority_multiplier * (reward_score / (reward_score + 10.0))  # squash into (0,1)

    # Normalize
    if total_weight > 0:
        final_score = score / total_weight
    else:
        final_score = 0.0

    return max(0.0, min(1.0, final_score))

def evaluate_suitability_distance(robot: CapabilityProfile, task: TaskDescription) -> float:
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
    
#     print(task.required_capabilities, robot.payload_capacity)
    # Check if robot meets the minimum requirements
    if any(req for req in task.required_capabilities if "payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot doesn't meet minimum requirements
    else:
        score += 1.0  # Add score if payload meets or exceeds requirements
    
#     print(task.tools_needed, robot.sensors+robot.manipulators)
    # Check if the robot has the necessary tools for the task
    if robot.sensors:
        if task.tools_needed and not all(item in robot.sensors for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools

    if robot.manipulators:
        if task.tools_needed and not all(item in robot.manipulators for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools
    
#     print(task.communication_requirements, robot.communication_protocols)
    # Check if the robot can communicate as required by the task
    if task.communication_requirements and not all(protocol in robot.communication_protocols for protocol in task.communication_requirements):
        score += 0.0  # Suitability is zero if the robot lacks required communication protocols
    else:
        score += 1.0  # Add score if robot has communication requirements
    
#     print(task.safety_protocols, robot.safety_features)
    # Check if the robot can safely perform the task
    if robot.safety_features and task.safety_protocols:
        if task.safety_protocols and not all(safety in robot.safety_features for safety in task.safety_protocols):
            score += 0.0  # Suitability is zero if the robot lacks required safety features
        else:
            score += 1.0  # Add score if robot meets safety requirements
    
#     print(task.environmental_conditions, robot.environmental_resistance)
    # Environmental compatibility: Can the robot operate in the task’s conditions?
    if robot.environmental_resistance and task.environmental_conditions:
        if task.environmental_conditions and not all(condition in robot.environmental_resistance for condition in task.environmental_conditions):
            score += 0.0  # Suitability is zero if the robot can't operate in required environmental conditions
        else:
            score += 1.0  # Add score if robot has required environmental resistances
    
#     print(task.required_capabilities, robot.reach)
    # Check if the robot meets reach requirements
    if any(req for req in task.required_capabilities if "reach" in req and robot.reach < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot cannot reach the task area as required
    else:
        score += 1.0  # Add score if reach meets or exceeds requirements
    
#     print(task.navigation_constraints, robot.mobility_type, robot.size)
    # Check navigation constraints based on mobility type and robot size
    if task.navigation_constraints:
        navigation_match = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_match == 0:
            score += 0.0
        else:
            score += navigation_match

    # NOTE: CHANGED TO WORK WITH COORDINATES
    # distance_to_task = ((robot.location[0] - task.location[0]) ** 2 + (robot.location[1] - task.location[1]) ** 2) ** 0.5
    # stop suitability matrix from going negative
    distance_to_task = max(0, len(robot.current_path) - 1)
#     print(robot.sensor_range)
    # Check sensor capabilities for the task
    if robot.sensor_range:
        if robot.sensor_range >= distance_to_task:
            score += 1.0
        elif robot.sensor_range >= distance_to_task/2:
            score += 0.5

    # Battery and distance check: Ensure the robot has sufficient battery to reach and complete the task
#     print(robot.max_speed, robot.battery_life, task.duration, distance_to_task)
    if ((distance_to_task / robot.max_speed)+task.time_to_complete) > robot.battery_life:
        score += 0.0  # Suitability is zero if the robot can't complete the task due to distance, speed, or battery limitations

    # Add to score based on proximity (closer robots get higher scores)
#     if distance_to_task < 20.0:
#         score += 1.0
#     elif distance_to_task < 50.0:
#         score += 0.5

#     print(task.priority_level, robot.autonomy_level)
    # Check if the robot's autonomy level matches the task's priority level
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        score += 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        score += 0.5

#     print(robot.battery_life, task.duration)
    # Evaluate battery life for task duration
    if robot.battery_life >= 2*((distance_to_task / robot.max_speed)+task.time_to_complete):
        score += 1.0
    else:
        score += 0.5

#     print(task.task_type, robot.special_functions)
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "inspection": ["object recognition", "object tracking", "gesture recognition"],
        "cleaning": ["object recognition"],
        "monitoring": ["speech output", "object tracking", "facial recognition"],
        "maintenance": ["object recognition", "path planning"],
        "assembly": ["object recognition"],
        "surveying": ["speech output", "facial recognition", "object recognition", "object tracking"],
        "data collection": ["object recognition", "object tracking", "facial recognition", "gesture recognition"],
        "assistance": ["speech output", "facial recognition", "gesture recognition"]
    }

    # Get the relevant functions for this task type
    required_functions = task_function_mapping[task.task_type]

    # Calculate the score based on matches between robot's functions and required functions
    if robot.special_functions:
        for function in robot.special_functions:
            if function in required_functions:
                score += 1.0  # Increase score for each match
    
#     # Dependencies
#     if task.dependencies:
#         # Assume dependencies are represented as tasks that must be completed first
#         score += 0.5 if all(dep in completed_tasks for dep in task.dependencies) else 0.0
    
#     print(task.difficulty, robot.processing_power)
    # Processing power: Certain tasks may benefit from higher processing power if they are computationally demanding
    if task.difficulty > 7 and robot.processing_power >= 5.0:  # Difficulty > 7 indicates a complex task
        score += 1.0
    elif task.difficulty > 4 and robot.processing_power >= 3.0:
        score += 1.0
    elif task.difficulty > 2 and robot.processing_power >= 1.5:
        score += 0.5

#     print(robot.adaptability)
    # Consider robot's adaptability to changing conditions
    if robot.adaptability:
        score += 0.5
    
#     print(task.task_type, robot.preferred_tasks)
    # Preference matching
    #if task.task_type in robot.preferred_tasks:
    #    score += 1.0

    # Score based on priority, reward, and difficulty
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    reward_to_difficulty_ratio = task.reward / task.difficulty
#     print(task.priority_level, task.reward, task.difficulty, priority_multiplier, reward_to_difficulty_ratio)
    score += priority_multiplier * reward_to_difficulty_ratio

    # Weight score by distance to task
    # NOTE: IF THE ROBOT IS AT THE TASK THIS CAN CAUSE A DIVIDE BY ZERO ERROR
    score = score / (distance_to_task + 1E-5)
    
    # Return the final suitability score
#     print(score)
    return score

def evaluate_suitability_priority(robot: CapabilityProfile, task: TaskDescription) -> float:
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
    
#     print(task.required_capabilities, robot.payload_capacity)
    # Check if robot meets the minimum requirements
    if any(req for req in task.required_capabilities if "payload capacity" in req and robot.payload_capacity < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot doesn't meet minimum requirements
    else:
        score += 1.0  # Add score if payload meets or exceeds requirements
    
#     print(task.tools_needed, robot.sensors+robot.manipulators)
    # Check if the robot has the necessary tools for the task
    if robot.sensors:
        if task.tools_needed and not all(item in robot.sensors for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools
    
    if robot.manipulators:
        if task.tools_needed and not all(item in robot.manipulators for item in task.tools_needed):
            score += 0.0  # Suitability is zero if the robot lacks necessary tools
        else:
            score += 1.0  # Add score if robot has necessary tools
    
#     print(task.communication_requirements, robot.communication_protocols)
    # Check if the robot can communicate as required by the task
    if task.communication_requirements and not all(protocol in robot.communication_protocols for protocol in task.communication_requirements):
        score += 0.0  # Suitability is zero if the robot lacks required communication protocols
    else:
        score += 1.0  # Add score if robot has communication requirements
    
#     print(task.safety_protocols, r
# obot.safety_features)
    # Check if the robot can safely perform the task
    if robot.safety_features and task.safety_protocols:
        if task.safety_protocols and not all(safety in robot.safety_features for safety in task.safety_protocols):
            score += 0.0  # Suitability is zero if the robot lacks required safety features
        else:
            score += 1.0  # Add score if robot meets safety requirements
    
#     print(task.environmental_conditions, robot.environmental_resistance)
    # Environmental compatibility: Can the robot operate in the task’s conditions?
    if robot.environmental_resistance and task.environmental_conditions:
        if task.environmental_conditions and not all(condition in robot.environmental_resistance for condition in task.environmental_conditions):
            score += 0.0  # Suitability is zero if the robot can't operate in required environmental conditions
        else:
            score += 1.0  # Add score if robot has required environmental resistances
    
#     print(task.required_capabilities, robot.reach)
    # Check if the robot meets reach requirements
    if any(req for req in task.required_capabilities if "reach" in req and robot.reach < float(req.split(">= ")[-1])):
        score += 0.0  # Suitability is zero if the robot cannot reach the task area as required
    else:
        score += 1.0  # Add score if reach meets or exceeds requirements
    
#     print(task.navigation_constraints, robot.mobility_type, robot.size)
    # Check navigation constraints based on mobility type and robot size
    if task.navigation_constraints:
        navigation_match = navigation_suitability(robot.mobility_type, robot.size, task.navigation_constraints)
        if navigation_match == 0:
            score += 0.0
        else:
            score += navigation_match

    # NOTE: CHANGED TO WORK WITH COORDINATES
    # distance_to_task = ((robot.location[0] - task.location[0]) ** 2 + (robot.location[1] - task.location[1]) ** 2) ** 0.5
    distance_to_task = len(robot.current_path) - 1
#     print(robot.sensor_range)
    # Check sensor capabilities for the task
    if robot.sensor_range:
        if robot.sensor_range >= distance_to_task:
            score += 1.0
        elif robot.sensor_range >= distance_to_task/2:
            score += 0.5

    # Battery and distance check: Ensure the robot has sufficient battery to reach and complete the task
#     print(robot.max_speed, robot.battery_life, task.duration, distance_to_task)
    if ((distance_to_task / robot.max_speed)+task.time_to_complete) > robot.battery_life:
        score += 0.0  # Suitability is zero if the robot can't complete the task due to distance, speed, or battery limitations

    # Add to score based on proximity (closer robots get higher scores)
    if distance_to_task < 20.0:
        score += 1.0
    elif distance_to_task < 50.0:
        score += 0.5

#     print(task.priority_level, robot.autonomy_level)
    # Check if the robot's autonomy level matches the task's priority level
    if task.priority_level in ["high", "urgent"] and robot.autonomy_level in ["fully autonomous", "teleoperated"]:
        score += 1.0
    elif task.priority_level in ["medium", "low"] and robot.autonomy_level in ["semi-autonomous", "fully autonomous"]:
        score += 0.5

#     print(robot.battery_life, task.duration)
    # Evaluate battery life for task duration
    if robot.battery_life >= 2*((distance_to_task / robot.max_speed)+task.time_to_complete):
        score += 1.0
    else:
        score += 0.5

#     print(task.task_type, robot.special_functions)
    task_function_mapping = {
        "delivery": ["object recognition", "speech output", "facial recognition"],
        "inspection": ["object recognition", "object tracking", "gesture recognition"],
        "cleaning": ["object recognition"],
        "monitoring": ["speech output", "object tracking", "facial recognition"],
        "maintenance": ["object recognition", "path planning"],
        "assembly": ["object recognition"],
        "surveying": ["speech output", "facial recognition", "object recognition", "object tracking"],
        "data collection": ["object recognition", "object tracking", "facial recognition", "gesture recognition"],
        "assistance": ["speech output", "facial recognition", "gesture recognition"]
    }

    # Get the relevant functions for this task type
    required_functions = task_function_mapping[task.task_type]

    # Calculate the score based on matches between robot's functions and required functions
    if robot.special_functions:
        for function in robot.special_functions:
            if function in required_functions:
                score += 1.0  # Increase score for each match
    
#     # Dependencies
#     if task.dependencies:
#         # Assume dependencies are represented as tasks that must be completed first
#         score += 0.5 if all(dep in completed_tasks for dep in task.dependencies) else 0.0
    
#     print(task.difficulty, robot.processing_power)
    # Processing power: Certain tasks may benefit from higher processing power if they are computationally demanding
    if task.difficulty > 7 and robot.processing_power >= 5.0:  # Difficulty > 7 indicates a complex task
        score += 1.0
    elif task.difficulty > 4 and robot.processing_power >= 3.0:
        score += 1.0
    elif task.difficulty > 2 and robot.processing_power >= 1.5:
        score += 0.5

#     print(robot.adaptability)
    # Consider robot's adaptability to changing conditions
    if robot.adaptability:
        score += 0.5
    
#     print(task.task_type, robot.preferred_tasks)
    # Preference matching
    #if task.task_type in robot.preferred_tasks:
    #    score += 1.0

    # Score based on reward and difficulty
    priority_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}[task.priority_level]
    reward_to_difficulty_ratio = task.reward / task.difficulty
#     print(task.priority_level, task.reward, task.difficulty, priority_multiplier, reward_to_difficulty_ratio)
    score += reward_to_difficulty_ratio

    # Weight based on priority
    score = score * priority_multiplier
    
    # Return the final suitability score
#     print(score)
    return score

# --- Minimal serializers so the prompt stays short & cheap ---
def _robot_to_dict(r):
    return {
        "id": r.robot_id,
        "payload_capacity": r.payload_capacity,
        "reach": r.reach,
        "sensor_range": r.sensor_range,
        "battery_life": r.battery_life,
        "processing_power": r.processing_power,
        "adaptability": bool(getattr(r, "adaptability", False)),
        "autonomy_level": getattr(r, "autonomy_level", None),
        "manipulators": getattr(r, "manipulators", []),
        "navigation": getattr(r, "navigation_constraints", []),
        "sensors": getattr(r, "sensors", []),
        "comm": getattr(r, "communication_protocols", []),
        "safety": getattr(r, "safety_features", []),
        "special": getattr(r, "special_functions", []),
        "location": tuple(getattr(r, "location", (0,0))),
        # add only what you truly need
    }

def _task_to_dict(t):
    return {
        "id": t.task_id,
        "tools_needed": getattr(t, "tools_needed", [[],[]]),  # [sensors, manipulators]
        "navigation_constraints": getattr(t, "navigation_constraints", []),
        "payload_required": getattr(t, "payload_required", 0.0),
        "reach_required": getattr(t, "reach_required", 0.0),
        "sensors_required": getattr(t, "sensors_required", []),
        "communications_required": getattr(t, "communications_required", []),
        "safety_protocols": getattr(t, "safety_protocols", []),
        "priority": getattr(t, "priority_level", None),
        "difficulty": getattr(t, "difficulty", None),
        "location": tuple(getattr(t, "location", (0,0))),
        "duration": float(getattr(t, "time_to_complete", 0.0)),
        "reset_progress": bool(getattr(t, "reset_progress", False)),
        # add only what you truly need
    }

def _json_default(o):
    # Enums -> names (or use .value if you prefer)
    if isinstance(o, Enum):
        return o.name
    # numpy scalars -> Python scalars
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    # numpy arrays -> lists
    if isinstance(o, np.ndarray):
        return o.tolist()
    # sets/tuples -> lists
    if isinstance(o, (set, tuple)):
        return list(o)
    # dataclasses -> dict
    if is_dataclass(o):
        return asdict(o)
    # fallback: string
    return str(o)

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

def _robot_min_view(r):
    return dict(
        robot_id=getattr(r, "robot_id", None),
        mobility_type=getattr(r, "mobility_type", None),
        manipulators=list(getattr(r, "manipulators", []) or []),
        payload_capacity=float(getattr(r, "payload_capacity", 0) or 0),
        reach=float(getattr(r, "reach", 0) or 0),
        sensors=list(getattr(r, "sensors", []) or []),
        sensor_range=float(getattr(r, "sensor_range", 0) or 0),
        communication_protocols=list(getattr(r, "communication_protocols", []) or []),
        safety_features=list(getattr(r, "safety_features", []) or []),
        special_functions=list(getattr(r, "special_functions", []) or []),
        processing_power=float(getattr(r, "processing_power", 0) or 0),
        autonomy_level=str(getattr(r, "autonomy_level", "")),
        battery_life=float(getattr(r, "battery_life", 0) or 0),
        max_speed=float(getattr(r, "max_speed", 1.0) or 1.0),
    )

def _task_name_view(t):
    # Try to read nl_description if present; otherwise just pass the task_type
    d = {
        "task_id": getattr(t, "task_id", None),
        "task_type": getattr(t, "task_type", None),
    }
    # If your TaskDescription stores the strict profile name or a custom field, add it here:
    desc = getattr(t, "nl_description", None) or getattr(t, "strict_profile_name", None)
    if isinstance(desc, str):
        d["nl_description"] = desc
    return d

def evaluate_suitability_from_names_with_llm(robots, tasks, model="meta-llama/Llama-4-Scout-17B-16E-Instruct:groq") -> np.ndarray:
    """
    Evaluate suitability of robots for tasks using an LLM based on names and minimal info.
    Returns an (R, T) float array with scores in [0,1].
    Parameters:
        robots: List of CapabilityProfile objects.
        tasks: List of TaskDescription objects.
        model: The LLM model to use (updated to  "meta-llama/Llama-4-Scout-17B-16E-Instruct:groq").
    Returns:
        M: An (R, T) numpy array of float suitability scores in [0,1].
    """
    evaluate_suitability_from_names_with_llm._is_llm_batch = True
    client = InferenceClient(api_key=os.environ["HF_TOKEN"],
)


    R = [_to_jsonable(_robot_min_view(r)) for r in robots]
    T = [_to_jsonable(_task_name_view(t)) for t in tasks]


    text_prompt = build_name_only_prompt(
        robots_json=json.dumps(R, ensure_ascii=False),
        tasks_json=json.dumps(T, ensure_ascii=False)
    )

    # Chat API (fill in your SDK call)
    # Example OpenAI Chat Completions:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content":  "You are a careful assistant that outputs only the requested format."},
            {"role": "user", "content": text_prompt}],
        # respect model limitations (e.g., do NOT set temperature for 5-nano if disallowed)
    )
    content = resp.choices[0].message.content

    # after you get the model's raw text in variable `text`
    M = _parse_output_matrix(content, nR=len(robots), nT=len(tasks))
    if M is None:
        # fallback: tiny random noise to break ties, or zeros
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
