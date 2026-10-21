import random
from typing import List

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

# STRICT ROBOT PROFILES FOR ROBOT GENERATION
# 16 items per profile
# LiDAR, GPS, proximity sensor, camera, infrared, magnetometer, ultrasonic
# Wi-Fi, 4G, 5G, Radio, Bluetooth
# "weatherproof", "heat-resistant", "cold-resistant", "shock-resistant", "dustproof", "heat-resistant", "wind-resistant"
# "obstacle detection", "emergency stop", "collision avoidance", "overheat protection", "overload protection", "fall detection"
STRICT_ROBOT_PROFILES = [
    {
        "name": "delivery",
        "mobility_type": "aerial",
        "environmental_resistance": ["weatherproof", "heat-resistant"],
        "sensors": ["GPS", "ultrasonic"],
        "manipulators": ["gripper", "cargo space"],
        "communication_protocols": ["Wi-Fi", "4G"],
        "special_functions": ["object recognition"],
        "safety_features": ["obstacle detection", "emergency stop"],
        "sensor_range": 120.0,
        "processing_power": 5.0,
        "autonomy_level": "fully autonomous",
        "payload_capacity": 30.0,
        "reach": 0.0,
        "battery_life": 80.0,
        "size": (2.0, 2.0, 2.0),
        "adaptability": False
    },
    {
        "name": "assembly",
        "mobility_type": "aerial",
        "environmental_resistance": ["heat-resistant", "dustproof"],
        "sensors": ["proximity", "camera"],
        "manipulators": ["drill", "welding tool"],
        "communication_protocols": ["4G", "Radio"],
        "special_functions": ["gesture recognition"],
        "safety_features": ["emergency stop", "overheat protection"],
        "sensor_range": 50.0,
        "processing_power": 7.0,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 10.0,
        "reach": 3.0,
        "battery_life": 80.0,
        "size": (3.0, 2.0, 2.0),
        "adaptability": True
    },
    {
        "name": "excavator",
        "mobility_type": "tracked",
        "environmental_resistance": ["weatherproof", "heat-resistant"],
        "sensors": ["magnetometer", "LiDAR"],
        "manipulators": ["hydraulic bucket", "hydraulic arm"],
        "communication_protocols": ["Wi-Fi", "Radio"],
        "special_functions": ["terrain leveling"],
        "safety_features": ["collision avoidance", "fall detection"],
        "sensor_range": 50.0,
        "processing_power": 8.0,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,
        "reach": 2.0,
        "battery_life": 80.0,
        "size": (4.2, 2.5, 3.0),
        "adaptability": True
    },
    {
        "name": "bricklayer",
        "mobility_type": "wheeled",
        "environmental_resistance": ["dustproof", "shock-resistant"],
        "sensors": ["camera", "proximity"],
        "manipulators": ["mixer drum", "hydraulic bucket"],
        "communication_protocols": ["5G", "Bluetooth"],
        "special_functions": ["precise alignment"],
        "safety_features": ["obstacle detection", "overheat protection"],
        "sensor_range": 35.0,
        "processing_power": 5.5,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 4.0,  
        "reach": 4.0,
        "battery_life": 80.0,
        "size": (4.0, 2.0, 2.0),
        "adaptability": False
    },
    {
        "name": "crane",
        "mobility_type": "tracked", 
        "environmental_resistance": ["cold-resistant", "dustproof"],
        "sensors": ["camera", "proximity"],
        "manipulators": ["gripper", "cable hoist"],
        "communication_protocols": ["5G", "Radio"],
        "special_functions": ["precise alignment"],
        "safety_features": ["overheat protection", "overload protection"],
        "sensor_range": 50.0,
        "processing_power": 6.5,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,
        "reach": 25.0,
        "battery_life": 80.0,
        "size": (3.5, 3.5, 10.0),
        "adaptability": False
    },
    {
        "name": "scaffolding",
        "mobility_type": "legged",
        "environmental_resistance": ["shock-resistant", "wind-resistant"],
        "sensors": ["LiDAR", "ultrasonic"],
        "manipulators": ["welding tool", "scissor lift", "platform"],
        "communication_protocols": ["Wi-Fi", "Bluetooth"],
        "special_functions": ["balance control"],
        "safety_features": ["collision avoidance", "fall detection"],
        "sensor_range": 75.0,
        "processing_power": 7.0,
        "autonomy_level": "fully autonomous",
        "payload_capacity": 8.0,  
        "reach": 8.0,  
        "battery_life": 80.0,
        "size": (1.5, 1.0, 2.0),
        "adaptability": True
    }
]

# STRICT TASK PROFILES FOR TASK GENERATION
# 15 items per profile counting payload and reach instead of required_capabilities dict
"""
robot                       -> task
name                        -> task_type
mobility_type               -> navigation_constraints (specifically the agents that can make it up stairs, navigate slippery, loose debris, no fly zone, windy, dense obstructions, smooth surfaces, TOO MUCH)
environmental_resistances   -> environmental_conditions
sensors                     -> sensors_needed
manipulators                -> manipulators_needed
communication_protocols     -> communication_requirements
special_functions           -> mapped to task_type hard coded, potential remove
saftey_features             -> safety_protocols
sensor_range                -> check suitability function (path length?)
processing_power            -> difficulty
autonomy_level              -> priority level. check suitability function
payload_capacity            -> payload
reach                       -> reach
battery_life                -> duration and path length (maybe others?)
size                        -> navigation_constraints (width -> narrow spaces, height -> low ceilings)
adaptability                -> flat boost to score, useless, get rid
"""
STRICT_TASK_PROFILES = [
    {
        # random choices between: 
        # Sensors[ultrasonic proximity], 
        # Communications[5G, Wi-Fi], 
        # environmental[dustproof, heat-resistant]
        # safety[overheat protection, emergency stop]
        "task_type": "utilities grip", 
        "priority_level": "medium",
        "reward": 6.0,
        "difficulty": 6.0,
        "navigation_constraints": ["uneven floors", "loose debris"],
        "required_capabilities": {
            "payload": 20.0,
            "reach": 0.0,
        },
        "environmental_conditions": ["cold-resistant", "weatherproof"],
        "sensors_needed": ["GPS", "camera"],
        "manipulators_needed": ["gripper", "cargo space"],
        "communication_requirements": ["Radio", "4G"],
        "safety_protocols": ["overload protection", "obstacle detection"],
        "duration": 3.0,
        "performance_metric": "safety compliance",
        "nl_description": "Operate valves, switches, and panels; carry small parts; precise manipulation in plant rooms."
    },


    {
        # random choice between: 
        # Sensors[lidar, proximity], 
        # Communications[Wi-Fi, Bluetooth], 
        # environmental[dustproof, weatherproof]
        # safety[fall detection, obstacle detection]
        "task_type": "utilities bucket", 
        "priority_level": "medium",
        "reward": 7.0,
        "difficulty": 6.0,
        "navigation_constraints": ["uneven floors", "loose debris"],
        "required_capabilities": {
            "payload": 4.0,
            "reach": 2.0,
        },
        "environmental_conditions": ["heat-resistant", "shock-resistant"],
        "sensors_needed": ["magnetometer", "camera"],
        "manipulators_needed": ["hydraulic bucket", "hydraulic arm"],
        "communication_requirements": ["Radio", "5G"],
        "safety_protocols": ["overheat protection", "collision avoidance"],
        "duration": 3.0,
        "performance_metric": "safety compliance",
        "nl_description": "Transport construction materials, clear or move small bulk materials near utility corridors; load/unload with a bucket."
    },


    {
        # random choice between: 
        # Sensors[camera, ultrasonic], 
        # Communications[4G, Bluetooth], 
        # environmental[wind-resistant, heat-resistant]
        # safety[collision avoidance, overheat protection]
        "task_type": "utilities weld", 
        "priority_level": "medium",
        "reward": 5.0,
        "difficulty": 4.0,
        "navigation_constraints": ["loose debris", "crowded"],
        "required_capabilities": {
            "payload": 8.0,
            "reach": 3.0,
        },
        "environmental_conditions": ["shock-resistant", "dustproof"],
        "sensors_needed": ["proximity", "LiDAR"],
        "manipulators_needed": ["welding tool", "drill"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["fall detection", "emergency stop"],
        "duration": 2.0,
        "performance_metric": "safety compliance",
        "nl_description": "Pick and remove scattered debris in cluttered passages; careful grasping and placement."
    },


    {
        # random choice between: 
        # Sensors[proximity, ultrasonic], 
        # Communications[5G, Bluetooth],
        # environmental[wind-resistant, dustproof]
        # safety[collision avoidance, overload protection]
        "task_type": "move crate", 
        "priority_level": "medium",
        "reward": 6.0,
        "difficulty": 4.0,
        "navigation_constraints": ["loose debris", "uneven floors"],
        "required_capabilities": {
            "payload": 8.0,
            "reach": 8.0,
        },
        "environmental_conditions": ["shock-resistant", "cold-resistant"],
        "sensors_needed": ["LiDAR", "camera"],
        "manipulators_needed": ["cable hoist", "scissor lift", "platform"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["fall detection", "overheat protection"],
        "duration": 2.0,
        "performance_metric": "safety compliance",
        "nl_description": "Scoop and relocate piles of loose debris; continuous removal in uneven terrain.",
        
    },


    {
        "task_type": "delivery",
        "priority_level": "low",
        "reward": 3.0,
        "difficulty": 5.0,
        "navigation_constraints": ["crowded", "elevator"],
        "required_capabilities": {
            "payload": 28.0,
            "reach": 0.0,
        },
        "environmental_conditions": ["weatherproof", "heat-resistant"],
        "sensors_needed": ["GPS"],
        "manipulators_needed": ["gripper", "cargo space"],
        "communication_requirements": ["Wi-Fi", "4G"],
        "safety_protocols": ["obstacle detection", "emergency stop"],
        "duration": 1.0,
        "performance_metric": "time taken",
        "nl_description": "Fetch-and-carry small payloads point-to-point through indoor/outdoor corridors.",
    },


    {
        "task_type": "assembly",
        "priority_level": "urgent",
        "reward": 10.0,
        "difficulty": 8.0,
        "navigation_constraints": ["crowded"],
        "required_capabilities": {
            "payload": 10.0,
            "reach": 3.0,
        },
        "environmental_conditions": ["heat-resistant", "dustproof"],
        "sensors_needed": ["proximity", "camera"],
        "manipulators_needed": ["drill", "welding tool"],
        "communication_requirements": ["4G", "Radio"],
        "safety_protocols": ["emergency stop", "overheat protection"],
        "duration": 4.0,
        "performance_metric": "accuracy",
        "nl_description": "Fixture placement, fastening, dispensing, or welding with moderate precision in crowded areas.",
    },


    {
        "task_type": "excavate",
        "priority_level": "urgent",
        "reward": 11.0,
        "difficulty": 8.0,
        "navigation_constraints": ["loose debris", "low visibility"],
        "required_capabilities": {
            "payload": 20.0,
            "reach": 2.0,
        },
        "environmental_conditions": ["weatherproof", "heat-resistant"],
        "sensors_needed": ["magnetometer", "LiDAR"],
        "manipulators_needed": ["hydraulic bucket", "hydraulic arm"],
        "communication_requirements": ["Wi-Fi", "Radio"],
        "safety_protocols": ["collision avoidance", "fall detection"],
        "duration": 4.0,
        "performance_metric": "safety compliance",
        "nl_description": "Dig, trench, or remove soil/rubble; sustained scooping with high payload demands.",
    },

    {
        "task_type": "lay bricks",
        "priority_level": "low",
        "reward": 8.0,
        "difficulty": 6.0,
        "navigation_constraints": ["crowded", "windy"],
        "required_capabilities": {
            "payload": 4.0,
            "reach": 4.0,
        },
        "environmental_conditions": ["dustproof", "shock-resistant"],
        "sensors_needed": ["camera", "proximity"],
        "manipulators_needed": ["mixer drum", "hydraulic bucket"],
        "communication_requirements": ["5G", "Bluetooth"],
        "safety_protocols": ["obstacle detection", "overheat protection"],
        "duration": 3.0,
        "performance_metric": "accuracy",
        "nl_description": "Pick, mortar/dispense, and place bricks with consistent accuracy and alignment.",
    },

    {
        "task_type": "item elevation",
        "priority_level": "medium",
        "reward": 7.0,
        "difficulty": 5.0,
        "navigation_constraints": ["low visibility", "crowded"],
        "required_capabilities": {
            "payload": 20.0,
            "reach": 25.0,
        },
        "environmental_conditions": ["cold-resistant", "dustproof"],
        "sensors_needed": ["camera", "proximity"],
        "manipulators_needed": ["cable hoist", "gripper"],
        "communication_requirements": ["Radio", "5G"],
        "safety_protocols": ["overheat protection", "overload protection"],
        "duration": 3.0,
        "performance_metric": "safety compliance",
        "nl_description": "Lift and hold items at height; stable hoisting and precise placement are important.",
    },

    {
        "task_type": "scaffold",
        "priority_level": "medium",
        "reward": 7.0,
        "difficulty": 7.0,
        "navigation_constraints": ["narrow spaces", "low ceilings"],
        "required_capabilities": {
            "payload": 8.0,
            "reach": 8.0,
        },
        "environmental_conditions": ["shock-resistant", "wind-resistant"],
        "sensors_needed": ["LiDAR", "ultrasonic"],
        "manipulators_needed": ["welding tool", "scissor lift"],
        "communication_requirements": ["Wi-Fi", "Bluetooth"],
        "safety_protocols": ["collision avoidance", "fall detection"],
        "duration": 3.0,
        "performance_metric": "safety compliance",
        "nl_description": "Work at elevation around narrow or vertical structures; drilling and placement on frames.",
    }
]