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
STRICT_ROBOT_PROFILES = [
    {
        "name": "delivery",
        "mobility_type": "aerial",
        "environmental_resistance": ["weatherproof", "heat-resistant", "cold-resistant"],
        "sensors": ["LiDAR", "GPS", "proximity sensor", "camera"],
        "manipulators": ["gripper"],
        "communication_protocols": ["Wi-Fi", "4G"],
        "special_functions": ["object recognition", "object tracking", "facial recognition"],
        "safety_features": ["obstacle detection", "emergency stop"],
        "sensor_range": 25.0,
        "processing_power": 5.0,
        "autonomy_level": "fully autonomous",
        "payload_capacity": 20.0,
        "reach": 0.0,
        "battery_life": 80.0,
        "size": (2.0, 2.0, 2.0),
        "adaptability": False
    },
    {
        "name": "assembly",
        "mobility_type": "hovering",
        "environmental_resistance": ["dustproof", "heat-resistant", "shock-resistant"],
        "sensors": ["camera", "proximity sensor", "infrared"],
        "manipulators": ["gripper", "drill", "dispenser", "welding tool"],
        "communication_protocols": ["Wi-Fi", "Radio"],
        "special_functions": ["object recognition", "object tracking", "gesture recognition", "precise alignment"],
        "safety_features": ["collision avoidance", "overheat protection", "emergency stop"],
        "sensor_range": 10.0,
        "processing_power": 7.0,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,
        "reach": 5.0,
        "battery_life": 75.0,
        "size": (3.0, 2.0, 2.0),
        "adaptability": True
    },
    {
        "name": "excavator",
        "mobility_type": "tracked",
        "environmental_resistance": ["dustproof", "shock-resistant", "weatherproof"],
        "sensors": ["LiDAR", "camera", "proximity sensor", "magnetometer"],
        "manipulators": ["hydraulic bucket", "cable hoist"],
        "communication_protocols": ["Wi-Fi", "Radio"],
        "special_functions": ["terrain leveling", "object recognition"],
        "safety_features": ["collision avoidance", "overload protection", "emergency stop"],
        "sensor_range": 15.0,
        "processing_power": 8.0,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,
        "reach": 8.0,
        "battery_life": 75.0,
        "size": (4.2, 2.5, 3.0),
        "adaptability": True
    },
    {
        "name": "bricklayer",
        "mobility_type": "wheeled",
        "environmental_resistance": ["dustproof", "waterproof", "shock-resistant"],
        "sensors": ["infrared", "camera", "proximity sensor"],
        "manipulators": ["mixer drum", "dispenser", "gripper", "cable hoist"],
        "communication_protocols": ["Wi-Fi", "Radio"],
        "special_functions": ["precise alignment", "concrete mixing"],
        "safety_features": ["obstacle detection", "emergency stop", "collision avoidance"],
        "sensor_range": 10.0,
        "processing_power": 5.5,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 20.0,  
        "reach": 5.5,
        "battery_life": 80.0,
        "size": (4.0, 2.0, 2.0),
        "adaptability": False
    },
    {
        "name": "crane",
        "mobility_type": "tracked", 
        "environmental_resistance": ["weatherproof", "wind-resistant", "dustproof"],
        "sensors": ["camera", "GPS", "proximity sensor", "ultrasonic"],
        "manipulators": ["gripper", "cable hoist"],
        "communication_protocols": ["5G", "Radio"],
        "special_functions": ["precise alignment", "object tracking", "object detection"],
        "safety_features": ["overload protection", "balance control", "collision avoidance", "emergency stop"],
        "sensor_range": 30.0,
        "processing_power": 6.5,
        "autonomy_level": "semi-autonomous",
        "payload_capacity": 35.0,
        "reach": 30.0,
        "battery_life": 70.0,
        "size": (3.5, 3.5, 10.0),
        "adaptability": False
    },
    {
        "name": "scaffolding",
        "mobility_type": "climbing",
        "environmental_resistance": ["dustproof", "shock-resistant", "wind-resistant"],
        "sensors": ["LiDAR", "camera", "ultrasonic", "proximity sensor"],
        "manipulators": ["gripper", "drill", "welding tool", "cable hoist"],
        "communication_protocols": ["Wi-Fi", "Bluetooth"],
        "special_functions": ["precise alignment", "balance control"],
        "safety_features": ["fall detection", "collision avoidance", "emergency stop"],
        "sensor_range": 12.0,
        "processing_power": 7.0,
        "autonomy_level": "fully autonomous",
        "payload_capacity": 20.0,  
        "reach": 10.0,  
        "battery_life": 65.0,
        "size": (1.5, 1.0, 2.0),
        "adaptability": True
    }
]

# STRICT TASK PROFILES FOR TASK GENERATION
STRICT_TASK_PROFILES = [
    {
        "task_type": "utilities", #gripper
        "priority_level": "medium",
        "reward": 6,
        "difficulty": 6,
        "navigation_constraints": ["uneven floors", "loose debris"],
        "required_capabilities": {
            "payload": 0.0,
            "reach": 3.0,
        },
        "environmental_conditions": ["shock-resistance"],
        "sensors_needed": ["LiDAR", "camera", "proximity sensor"],
        "manipulators_needed": ["gripper"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": 3,
        "performance_metric": "safety compliance",
        "nl_description": "Operate valves, switches, and panels; carry small parts; precise manipulation in plant rooms."
    },


    {
        "task_type": "utilities", #bucket
        "priority_level": "medium",
        "reward": 6,
        "difficulty": 6,
        "navigation_constraints": ["uneven floors", "loose debris"],
        "required_capabilities": {
            "payload": 10.0,
            "reach": 0.0,
        },
        "environmental_conditions": ["dustproof"],
        "sensors_needed": ["LiDAR", "camera", "proximity sensor"],
        "manipulators_needed": ["hydraulic bucket"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": 3,
        "performance_metric": "safety compliance",
        "nl_description": "Transport construction materials, clear or move small bulk materials near utility corridors; load/unload with a bucket."
    },


    {
        "task_type": "debris", #gripper
        "priority_level": "medium",
        "reward": 4,
        "difficulty": 4,
        "navigation_constraints": ["loose debris", "crowded"],
        "required_capabilities": {
            "payload": 10.0,
            "reach": 2.0,
        },
        "environmental_conditions": ["weatherproof"],
        "sensors_needed": ["LiDAR", "camera", "ultrasonic", "proximity sensor"],
        "manipulators_needed": ["gripper"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": 2,
        "performance_metric": "safety compliance",
        "nl_description": "Pick and remove scattered debris in cluttered passages; careful grasping and placement."
    },


    {
        "task_type": "debris", #bucket
        "priority_level": "medium",
        "reward": 4,
        "difficulty": 4,
        "navigation_constraints": ["loose debris", "uneven floors"],
        "required_capabilities": {
            "payload": 10.0,
            "reach": 2.0,
        },
        "environmental_conditions": ["weatherproof"],
        "sensors_needed": ["LiDAR", "camera", "ultrasonic", "proximity sensor"],
        "manipulators_needed": ["hydraulic bucket"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": 2,
        "performance_metric": "safety compliance",
        "nl_description": "Scoop and relocate piles of loose debris; continuous removal in uneven terrain.",
        
    },


    {
        "task_type": "delivery",
        "priority_level": "low",
        "reward": 2,
        "difficulty": 2,
        "navigation_constraints": ["crowded", "elevator"],
        "required_capabilities": {
            "payload": 1.0,
            "reach": 0.0,
        },
        "environmental_conditions": ["weatherproof", "dustproof"],
        "sensors_needed": ["camera", "proximity sensor", "GPS"],
        "manipulators_needed": ["gripper"],
        "communication_requirements": ["Wi-Fi", "4G"],
        "safety_protocols": ["obstacle detection", "emergency stop"],
        "duration": 1,
        "performance_metric": "time taken",
        "nl_description": "Fetch-and-carry small payloads point-to-point through indoor/outdoor corridors.",
    },


    {
        "task_type": "assembly",
        "priority_level": "high",
        "reward": 8,
        "difficulty": 8,
        "navigation_constraints": ["crowded"],
        "required_capabilities": {
            "payload": 5.0,
            "reach": 0.0,
        },
        "environmental_conditions": ["heat-resistant"],
        "sensors_needed": ["camera", "infrared", "dispenser"],
        "manipulators_needed": ["gripper", "drill", "welding tool"],
        "communication_requirements": ["Wi-Fi", "Radio"],
        "safety_protocols": ["collision avoidance", "emergency stop"],
        "duration": 4,
        "performance_metric": "accuracy",
        "nl_description": "Fixture placement, fastening, dispensing, or welding with moderate precision in crowded areas.",
    },


    {
        "task_type": "excavate",
        "priority_level": "high",
        "reward": 8,
        "difficulty": 8,
        "navigation_constraints": ["loose debris", "low visibility"],
        "required_capabilities": {
            "payload": 15.0,
            "reach": 2.0,
        },
        "environmental_conditions": ["dustproof", "shock-resistant"],
        "sensors_needed": ["LiDAR", "camera", "proximity sensor"],
        "manipulators_needed": ["hydraulic bucket"],
        "communication_requirements": ["Radio"],
        "safety_protocols": ["overload protection", "obstacle detection"],
        "duration": 4,
        "performance_metric": "safety compliance",
        "nl_description": "Dig, trench, or remove soil/rubble; sustained scooping with high payload demands.",
    },


    {
        "task_type": "item elevation",
        "priority_level": "medium",
        "reward": 5,
        "difficulty": 5,
        "navigation_constraints": ["low visibility", "crowded"],
        "required_capabilities": {
            "payload": 2.0,
            "reach": 10.0,
        },
        "environmental_conditions": ["wind-resistant"],
        "sensors_needed": ["camera", "GPS", "proximity sensor", "ultrasonic"],
        "manipulators_needed": ["cable hoist", "gripper"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "emergency stop"],
        "duration": 3,
        "performance_metric": "safety compliance",
        "nl_description": "Lift and hold items at height; stable hoisting and precise placement are important.",
    },


    {
        "task_type": "lay bricks",
        "priority_level": "low",
        "reward": 6,
        "difficulty": 6,
        "navigation_constraints": ["crowded", "windy"],
        "required_capabilities": {
            "payload": 4.0,
            "reach": 4.0,
        },
        "environmental_conditions": ["dustproof", "weatherproof"],
        "sensors_needed": ["LiDAR", "camera", "proximity sensor"],
        "manipulators_needed": ["gripper", "dispenser"],
        "communication_requirements": ["Wi-Fi"],
        "safety_protocols": ["collision avoidance", "overheat protection"],
        "duration": 3,
        "performance_metric": "accuracy",
        "nl_description": "Pick, mortar/dispense, and place bricks with consistent accuracy and alignment.",
    },


    {
        "task_type": "scaffold",
        "priority_level": "medium",
        "reward": 7,
        "difficulty": 7,
        "navigation_constraints": ["narrow spaces", "low ceilings"],
        "required_capabilities": {
            "payload": 4.0,
            "reach": 6.0,
        },
        "environmental_conditions": ["weatherproof"],
        "sensors_needed": ["LiDAR", "camera", "ultrasonic", "proximity sensor"],
        "manipulators_needed": ["gripper", "drill"],
        "communication_requirements": ["Radio", "Wi-Fi"],
        "safety_protocols": ["overload protection", "balance control", "emergency stop"],
        "duration": 3,
        "performance_metric": "safety compliance",
        "nl_description": "Work at elevation around narrow or vertical structures; drilling and placement on frames.",
    }
]