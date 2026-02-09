from typing import List, Tuple, Optional
from enum import Enum

class CapabilityProfile:
    def __init__(self,
                 robot_id: str,
                 mobility_type: str,
                #  max_speed: float,
                 max_speed: int, # keep speed a constant integer of 1 for CBS purposes for now
                 payload_capacity: Optional[float],
                 reach: Optional[float],
                 battery_life: float,
                 size: Tuple[float, float, float],
                 environmental_resistance: Optional[List[str]],
                 sensors: Optional[List[str]],
                 sensor_range: Optional[float],
                 manipulators: Optional[List[str]],
                 communication_protocols: List[str],
                 processing_power: float,
                 autonomy_level: str,
                 special_functions: Optional[List[str]],
                 safety_features: Optional[List[str]],
                 adaptability: bool,
                 location: Tuple[float, float, float],
                 #preferred_tasks: List[str],
                 current_task: Optional["TaskDescription"],
                 remaining_distance: float,
                 time_on_task: float,
                 tasks_attempted: int,
                 tasks_successful: int,
                 current_path: list, # the path the robot is currently following, if any
                 assigned: bool, # Indicates if the robot is currently assigned to a task
                 strict_profile_name: str = "",
                 current_task_suitability: float = 0.5,
                 ): 
        self.robot_id = robot_id
        self.mobility_type = mobility_type
        self.max_speed = max_speed
        self.payload_capacity = payload_capacity
        self.reach = reach
        self.battery_life = battery_life
        self.size = size  # (length, width, height)
        self.environmental_resistance = environmental_resistance
        self.sensors = sensors
        self.sensor_range = sensor_range
        self.manipulators = manipulators
        self.communication_protocols = communication_protocols
        self.processing_power = processing_power
        self.autonomy_level = autonomy_level
        self.special_functions = special_functions
        self.safety_features = safety_features
        self.adaptability = adaptability
        self.location = location  # (x, y, z) coordinates
        #self.preferred_tasks = preferred_tasks
        self.current_task = current_task  # The task the robot is currently assigned to
        self.remaining_distance = remaining_distance  # Distance left to the current task location
        self.time_on_task = time_on_task  # Time spent on the current task
        self.tasks_attempted = tasks_attempted
        self.tasks_successful = tasks_successful
        self.current_path = current_path
        self.assigned = assigned # says if the robot is assigned to a task to replace the index lists below
        self.strict_profile_name = strict_profile_name # if the robot was generated from a strict profile, store the name here for reference
        self.current_task_suitability = current_task_suitability # store the suitability score for the current task assignment

    def __repr__(self):
        return (f"CapabilityProfile(robot_id={self.robot_id}, mobility_type={self.mobility_type}, "
                f"max_speed={self.max_speed}, payload_capacity={self.payload_capacity}, "
                f"reach={self.reach}, battery_life={self.battery_life}, size={self.size}, "
                f"environmental_resistance={self.environmental_resistance}, sensors={self.sensors}, "
                f"sensor_range={self.sensor_range}, manipulators={self.manipulators}, communication_protocols={self.communication_protocols}, "
                f"processing_power={self.processing_power}, autonomy_level={self.autonomy_level}, "
                f"special_functions={self.special_functions}, safety_features={self.safety_features}, "
                f"adaptability={self.adaptability}, location={self.location}, remaining_distance={self.remaining_distance}, time_on_task={self.time_on_task}, tasks_attempted={self.tasks_attempted}, tasks_successful={self.tasks_successful})")

class TaskDescription:
    def __init__(self,
                 task_id: str,
                 task_type: str,
                 objective: str,
                 priority_level: str,
                 reward: float,
                 difficulty: float,
                 location: Tuple[float, float, float],
                 navigation_constraints: Optional[List[str]],
                 required_capabilities: List[str],
                 time_window: Optional[Tuple[str, str]],
                 environmental_conditions: Optional[List[str]],
                 dependencies: Optional[List[str]],
                 tools_needed: Optional[List[List[str]]],
                 communication_requirements: Optional[List[str]],
                 safety_protocols: Optional[List[str]],
                 performance_metrics: List[str],
                 success_criteria: str,
                 assigned_robot: Optional["CapabilityProfile"],
                 time_to_complete: float,
                 assigned: bool,
                 strict_profile_name: str = "",
                 current_suitability: float = 0.5
                 ):
        self.task_id = task_id
        if (task_id in {"maintenance", "cleaning"}): #figure out tasks that won't have progress reset after interruption
            self.reset_progress = False 
        else:
            self.reset_progress = True
        self.task_type = task_type
        self.objective = objective
        self.priority_level = priority_level
        self.reward = reward
        self.difficulty = difficulty
        self.location = location  # (x, y, z) coordinates
        self.navigation_constraints = navigation_constraints
        self.required_capabilities = required_capabilities
        self.time_window = time_window  # (start_time, end_time) or None
        self.environmental_conditions = environmental_conditions
        self.dependencies = dependencies
        self.tools_needed = tools_needed
        self.communication_requirements = communication_requirements
        self.safety_protocols = safety_protocols
        self.performance_metrics = performance_metrics
        self.success_criteria = success_criteria
        self.assigned_robot = assigned_robot  # The robot currently assigned to this task
        self.time_to_complete = time_to_complete  # Total time required to complete the task
        self.time_left = time_to_complete # Counter used during the step method, resets back to time_to_complete based on reset_progress
        self.assigned = assigned
        self.strict_profile_name = strict_profile_name # if the task was generated from a strict profile, store the name here for reference

    def __repr__(self):
        return (f"TaskDescription(task_id={self.task_id}, task_type={self.task_type}, "
                f"objective={self.objective}, priority_level={self.priority_level}, "
                f"location={self.location}, reward={self.reward}, difficulty={self.difficulty}, navigation_constraints={self.navigation_constraints}, "
                f"required_capabilities={self.required_capabilities}, time_window={self.time_window}, "
                f"environmental_conditions={self.environmental_conditions}, "
                f"dependencies={self.dependencies}, tools_needed={self.tools_needed}, "
                f"communication_requirements={self.communication_requirements}, "
                f"safety_protocols={self.safety_protocols}, performance_metrics={self.performance_metrics}, "
                f"success_criteria={self.success_criteria}, time_to_complete={self.time_to_complete}), time_left={self.time_left}) ")
    
# CATEGORIES
class AutonomyLevel(Enum):
    TELEOPERATED = "teleoperated"
    SEMI_AUTONOMOUS = "semi-autonomous"
    FULLY_AUTONOMOUS = "fully autonomous"

class TaskTypes(Enum):
    DELIVERY = "delivery"
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"
    ASSEMBLY = "assembly"
    SURVEYING = "surveying"
    DATA_COLLECTION = "data collection"
    ASSISTANCE = "assistance"

class Priorities(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class PerformanceMetrics(Enum):
    TIME_TAKEN = "time taken"
    ACCURACY = "accuracy"
    ENERGY_CONSUMPTION = "energy consumption"
    SAFETY_COMPLIANCE = "safety compliance"
    COMPLETION_RATE = "completion rate"
