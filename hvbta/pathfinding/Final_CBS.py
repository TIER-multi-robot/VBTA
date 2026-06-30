import heapq
import sys
sys.path.insert(0, '../')
import argparse
import yaml
from math import fabs
from itertools import combinations
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from hvbta.pathfinding.a_star import AStar

def solve_for_agent(args: tuple) -> tuple:
    """
    A top-level worker function to find a path for a single agent.
    This function is designed to be called by a ProcessPoolExecutor.
    
    Parameters:
        args (tuple): A tuple containing the following elements:
            - agent_name (str): The name of the agent
            - agent_dict (dict): A dictionary containing agent information
            - dimension (tuple): The dimensions of the environment (width, height)
            - obstacles (list): A list of obstacle locations
            - constraints (list): A list of constraints for the agent

    Returns:
        tuple: A tuple containing the agent's name and its path
    """
    # Unpack all the arguments
    agent_name, agent_dict, dimension, obstacles, constraints = args

    # Create a temporary environment for this specific task
    # Pass data, not the complex stateful CBS environment object
    temp_env = Environment(dimension, [], obstacles) # Agents list can be empty
    temp_env.agent_dict = {agent_name: agent_dict[agent_name]}
    temp_env.a_star.agent_dict = temp_env.agent_dict
    temp_env.constraints = constraints
    
    # The A* search is now run within this isolated process
    path = temp_env.a_star.search(agent_name)
    
    # Return the agent's name along with its path
    return agent_name, path

class Location(object):
    """Location class, represents agent location in environment as an (x, y) coordinate tuple"""
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
    def __eq__(self, other: 'Location'):
        return self.x == other.x and self.y == other.y
    def __str__(self):
        return str((self.x, self.y))

class State(object):
    """
    State class, represents the state of the agent as its time step and location
    location is a Location object
    time is an int starting at 0
    """
    def __init__(self, time: int, location: Location):
        self.time = time
        self.location = location
    def __eq__(self, other: 'State'):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time) + str(self.location.x) + str(self.location.y))
    def is_equal_except_time(self, state: 'State') -> bool:
        """Check if two states are equal based on location only, ignoring time
         Parameters:
            state (State): The other state to compare with.
        Returns:
            bool: True if the locations are the same, False otherwise.
        """
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location.x, self.location.y))

class Conflict(object):
    """Single Rule, Conflict class differentiates between vertex and edge conflicts"""
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint(object):
    """Single Rule, Vertex constraint class, imposes vertex constraints on high level search nodes at a specified time and location"""
    def __init__(self, time: int, location: Location):
        self.time = time
        self.location = location

    def __eq__(self, other: 'VertexConstraint'):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    """Single Rule, Edge constraint class, imposes edge constraints on high level search nodes at a specified time and location, has 2 locations as opposed to vertex constraints (needs both verticies)"""
    def __init__(self, time: int, location_1: Location, location_2: Location):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other: 'EdgeConstraint'):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints(object):
    """Tracker of all constraints, one for each agent, Base constraint class, keeps track of sets of constraints and can add new ones as needed"""
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other: 'Constraints'):
        """Add new constraints to existing set of constraints
        Parameters:
            other (Constraints): The other Constraints object to merge with this one.
        """
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

class Environment(object):
    """Environment class represents the navigating environment"""
    def __init__(self, dimension: tuple[int, int], agents: list[dict], obstacles: list[tuple[int, int]]):
        """
        Initialize the environment with dimensions, agents, and obstacles.
        Parameters:
            dimension (tuple): A tuple representing the dimensions of the environment (width, height).
            agents (list): A list of dictionaries, each containing 'name', 'start', and 'goal' keys for an agent.
            obstacles (list): A list of tuples representing the coordinates of obstacles in the environment (x, y).
        """
        self.dimension = dimension
        self.obstacles = obstacles

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

        self.a_star = AStar(self)

    def get_neighbors(self, state: State) -> list[State]:
        """Get all valid neighboring states from the current state
        Parameters:
            state (State): The current state of the agent.
        Returns:
            neighbors (list): A list of valid neighboring State objects."""
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Diagonal actions
        # Up/Right
        # n = State(state.time + 1, Location(state.location.x+1, state.location.y+1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)
        # # Up/Left
        # n = State(state.time + 1, Location(state.location.x-1, state.location.y+1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)
        # # Down/Right
        # n = State(state.time + 1, Location(state.location.x+1, state.location.y-1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)
        # # Down/Left
        # n = State(state.time + 1, Location(state.location.x-1, state.location.y-1))
        # if self.state_valid(n) and self.transition_valid(state, n):
        #     neighbors.append(n)

        return neighbors


    def get_first_conflict(self, solution: dict) -> Conflict | bool:
        """Check for conflicts"""
        max_time = max([len(plan) for plan in solution.values()])
        result = Conflict()
        for t in range(max_time):
            # Check for vertex conflicts
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1 = self.get_state(agent_1, solution, t)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2):
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            # Check for edge conflicts
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False

    def create_constraints_from_conflict(self, conflict: Conflict) -> dict:
        """Create a new constraint dictionary based on a conflict
        
        Parameters:
            conflict (Conflict): The conflict to create constraints from.
        Returns:
            constraint_dict (dict): A dictionary mapping agent names to their respective Constraints objects.
        """
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name: str, solution: dict, t: int) -> State:
        """Get the state of an agent at any timestep t if it exists, otherwise return the last state
        Parameters:
            agent_name (str): The name of the agent.
            solution (dict): The solution dictionary mapping agent names to their paths.
            t (int): The timestep to get the state for.
        Returns:
            State: The state of the agent at timestep t or the last state if t exceeds the path length.
        """
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state: State) -> bool:
        """
        check location.x is within bounds and location.y is 
        within bounds and we are not violating a vertex constraint or in an obstacle
        Parameters:
            state (State): The state to validate.
        Returns:
            bool: True if the state is valid, False otherwise.
        """
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints \
            and (state.location.x, state.location.y) not in self.obstacles

    def transition_valid(self, state_1: State, state_2: State) -> bool:
        """
        check that the transition did not violate an edge constraint
        Parameters:
            state_1 (State): The starting state of the transition.
            state_2 (State): The ending state of the transition.
        Returns:
            bool: True if the transition is valid, False otherwise.
        """
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints


    def admissible_heuristic(self, state: State, agent_name: str) -> float:
        """
        Distance heuristic
        Parameters:
            state (State): The current state of the agent.
            agent_name (str): The name of the agent.
        Returns:
            float: The heuristic value (Manhattan distance to the goal).
        """
        goal = self.agent_dict[agent_name]["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)


    def is_at_goal(self, state: State, agent_name: str) -> bool:
        """
        Check if we are at the goal position
        Parameters:
            state (State): The current state of the agent.
            agent_name (str): The name of the agent.
        Returns:
            bool: True if the agent is at its goal location, False otherwise.
        """
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        """Create a new agent dictionary with 0 time and start and goal locations"""
        for agent in self.agents:
            start_state = State(0, Location(agent['start'][0], agent['start'][1]))
            goal_state = State(0, Location(agent['goal'][0], agent['goal'][1]))

            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    # def compute_solution(self):
    #     """Find the solution"""
    #     solution = {}
    #     for agent in self.agent_dict.keys():
    #         # set each agents constraint dictionary to the default empty constraint class
    #         self.constraints = self.constraint_dict.setdefault(agent, Constraints())
    #         # find local solution for agent
    #         print(f"A STAR PLANNING FOR AGENT {agent}")
    #         local_solution = self.a_star.search(agent)
    #         if not local_solution:
    #             return False
    #         solution.update({agent:local_solution})
    #     return solution

    def compute_solution(self):
        """
        Finds the solution in parallel for each agent using ProcessPoolExecutor
        Returns:
            solution (dict): A dictionary mapping agent names to their paths, or False if no solution is found.
        """
        solution = {}
        tasks = []

        for agent_name in self.agent_dict.keys():
            agent_constraints = self.constraint_dict.get(agent_name, Constraints())
            task_args = (
                agent_name,
                self.agent_dict,
                self.dimension,
                self.obstacles,
                agent_constraints
            )
            tasks.append(task_args)

        with ProcessPoolExecutor() as executor:
            results = executor.map(solve_for_agent, tasks)

            for agent_name, path in results:
                if not path:
                    return False
                solution[agent_name] = path

        return solution

    def compute_solution_cost(self, solution: dict) -> int:
        """
        compute total solution cost
        Parameters:
            solution (dict): A dictionary mapping agent names to their paths.
        Returns:
            int: The total cost of the solution (sum of path lengths for all agents).
        """
        return sum([len(path) for path in solution.values()])

class HighLevelNode(object):
    """a CBS node that contains a solution, constraint dictionary, and solution cost for a single agent"""
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    # def __eq__(self, other):
    #     """Check for equivalent nodes"""
    #     if not isinstance(other, type(self)): return NotImplemented
    #     return self.solution == other.solution and self.cost == other.cost

    def __eq__(self, other):
        """Check for equivalent nodes based on constraints"""
        if not isinstance(other, type(self)): return NotImplemented
        return self.constraint_dict_to_tuple(self.constraint_dict) == self.constraint_dict_to_tuple(other.constraint_dict)
    
    def __hash__(self):
        """Hash based on constraints"""
        return hash(self.constraint_dict_to_tuple(self.constraint_dict))
    
    def constraint_dict_to_tuple(self, constraint_dict: dict) -> tuple:
        """
        Convert constraint dictionary to a hashable tuple
        Parameters:
            constraint_dict (dict): A dictionary mapping agent names to their Constraints objects.
        Returns:
            frozen_constraints (tuple): A hashable representation of the constraint dictionary.
        """
        frozen_constraints = set()
        for agent, constraints in constraint_dict.items():
            vc_tuples = frozenset((vc.time, vc.location.x, vc.location.y) for vc in constraints.vertex_constraints)
            ec_tuples = frozenset((ec.time, ec.location_1.x, ec.location_1.y, ec.location_2.x, ec.location_2.y) for ec in constraints.edge_constraints)
            frozen_constraints.add((agent, vc_tuples, ec_tuples))
        return frozenset(frozen_constraints)

    # def __hash__(self):
    #     return hash((self.cost))

    def __lt__(self, other):
        """Compare costs of node solutions"""
        return self.cost < other.cost

class CBS(object):
    """CBS search class"""
    def __init__(self, environment: Environment):
        self.env = environment
        # self.open_set = set()
        self.open_set = []
        self.closed_set = set()
    def search(self):
        """Perform CBS search"""
        start = HighLevelNode()
        # print(f"\n\n\n COMPUTING INITIAL SOLUTION\n\n\n")
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            # INITIAL PROBLEM Every agent starts with an empty constraint set
            start.constraint_dict[agent] = Constraints()
        # compute solution
        start.solution = self.env.compute_solution()
        if not start.solution:
            return {}
        # compute cost of solution
        start.cost = self.env.compute_solution_cost(start.solution)

        # count conflicts resolved for a difficulty metric
        total_conflicts = 0

        # set open_set to the HighLevelNode
        # self.open_set |= {start}
        # Push the start node onto the heap
        heapq.heappush(self.open_set, start)
        count = 0

        # Search open_set starting with the lowest cost nodes first, adding them to closed_set as we go
        while self.open_set:
            # print(f"\n\n\n EXPANDING HIGH LEVEL NODE {count} \n\n\n")
            # compares nodes based on cost of their solutions (uses __lt__ in HighLevelNode)
            # P = min(self.open_set)
            # self.open_set -= {P}
            # self.closed_set |= {P}

            # Pop the node with the smallest cost from the heap
            P = heapq.heappop(self.open_set)
            self.closed_set.add(P)
            
            # TEMP WORKING COPY OF CONSTRAINT DICTIONARY to solve solutions
            # each high level node has its own permanent constraint dictionary
            # set the environment constraint dictionary to the current node's constraint dictionary
            # now the environment is configured with the specific set of rules required for this branch of the search tree
            self.env.constraint_dict = P.constraint_dict
            # print(f"GETTING FIRST CONFLICT")
            # find first conflict
            conflict_dict = self.env.get_first_conflict(P.solution)
            # print(f"GOT FIRST CONFLICT")

            if not conflict_dict:
                # print(f"Solution found after expanding {len(self.closed_set)} high level nodes")
                # print(f"Total conflicts identified {total_conflicts}")
                return self.generate_plan(P.solution), len(self.closed_set), total_conflicts
            
            total_conflicts += 1
            # print(f"CREATING CONSTRAINTS FROM CONFLICT")
            # create new constraints from conflict to avoid it in future solutions
            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            # create a new node for each agent in the conflict with the new constraints added
            for agent in constraint_dict.keys():
                # print(f"ADDING CONSTRAINTS FOR AGENT {agent}")
                # copy entire parent node (all constraints and solution)
                new_node = deepcopy(P)
                # add new constraint to the specific agent
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                # Check if this new configuration of constraints has been seen before
                if new_node in self.closed_set:
                    continue

                self.env.constraint_dict = new_node.constraint_dict
                # print(f"RECOMPUTING SOLUTION FOR AGENT {agent}")
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                # Push the new node onto the heap
                heapq.heappush(self.open_set, new_node)
            count += 1

        return {}
        #     if conflict_dict:
        #         total_conflicts += 1
        #     else:
        #         print(f"Solution found after expanding {len(self.closed_set)} high level nodes")
        #         print(f"Total conflicts identified {total_conflicts}")

        #         return self.generate_plan(P.solution), len(self.closed_set), total_conflicts

        #     constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

        #     for agent in constraint_dict.keys():
        #         new_node = deepcopy(P)
        #         new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

        #         self.env.constraint_dict = new_node.constraint_dict
        #         new_node.solution = self.env.compute_solution()
        #         if not new_node.solution:
        #             continue
        #         new_node.cost = self.env.compute_solution_cost(new_node.solution)

        #         if new_node not in self.closed_set:
        #             self.open_set |= {new_node}

        # return {}

    def generate_plan(self, solution: dict) -> dict:
        """
        Generate the final plan from the solution dictionary.
        Parameters:
            solution (dict): A dictionary mapping agent names to their paths.
        Returns:
            plan (dict): A dictionary mapping agent names to their movement schedules.
        """
        plan = {}
        for agent, path in solution.items():
            # dictionary for output
            path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path]
            plan[agent] = path_dict_list
        return plan


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    args = parser.parse_args(args)

    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']

    env = Environment(dimension, agents, obstacles)

    # Searching
    cbs = CBS(env)
    solution_data = cbs.search()
    if solution_data:
        solution, nodes_expanded, total_conflicts = solution_data
        print("Solution found")
    
        # Calculate makespan
        if solution:
            makespan = 0
            # NOTE: may have to iterate thru agents then paths
            for agent, path in solution.items():
                if path:
                    makespan = max(makespan, path[-1]['t'])


        # Write to output file
        output = dict()
        # agent movement schedule
        output["schedule"] = solution

        # SOLUTION METRICS
        # Sum of costs, indicates solution quality and correlates with ovferal difficulty
        output["cost"] = env.compute_solution_cost(solution)
        # Makespan indicates last agent to reach goal, indicates congestion or bottlenecks
        output["makespan"] = makespan

        # SEARCH EFFORT METRICS
        # high level nodes expanded indicates that the high-level search had to explore more possibilities in the constraint tree to find a conflict-free solution
        output["high_level_nodes_expanded"] = nodes_expanded
        # total number of conflicts found across all high-level nodes that were checked, each leading to a branching decision in the constraint tree. More conflicts generally implies a more complex interaction between agents
        output["num_of_conflicts_identified"] = total_conflicts
    else:
        print("Solution not found")

    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)


if __name__ == "__main__":
    main()
