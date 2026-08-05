"""
CBS behind the NATIVE-FULL-PLAN engine contract.

    plan(agents, map_dict) -> {robot_id: [(x,y), ...]} | None

`agents` is the list already augmented with stationary blockers by
planner_common._prepare_agents -- so this adapter must NOT augment again.
It only translates the agent dicts into the shape Environment expects and
runs CBS.

Design note (parity): the old run_cbs called build_cbs_agents(robots, ...)
which filtered with `r.assigned and r.current_task is not None`, so its
stationary-blocker augmentation was a no-op -- unassigned idle robots never
reached Environment. This adapter deliberately passes ALL agents (movers +
stationary blockers) to Environment so CBS routes movers around idle robots,
matching what PP / PIBT / LaCAM get via _prepare_agents. That is the intended
new behavior; the Phase 6 parity gate is split accordingly (exact-path parity
only on no-idle instances; collision-free w.r.t. idle cells on idle instances).
"""

from hvbta.pathfinding.Final_CBS import CBS, Environment


def plan(agents, map_dict):
    """
    agents: list of {"id":rid, "start":(x,y), "goal":(x,y)} (already augmented
            with stationary blockers; do not augment again here).
    map_dict: {"dimension":(rows,cols), "obstacles":[(r,c), ...]}
    Returns {robot_id: [(x,y), ...]} or None on failure.
    """
    # Translate {"id","start","goal"} -> {"name","start","goal"} for Environment.
    cbs_agents = [
        {"name": a["id"], "start": a["start"], "goal": a["goal"]}
        for a in agents
    ]

    env = Environment(
        dimension=map_dict["dimension"],
        agents=cbs_agents,
        obstacles=map_dict["obstacles"],
    )
    res = CBS(env).search()
    # CBS.search() returns {} on failure, (plan, nodes, conflicts) on success.
    if not res:
        return None

    solution, _nodes_expanded, _conflicts = res
    # solution: {agent_name: [{'t':..,'x':..,'y':..}, ...]}
    paths = {
        agent_name: [(p["x"], p["y"]) for p in schedule]
        for agent_name, schedule in solution.items()
    }
    return paths
