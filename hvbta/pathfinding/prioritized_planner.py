"""
Prioritized Planning engine behind the NATIVE-FULL-PLAN contract.

    plan(agents, map_dict) -> {robot_id: [(x,y), ...]} | None

Stub (Phase 4 wiring). Real implementation (Phase 8): space-time A* per agent
in priority order, treating already-planned paths as time-indexed obstacles.
Reuse Astar.py if its interface allows a time dimension; otherwise a small
space-time A* here.

`agents` is already augmented with stationary blockers by _prepare_agents; do
not augment again. Return None/{} on failure.
"""


def plan(agents, map_dict):
    raise NotImplementedError("prioritized_planner.plan not implemented yet (Phase 8).")
