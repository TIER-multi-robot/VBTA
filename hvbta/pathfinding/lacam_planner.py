"""
LaCAM engine behind the NATIVE-FULL-PLAN contract.

    plan(agents, map_dict) -> {robot_id: [(x,y), ...]} | None

Stub (Phase 4 wiring). Real implementation (Phase 8, optional/highest effort):
configuration-space search using pibt_planner.step as the successor generator;
complete; returns full plans natively. This is the strongest-for-the-paper
option.

`agents` is already augmented with stationary blockers by _prepare_agents; do
not augment again. The real impl will `from hvbta.pathfinding.pibt_planner
import step` as its successor generator. Return None/{} on failure.
"""


def plan(agents, map_dict):
    raise NotImplementedError("lacam_planner.plan not implemented yet (Phase 8).")
