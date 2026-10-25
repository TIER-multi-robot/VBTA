import random
from typing import List, Tuple
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.allocators.misc_assignment import unassign_task_from_robot
from hvbta.suitability import navigation_suitability

# Master switch for the safety-net nudge (the idle-yield swap is GONE:
# the two-phase timestep makes it unnecessary - all movement is
# synchronous and jointly planned, so there is no sequential processing
# window for a mover to step onto an idle robot). The nudge remains as
# a counted last resort for edge cases; the target is zero fires.
COLLISION_AVOIDANCE = True


def simulate_time_step(
    robots: List[CapabilityProfile],
    tasks: List[TaskDescription],
    unassigned_robots: List[str],
    unassigned_tasks: List[str],
    suitability_method: str,
    occupied_locations: set,
    start_positions: dict,
    goal_positions: dict,
    time_step: float = 1.0,
    total_reward: float = 0.0,
    total_success: int = 0,
    po_state=None,
    current_tick: int = None,
    pending_event_steps: List[int] = None,
    obstacles: frozenset = None,
    collision_counters: dict = None
) -> Tuple[int, int, float, int]:
    """
    TWO-PHASE synchronous timestep with transitive waiting. Movement and
    status changes are strictly separated so no robot's mid-timestep state
    change can affect another robot's movement within the same tick.

    Phase 1: compute every robot's next cell from its current path
             (robots without a path stay put).
    Phase 1.5: transitive waiting. Robots without a usable path (newly
             assigned after a failed replan, newly arrived, unassigned
             mid-route) stay put; any robot whose next cell is occupied
             by a staying robot also stays put, repeated to a fixed
             point. Old paths are jointly conflict-free and stayers never
             move, so no vertex or swap conflict can occur between the
             survivors. Forced waits are counted in propagated_waits.
    Phase 2: check the full next-position set for vertex conflicts (two
              robots in one cell) and swap conflicts (two robots exchanging
              cells). Jointly planned paths should produce none; any that
              appear are from stale paths and are logged in detail, counted
              in Planned_Conflict_Count - NOT patched.
    Phase 3: move every robot at once. A robot consumes a step of its
              path when it ENDS the tick on the step's cell - moving
              onto it OR planning a wait there. A forced (propagated)
              wait consumes nothing: the robot retries the same step next
              tick.
    Phase 4: apply status changes (task progress and completions, battery
              drain and deaths, reveals and mis-assignments). A robot that
              stops ends the tick at its current cell; the next tick's
              top-of-tick replan includes it before anyone moves. Work
              requires the robot to be physically AT the task location -
              path bookkeeping alone never teleports a robot onto a task.
    Phase 5: safety nudge (last resort, should never fire).

    Returns:
        (tasks_completed, count, total_reward, total_success)
    """
    unassigned_count = 0
    tasks_completed = 0
    _start_locs = {r.robot_id: r.location for r in robots}

    # ================================================================
    # Phase 1: compute every robot's next cell from its current path.
    # ================================================================
    next_positions = {}
    for robot in robots:
        if robot.current_path and len(robot.current_path) > 1:
            next_positions[robot.robot_id] = robot.current_path[1]
        else:
            next_positions[robot.robot_id] = robot.location

    # ================================================================
    # Phase 1.5: transitive waiting (v7 stale-path safety). Seed: every
    # robot that stays put this tick (no usable path, path exhausted, or
    # a planned wait). Propagate: any robot whose next cell is occupied
    # by a staying robot also stays put, to a fixed point. Survivors move
    # on old paths that are jointly conflict-free, and stayers never
    # move, so no mover-stayer vertex or swap conflict is possible.
    # Forced waits (a robot that intended to move but must hold) are
    # counted in propagated_waits.
    # ================================================================
    loc_by_id = {r.robot_id: r.location for r in robots}
    staying = set()
    for rid, pos in next_positions.items():
        if pos == loc_by_id[rid]:
            staying.add(rid)
    stay_cells = {loc_by_id[rid] for rid in staying}
    changed = True
    while changed:
        changed = False
        for rid, pos in next_positions.items():
            if rid in staying:
                continue
            if pos in stay_cells:
                staying.add(rid)
                stay_cells.add(loc_by_id[rid])
                changed = True
    forced_waits = 0
    for rid in staying:
        if next_positions[rid] != loc_by_id[rid]:
            next_positions[rid] = loc_by_id[rid]
            forced_waits += 1
    if collision_counters is not None and forced_waits:
        collision_counters["propagated_waits"] = (
            collision_counters.get("propagated_waits", 0) + forced_waits)

    # ================================================================
    # Phase 2: check the full next-position set for vertex and swap
    # conflicts. Jointly planned paths should produce none. Any that
    # appear are logged in detail (NOT patched) and counted.
    # ================================================================
    if collision_counters is not None:
        planned = []
        # vertex conflicts: two robots moving to (or staying in) the same cell
        by_next = {}
        for rid, pos in next_positions.items():
            by_next.setdefault(pos, []).append(rid)
        for pos, rids in sorted(by_next.items()):
            if len(rids) > 1:
                planned.append({
                    "type": "vertex", "cell": pos, "robots": sorted(rids),
                    "tick": current_tick,
                })
        # swap conflicts: A at X → Y while B at Y → X
        ids = sorted(next_positions)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if (next_positions[a] == _start_locs.get(b)
                        and next_positions[b] == _start_locs.get(a)
                        and next_positions[a] != next_positions[b]):
                    planned.append({
                        "type": "swap", "robots": [a, b],
                        "cells": [next_positions[a], next_positions[b]],
                        "tick": current_tick,
                    })
        if planned:
            collision_counters["planned_conflicts"] += len(planned)
            for pc in planned:
                print(f"  PLANNED CONFLICT {pc}")

    # ================================================================
    # Phase 3: move every robot at once. A path step is consumed when the
    # robot ends the tick on that step's cell: moving onto it OR the step
    # being a planned wait (path[1] == own location). A propagated
    # (forced) wait consumes NOTHING - the robot retries the same step
    # next tick, so its path always starts at its physical location.
    # ================================================================
    for robot in robots:
        rid = robot.robot_id
        next_pos = next_positions[rid]
        if next_pos != robot.location:
            occupied_locations.discard(robot.location)
            robot.location = next_pos
            occupied_locations.add(next_pos)
            if robot.assigned:
                start_positions[rid] = next_pos
            if po_state is not None and robot.assigned:
                po_state.on_move(robot)
        if (robot.current_path and len(robot.current_path) > 1
                and robot.current_path[1] == next_pos):
            robot.current_path.pop(0)
            robot.remaining_distance = max(0, len(robot.current_path) - 1)

    # ================================================================
    # Phase 4: apply status changes. All robots have already moved; this
    # only processes work, completions, battery drain, and unassignments.
    # A robot that stops (battery death, mis-assignment reveal, nav drop)
    # ends the tick at its current cell; the next tick's replan handles
    # it before anyone moves.
    # ================================================================
    for robot in robots:
        if not robot.assigned:
            unassigned_count += 1
            continue
        if robot.current_task is None:
            continue

        task = robot.current_task

        # at task location -> work on it. The gate is the robot's
        # PHYSICAL location, never path bookkeeping: a pathless robot
        # (newly assigned after a failed replan) has remaining_distance 0
        # and must NOT teleport onto the task.
        if tuple(robot.location) != tuple(task.location):
            continue

        # PO arrival hook: reveal the task's mask (once) and run
        # balanced's hard-fail checks against the TRUE attributes.
        # A mis-assignment frees the robot, returns the task to the
        # pool unmasked, and raises a replan event.
        if po_state is not None and po_state.handle_arrival(
                robot, task, unassigned_robots, unassigned_tasks,
                current_tick, pending_event_steps):
            unassigned_count += 1
            continue

        # (the runtime payload / manipulator re-checks in the old code
        # parsed a legacy string format and never fired; omitted)

        suitability = getattr(robot, "current_task_suitability", 0.5)
        if suitability is None:
            suitability = 0.5
        suitability = max(0.0, min(1.0, suitability))
        speed_factor = 0.5 + 1.5 * suitability  # [0.5, 2.0]

        robot.location = task.location
        start_positions[robot.robot_id] = robot.location
        robot.battery_life -= time_step
        task.time_left -= time_step * speed_factor

        # task completion
        if task.time_left <= 0:
            total_reward += task.reward
            total_success += 1
            task.assigned = False
            task.assigned_robot = None
            robot.current_task = None
            robot.tasks_successful += 1
            robot.assigned = False
            robot.time_on_task = 0
            robot.current_path = []
            robot.remaining_distance = 0
            robot.current_task_suitability = None
            task.current_suitability = None
            rid = robot.robot_id
            if rid not in unassigned_robots:
                unassigned_robots.append(rid)
            tasks_completed += 1
            try:
                tasks.remove(task)
            except ValueError:
                pass
            unassigned_count += 1
            continue

        # battery exhausted while working
        if robot.battery_life <= 0:
            if getattr(task, "reset_progress", False):
                task.time_left = task.time_to_complete
            else:
                task.time_to_complete = task.time_left
            if po_state is not None:
                po_state.on_unassign(robot)
            unassign_task_from_robot(
                robot, task,
                unassigned_robots=unassigned_robots,
                unassigned_tasks=unassigned_tasks,
            )
            unassigned_count += 1
            continue

        # safety-compliance stochastic failure
        if task.performance_metrics == "safety compliance":
            if robot.safety_features:
                matched_safety = sum(safety in robot.safety_features for safety in task.safety_protocols)
                safety_score = matched_safety / max(1, len(task.safety_protocols))
                if safety_score < 0.75 and (random.random() > 0.75):
                    if po_state is not None:
                        po_state.on_unassign(robot)
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    unassigned_count += 1
                    continue
            else:
                if po_state is not None:
                    po_state.on_unassign(robot)
                unassign_task_from_robot(
                    robot, task,
                    unassigned_robots=unassigned_robots,
                    unassigned_tasks=unassigned_tasks,
                )
                unassigned_count += 1
                continue

        # navigation drop (pre-existing behavior)
        nav_score = navigation_suitability(
            robot.mobility_type, robot.size, robot.sensor_range,
            task.navigation_constraints or [],
        )
        if nav_score == 0.0:
            if po_state is not None:
                po_state.record_nav_drop(robot)
                po_state.on_unassign(robot)
            unassign_task_from_robot(
                robot, task,
                unassigned_robots=unassigned_robots,
                unassigned_tasks=unassigned_tasks,
            )
            unassigned_count += 1
            continue

    # ================================================================
    # Phase 5: safety nudge (last resort; should never fire with the
    # two-phase timestep + idle agents in the plan). A PATHLESS robot
    # that ended the tick sharing a cell is nudged to a free neighbor.
    # ================================================================
    if obstacles is not None and COLLISION_AVOIDANCE:
        by_loc = {}
        for r in robots:
            by_loc.setdefault(r.location, []).append(r)
        for loc, group in by_loc.items():
            if len(group) < 2:
                continue
            rr, cc = loc
            for occupant in group:
                if occupant.assigned or (occupant.current_path and len(occupant.current_path) > 1):
                    continue  # only nudge PARKED robots
                for cand in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if cand in obstacles or cand in occupied_locations:
                        continue
                    if any(other.location == cand for other in group if other is not occupant):
                        continue
                    occupied_locations.discard(occupant.location)
                    occupant.location = cand
                    occupied_locations.add(cand)
                    if collision_counters is not None:
                        collision_counters["nudge"] += 1
                    break

    # ================================================================
    # End-of-tick invariant: no two robots share a cell and no two
    # robots swapped cells this tick (MAPF validity). Counted every tick
    # regardless of the safety-net switch; violations are bugs or
    # stale-path conflicts the nets could not repair.
    # ================================================================
    end_locs = {r.robot_id: r.location for r in robots}
    by_end = {}
    for rid, loc in end_locs.items():
        by_end.setdefault(loc, []).append(rid)
    colocated = sum(len(g) - 1 for g in by_end.values() if len(g) > 1)
    swapped = 0
    ids = sorted(end_locs)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if (end_locs[a] == _start_locs.get(b) and end_locs[b] == _start_locs.get(a)
                    and end_locs[a] != end_locs[b]):
                swapped += 1
    if collision_counters is not None and (colocated or swapped):
        collision_counters["violations"] += colocated + swapped

    return tasks_completed, unassigned_count, total_reward, total_success
