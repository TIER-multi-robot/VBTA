import random
from typing import List, Tuple
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.allocators.misc_assignment import unassign_task_from_robot
from hvbta.suitability import navigation_suitability

# The v6-era safety-nudge net is REMOVED (v9): with mover-mover
# resolution + transitive waiting + the occupied-set race fixed, the
# pilot shows zero nudge fires, zero planned conflicts, and zero
# violations, so the net never fires and only masks invariant
# breaches. The end-of-tick invariant check below stays as the hard
# detector (Collision_Violation_Count).


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
    Phase 1.5a: mover-mover conflict resolution. Movers targeting the
             same cell, or exchanging cells, are resolved deterministi-
             cally: the lower robot ID keeps its move, the higher one
             waits (Mover_Conflict_Waits).
    Phase 1.5b: transitive waiting. Robots without a usable path (newly
             assigned after a failed replan, newly arrived, unassigned
             mid-route) stay put; any robot whose next cell is occupied
             by a staying robot also stays put, repeated to a fixed
             point. Forced waits are counted in propagated_waits.
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
    Phase 3.5: deviation detection. Any off-plan wait or an assigned
              robot that cannot follow any plan sets the plan_deviation
              flag; the main loop replans at the top of the next tick,
              before movement.
    Phase 4: apply status changes (task progress and completions, battery
              drain and deaths, reveals and mis-assignments). A robot that
              stops ends the tick at its current cell; the next tick's
              top-of-tick replan includes it before anyone moves. Work
              requires the robot to be physically AT the task location -
              path bookkeeping alone never teleports a robot onto a task.

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
    # Phase 1.5a: mover-mover conflict resolution. Two MOVERS targeting
    # the same cell, or two movers exchanging cells, cannot both
    # proceed. Deterministic rule: the LOWER robot ID keeps its move;
    # the higher one waits at its current cell (counted in
    # mover_conflict_waits). Resolution runs BEFORE the transitive-wait
    # fixed point because a resolved waiter becomes a stayer that can
    # block further robots.
    # ================================================================
    loc_by_id = {r.robot_id: r.location for r in robots}
    def _rid_key(rid):
        # numeric robot id order (R12 > R2), falling back to the string
        import re as _re
        m = _re.match(r"R(\d+)$", rid)
        return (int(m.group(1)) if m else 0, rid)
    movers = [rid for rid in next_positions if next_positions[rid] != loc_by_id[rid]]
    mover_waits = 0
    # vertex: >1 movers targeting the same cell -> lowest id keeps moving
    by_target = {}
    for rid in movers:
        by_target.setdefault(next_positions[rid], []).append(rid)
    for cell in sorted(by_target):
        rids = sorted(by_target[cell], key=_rid_key)
        for rid in rids[1:]:
            next_positions[rid] = loc_by_id[rid]
            mover_waits += 1
    # swap: two movers exchanging cells -> higher id waits (the lower
    # one is then caught by the fixed point below if the waiter's cell
    # blocks it, which correctly forces both to hold this tick)
    movers = [rid for rid in next_positions if next_positions[rid] != loc_by_id[rid]]
    movers.sort(key=_rid_key)
    for i in range(len(movers)):
        for j in range(i + 1, len(movers)):
            a, b = movers[i], movers[j]
            if (next_positions[a] == loc_by_id[b]
                    and next_positions[b] == loc_by_id[a]):
                loser = b  # higher _rid_key of the pair
                next_positions[loser] = loc_by_id[loser]
                mover_waits += 1
    if collision_counters is not None and mover_waits:
        collision_counters["mover_conflict_waits"] = (
            collision_counters.get("mover_conflict_waits", 0) + mover_waits)

    # ================================================================
    # Phase 1.5b: transitive waiting (v7 stale-path safety). Seed: every
    # robot that stays put this tick (no usable path, path exhausted, a
    # planned wait, or a mover-conflict resolution above). Propagate:
    # any robot whose next cell is occupied by a staying robot also
    # stays put, to a fixed point. Survivors move on old paths that are
    # jointly conflict-free, and stayers never move, so no
    # mover-stayer vertex or swap conflict is possible.
    # Forced waits (a robot that intended to move but must hold) are
    # counted in propagated_waits.
    # ================================================================
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
    # The occupied-set update is TWO-PASS (discard every vacated cell,
    # then add every entered cell): updating per-robot inside the loop
    # loses a cell whenever a robot enters a cell another robot is
    # vacating in the same synchronous move (the later robot's discard
    # deletes the earlier robot's fresh entry). The set would decay
    # over the run and arrival nudges would place new robots onto
    # occupied cells.
    vacated = [robot.location for robot in robots]
    entered = [next_positions[robot.robot_id] for robot in robots]
    for robot in robots:
        rid = robot.robot_id
        next_pos = next_positions[rid]
        if next_pos != robot.location:
            robot.location = next_pos
            if robot.assigned:
                start_positions[rid] = next_pos
            if po_state is not None and robot.assigned:
                po_state.on_move(robot)
        if (robot.current_path and len(robot.current_path) > 1
                and robot.current_path[1] == next_pos):
            robot.current_path.pop(0)
            robot.remaining_distance = max(0, len(robot.current_path) - 1)
    for cell in vacated:
        occupied_locations.discard(cell)
    for cell in entered:
        occupied_locations.add(cell)

    # ================================================================
    # Phase 3.5: deviation detection (v8 replan-on-deviation). Any
    # off-plan wait (a forced propagated wait, a mover-conflict
    # resolution) or an assigned robot that cannot follow any plan
    # (pathless and not at its task) marks the plan STALE. The main
    # loop reads this flag after the timestep and triggers a planner
    # call at the top of the next tick, before movement.
    # ================================================================
    deviation = (forced_waits > 0) or (mover_waits > 0)
    for robot in robots:
        if (robot.assigned and robot.current_task is not None
                and (not robot.current_path or len(robot.current_path) <= 1)
                and tuple(robot.location) != tuple(robot.current_task.location)):
            deviation = True
    if collision_counters is not None:
        collision_counters["plan_deviation"] = 1 if deviation else 0

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
    # End-of-tick invariant: no two robots share a cell and no two
    # robots swapped cells this tick (MAPF validity). Counted every
    # tick; any violation is an unpatched collision bug.
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
