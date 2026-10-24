import random
from typing import List, Tuple
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.allocators.misc_assignment import unassign_task_from_robot
from hvbta.suitability import navigation_suitability

# Master switch for the Phase 3 anti-collision fixes (idle-robot yield
# swap + post-tick reconciliation). Exists so the pre-fix behavior can be
# reproduced for measurement/ablation; leave True.
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
    obstacles: frozenset = None
) -> Tuple[int, int, float, int]:
    """
    Simulates a single time step, updating robot positions, task progress, and handling failures.

    Parameters:
        robots: List of all robots.
        tasks: List of all tasks.
        time_step: The time increment for the simulation step.
        total_reward: Accumulated reward from successfully completed tasks.
        po_state: Optional PartialObservabilityState (partial-observability
            experiment). When provided, robot arrivals reveal task masks and
            run the execution-time feasibility check; mis-assignments free
            the robot and raise a replan event. Default None = no-op.
        current_tick: Current loop timestep (only used by po_state for
            event bookkeeping).
        pending_event_steps: The main loop's pending-event FIFO (only used
            by po_state to raise replan events on mis-assignment).
        obstacles: Optional set of blocked cells. When provided, a
            post-movement reconciliation nudges any pathless robot that
            ended the tick sharing a cell with another robot into a free
            neighboring cell (see the reconciliation block below).

    Returns:
        (tasks_completed, count, total_reward, total_success): A count of unassigned robots and the updated total reward.
    """
    unassigned_count = 0
    tasks_completed = 0

    for robot in robots:
        if robot.assigned and robot.current_task and robot.current_path:
            task = robot.current_task

            # walk one step along the precomputed path (built by CBS upstream)
            if robot.current_path and len(robot.current_path) > 1:
                next_position = robot.current_path[1]
                # Idle-robot yield: the planner treats unassigned robots as
                # non-blocking (they are neither movers nor stationary
                # blockers in _prepare_agents), so a mover's path may route
                # straight through an idle robot's cell. Idle robots have no
                # path and never move on their own, so without this the
                # mover steps onto them, and once the co-located idle robot
                # is later assigned, start_positions contains two robots on
                # one cell - the duplicate-start cascade trigger. Fix: the
                # idle robot yields into the cell the mover is vacating
                # (the mover is by definition adjacent). A swap leaves the
                # occupied_locations set unchanged.
                idle_hit = next(
                    (r for r in robots
                     if r is not robot and not r.assigned
                     and r.location == next_position), None) if COLLISION_AVOIDANCE else None
                if idle_hit is not None:
                    # swap cells: the vacated cell stays occupied (by the
                    # idle robot), the target cell stays occupied (by us)
                    idle_hit.location = robot.location
                else:
                    occupied_locations.discard(robot.location)
                robot.location = next_position
                start_positions[robot.robot_id] = next_position
                occupied_locations.add(next_position)
                robot.current_path.pop(0)
                robot.remaining_distance = max(0, len(robot.current_path) - 1)
                if po_state is not None:
                    po_state.on_move(robot)

                if robot.remaining_distance <= 1:
                    robot.remaining_distance = 0

                if robot.battery_life == 0:
                    if po_state is not None:
                        po_state.on_unassign(robot)
                    unassign_task_from_robot(robot, task, unassigned_robots=unassigned_robots, unassigned_tasks=unassigned_tasks)
                    continue

            # at task location -> work on it
            if robot.remaining_distance == 0:
                # Partial-observability arrival hook: reveal the task's mask
                # (once) and run balanced's hard-fail checks against the TRUE
                # attributes. A mis-assignment frees the robot, returns the
                # task to the pool unmasked, and raises a replan event.
                if po_state is not None and po_state.handle_arrival(
                        robot, task, unassigned_robots, unassigned_tasks,
                        current_tick, pending_event_steps):
                    continue

                required_caps = getattr(task, "required_capabilities", None) or []
                payload_required = None
                for req in required_caps:
                    if isinstance(req, str) and "payload" in req.lower():
                        try:
                            payload_required = float(req.split(">=")[-1].strip())
                        except ValueError:
                            payload_required = None
                tools_needed = getattr(task, "tools_needed", None) or []
                sensors_required = tools_needed[0] if len(tools_needed) > 0 and tools_needed[0] else []
                manipulators_required = tools_needed[1] if len(tools_needed) > 1 and tools_needed[1] else []

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
                    else:
                        if po_state is not None:
                            po_state.on_unassign(robot)
                        unassign_task_from_robot(
                            robot, task,
                            unassigned_robots=unassigned_robots,
                            unassigned_tasks=unassigned_tasks,
                        )

                # runtime capability re-checks
                if payload_required is not None and robot.payload_capacity < payload_required:
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    continue

                if manipulators_required and not any(tool in robot.manipulators for tool in manipulators_required):
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    continue

                nav_score = navigation_suitability(
                    robot.mobility_type, robot.size, robot.sensor_range,
                    task.navigation_constraints or [],
                )
                if nav_score == 0.0:
                    if po_state is not None:
                        # Pre-existing navigation drop: count it, don't
                        # change the behavior (spec §6.5).
                        po_state.record_nav_drop(robot)
                        po_state.on_unassign(robot)
                    unassign_task_from_robot(
                        robot, task,
                        unassigned_robots=unassigned_robots,
                        unassigned_tasks=unassigned_tasks,
                    )
                    continue

        elif not robot.assigned:
            unassigned_count += 1

    # ---- Post-movement reconciliation (Phase 3 cascade fix) ----
    # No two robots may share a cell at the end of a tick. The idle-yield
    # swap above handles movers stepping onto idle robots, but a robot can
    # also be frozen mid-plan on a cell another agent's plan depends on:
    # a task cancellation / battery death / mis-assignment reveal clears
    # its path while it stands where a later mover (or a LaCAM goal
    # displacement schedule) expects it to vacate. Nudge any PATHLESS
    # robot that shares a cell into a free neighboring cell; if none
    # exists, leave it (the next planner call fails as before).
    if obstacles is not None and COLLISION_AVOIDANCE:
        by_loc = {}
        for r in robots:
            by_loc.setdefault(r.location, []).append(r)
        for loc, group in by_loc.items():
            if len(group) < 2:
                continue
            rr, cc = loc
            for occupant in group:
                if occupant.current_path or occupant.assigned:
                    continue  # only nudge frozen (pathless, unassigned) robots
                for cand in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if cand in obstacles or cand in occupied_locations:
                        continue
                    if any(other.location == cand for other in group if other is not occupant):
                        continue
                    occupied_locations.discard(occupant.location)
                    occupant.location = cand
                    occupied_locations.add(cand)
                    break

    return tasks_completed, unassigned_count, total_reward, total_success
