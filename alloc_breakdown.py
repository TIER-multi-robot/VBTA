"""Item 3 (2026-10-27): per-method allocation-time composition.

What the per-call allocation timer covers (established from the code):

  - The SWEEP metric (CSV: (Planning_Time_us + total_reassignment_time) /
    (1 + total_reassignments)) times each allocator's CORE only:
      voting:  assign_tasks_with_voting = pool assembly (warm-start +
               hall-of-fame translation) + candidate generation (with the
               diversity filter) + ranking / anytime loop + top-K output.
      Murty's: murtys_k_best_assignments (k=10 on the replan path, k=1 on
               the initial path).
      perturbed JV: assign_tasks_with_method = the JV solve on the
               perturbed submatrix.
    For EVERY method this EXCLUDES the per-call submatrix scoring (the
    suitability_source call at the top of the reassign entry) and the
    post-solve stealing loop - so the sweep's per-call metric is
    core-comparable across methods.

  - The REAL-TIME check (item 6) wraps the whole ENTRY functions
    (reassign_robots_to_tasks / _with_method / _with_murtys + the initial
    assigns), so its alloc time = scoring + core + steal for every method
    - the right measure for the 1-second-tick question, and comparable
    because the same entry points are wrapped for all methods.

This driver makes the composition explicit with sub-timers (wrapper-based,
no core changes): per allocation call it records
  scoring   = suitability-scorer time before the core (the per-call
              submatrix build),
  core      = candidate generation + ranking (voting) / the solve
              (Murty's, perturbed JV),
  post-core = total - scoring - core (the stealing loop INCLUDING its
              scorer calls, plus assignment bookkeeping).

One noise-sweep group per map (lak102d 10x10, den204d 20x20, lak307d
50x50; scenario calm, rep 0, sigma_obs = 0), all 8 methods, run serially
in-process. NOTE: run alongside the hyper sweep, so absolute times are
load-inflated; the scoring/core/post-core SHARES are the point.

Output: hvbta/io/results/alloc_breakdown/alloc_breakdown.json + table.

Usage: python alloc_breakdown.py
"""
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

os.environ["HVBTA_NOISE_SIGMAS"] = "0"

_CALLS = []       # (map, method, phase, scoring_us, core_us, post_us)
_DEPTH = [0]      # > 0 while inside a reassign entry
_CUR = {}         # per-entry accumulators
_PHASE = ["pre-core"]

_ORIG = {}


def _wrap(mod, name):
    fn = getattr(mod, name)
    _ORIG[(mod, name)] = fn
    setattr(mod, name, lambda *a, _f=fn, **kw: _timed(mod, name, _f, a, kw))


def _timed(mod, name, fn, args, kw):
    is_entry = name in ("reassign_robots_to_tasks",
                        "reassign_robots_to_tasks_with_method",
                        "reassign_robots_to_tasks_with_murtys")
    is_core = name in ("assign_tasks_with_voting", "assign_tasks_with_method",
                       "murtys_k_best_assignments")
    t0 = time.perf_counter_ns()
    if is_entry:
        _DEPTH[0] += 1
        _CUR["scoring"] = 0.0
        _CUR["core"] = 0.0
        _PHASE[0] = "pre-core"
    try:
        result = fn(*args, **kw)
    finally:
        dt = (time.perf_counter_ns() - t0) / 1000.0
        if is_entry:
            _DEPTH[0] -= 1
            _CALLS.append(_CALL_META[:] + [
                _CUR["scoring"], _CUR["core"],
                max(0.0, dt - _CUR["scoring"] - _CUR["core"])])
        elif is_core:
            if _DEPTH[0] > 0:
                _CUR["core"] += dt
                _PHASE[0] = "post-core"
            else:
                # top-level initial assign: the matrix was built outside;
                # the whole call is core.
                _CALLS.append(_CALL_META[:] + [0.0, dt, 0.0])
        # scorer wrappers handled by _wrap_scoring
    return result


def _wrap_scoring():
    import hvbta.suitability as S
    fn = S.evaluate_suitability_balanced

    def timed_scoring(*args, **kw):
        t0 = time.perf_counter_ns()
        try:
            return fn(*args, **kw)
        finally:
            if _DEPTH[0] > 0 and _PHASE[0] == "pre-core":
                _CUR["scoring"] += (time.perf_counter_ns() - t0) / 1000.0

    _ORIG[("S", "balanced")] = fn
    S.evaluate_suitability_balanced = timed_scoring


_CALL_META = ["", ""]   # [map, method] set around each run


def main():
    import copy
    import numpy as np
    import multiprocessing as mp
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    import hvbta.simulation.main_simulation as MS
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import (
        PartialObservabilityState, noisy_scorer)

    # Install wrappers (must be patched before the masked/noisy scorer
    # wrappers below capture the balanced scorer).
    _wrap_scoring()
    _wrap(V, "assign_tasks_with_voting")
    _wrap(O, "assign_tasks_with_method")
    _wrap(V, "reassign_robots_to_tasks")
    _wrap(O, "reassign_robots_to_tasks_with_method")
    _wrap(O, "reassign_robots_to_tasks_with_murtys")
    _wrap(O, "murtys_k_best_assignments")
    _wrap(MS, "run_planner")   # not bucketed; keeps call parity harmless

    methods = [
        ("rank_assignments_borda", V.rank_assignments_borda),
        ("rank_assignments_approval", V.rank_assignments_approval),
        ("rank_assignments_majority_judgment", V.rank_assignments_majority_judgment),
        ("rank_assignments_cumulative_voting", V.rank_assignments_cumulative_voting),
        ("rank_assignments_condorcet_method", V.rank_assignments_condorcet_method),
        ("rank_assignments_range", V.rank_assignments_range),
        ("murtys_task_allocation", O.murtys_task_allocation),
        ("perturbed_jv_task_allocation", O.perturbed_jv_task_allocation),
    ]

    out_dir = os.path.join("hvbta", "io", "results", "alloc_breakdown")
    os.makedirs(out_dir, exist_ok=True)

    groups = [("lak102d.map", 10, 10), ("den204d.map", 20, 20),
              ("lak307d.map", 50, 50)]
    for map_name, nr, nt in groups:
        rep, scenario = 0, "calm"
        grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
        HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
        map_dict = {"dimension": (len(grid), len(grid[0])),
                    "obstacles": create_obstacle_list(grid)}
        nc = max(nr, nt) * 6
        import zlib
        seed_tuple = (map_name, nr, nt, nc, "evaluate_suitability_balanced",
                      rep, scenario)
        schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF

        _seed_rngs(_derived_seed(schedule_seed, "entities"))
        robot_locs, robots = set(), []
        for i in range(nr):
            r = G.generate_random_robot_profile_strict(f"R{i+1}", grid, robot_locs)
            robots.append(r)
            robot_locs.add(r.location)
        task_locs, tasks = set(), []
        for i in range(nt):
            t = G.generate_random_task_description_strict(f"T{i+1}", grid, task_locs, [])
            tasks.append(t)
            task_locs.add(t.location)

        regime = REGIMES[scenario]
        provisional = max(200, int(HYP * 3))
        schedule = build_event_schedule(
            robots, tasks, grid, provisional,
            add_tasks=True, add_robots=True, remove_robots_flag=True,
            remove_tasks_flag=True,
            task_arrival_rate=regime["task_arrival_rate"],
            robot_arrival_rate=regime["robot_arrival_rate"],
            robot_departure_rate=regime["robot_departure_rate"],
            task_cancellation_rate=regime["task_cancellation_rate"],
            tasks_to_add=2, robots_to_add=1, robots_to_remove=1,
            tasks_to_remove=1, seed=schedule_seed)
        horizon = horizon_for_event_count(schedule, 125)
        max_steps = max(200, horizon, int(HYP * 1.5))
        schedule = schedule[:max_steps]

        sm = S.evaluate_suitability_balanced
        O.PERTURBED_JV_SIGMA = 0.05
        po_cond = PartialObservabilityState(schedule_seed, "none", "none")
        sm_for_matrix = noisy_scorer(po_cond.masked_scorer(sm), 0.0,
                                     schedule_seed, [0])
        matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYP)

        for pretty, method in methods:
            _CALL_META[:] = [map_name, pretty]
            _seed_rngs(_derived_seed(schedule_seed, "initial", pretty))
            if pretty == "murtys_task_allocation":
                out, score, length, per = O.assign_tasks_with_method(method, matrix)
            elif pretty == "perturbed_jv_task_allocation":
                init_fn = O.perturbed_jv_factory(
                    _derived_seed(schedule_seed, "run", pretty, "pjv", "initial"))
                out, score, length, per = O.assign_tasks_with_method(init_fn, matrix)
            else:
                out, score, length, per = V.assign_tasks_with_voting(
                    robots, tasks, matrix, nc, method,
                    tie_seed=_derived_seed(schedule_seed, "ties", "initial", pretty))
            benchmark_simulation(
                output=out, robots=copy.deepcopy(robots),
                tasks=copy.deepcopy(tasks),
                num_candidates=nc, voting_method=method, grid=grid,
                map_dict=map_dict, suitability_method=sm,
                suitability_matrix=matrix,
                max_time_steps=max_steps, add_tasks=True, add_robots=True,
                remove_robots=True, event_schedule=schedule,
                mask_level="none", mask_policy="none", noise_sigma=0.0,
                schedule_seed=schedule_seed,
                run_seed=_derived_seed(schedule_seed, "run", pretty),
                planner="lacam", scenario=scenario,
                scheduled_event_count=0, horizon_binding_term="")
        print(f"  group {map_name} {nr}x{nt} done "
              f"(calls so far: {len(_CALLS)})", flush=True)

    # ---- report ----
    def stats(vals):
        a = np.asarray(vals) if vals else np.asarray([float("nan")])
        return (float(np.percentile(a, 50)), float(np.percentile(a, 95)),
                float(a.max()), len(vals))

    print("\n=== allocation-time composition per method "
          "(one group per map; run under hyper-sweep load) ===")
    print(f"  {'method':38s} {'n':>5s} | "
          f"{'scor p50':>9s} {'p95':>9s} {'max':>9s} | "
          f"{'core p50':>9s} {'p95':>9s} {'max':>9s} | "
          f"{'post p50':>9s} {'p95':>9s} {'max':>9s}   (us/call)")
    summary = {}
    for pretty, _ in methods:
        sel = [c for c in _CALLS if c[1] == pretty]
        scoring = stats([c[2] for c in sel])
        core = stats([c[3] for c in sel])
        post = stats([c[4] for c in sel])
        summary[pretty] = {
            "n_calls": len(sel),
            "scoring_us_p50_p95_max": scoring,
            "core_us_p50_p95_max": core,
            "post_core_us_p50_p95_max": post,
            "share_scoring": float(np.mean([c[2] for c in sel])) if sel else None,
        }
        print(f"  {pretty:38s} {len(sel):5d} | "
              f"{scoring[0]:9.0f} {scoring[1]:9.0f} {scoring[2]:9.0f} | "
              f"{core[0]:9.0f} {core[1]:9.0f} {core[2]:9.0f} | "
              f"{post[0]:9.0f} {post[1]:9.0f} {post[2]:9.0f}")
    with open(os.path.join(out_dir, "alloc_breakdown.json"), "w",
              encoding="utf-8") as f:
        json.dump({"calls": _CALLS, "summary": summary}, f, indent=2)
    print(f"\nwritten: {os.path.join(out_dir, 'alloc_breakdown.json')}")


if __name__ == "__main__":
    main()