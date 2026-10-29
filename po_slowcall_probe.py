"""Item 3 (2026-10-28): where the slow pessimistic-mask voting calls spend
time on lak307d. Report-only diagnostic; no core-code changes, no results
directory touched (writes its own probe dir).

Runs ONE in-vivo simulation run (lak307d, 100x100, calm, rep 0, mask
low/pessimistic, borda - a config class whose per-call allocation time
exploded from ~0.4 s to a 13 s mean / 50-65 s p95 under load in the PO
sweep), with wrapper-based sub-timers on every phase of
assign_tasks_with_voting reachable via module attributes:

  generator     - generate_high_suitability_assignments (candidate batches)
  zero_filter   - _filter_zero_pairs (per-candidate zero-pair filtering;
                  None returns = candidates rejected as all-infeasible)
  upper_bounds  - _upper_bounds (per call)
  weights       - apply_priority_weights (per call, before ranking)
  rule          - the voting rule itself (passed as a wrapped callable)
  inline        - remainder inside assign_tasks_with_voting: pool
                  assembly, the diversity filter's overlap checks and its
                  retry loop, and the stall/anytime accumulation loop

Per call it also records the masked view's zero-entry fraction (the
suitability matrix the allocator sees), the fractions of robots/tasks
with >= 1 positive entry, call dimensions, and V.POOL_STATS pool sizes.
The whole run runs under cProfile; the report prints the aggregate phase
shares and the profile's top functions in the voting path.

Usage: python po_slowcall_probe.py
"""

import cProfile
import io
import json
import os
import pstats
import sys
import time
import zlib
import copy

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

_CALLS = []
_PHASES = {}
_COUNTS = {}
_ORIG = {}


def _wrap(mod, name, label):
    fn = getattr(mod, name)

    def timed(*a, **kw):
        t0 = time.perf_counter_ns()
        result = fn(*a, **kw)
        _PHASES[label] = _PHASES.get(label, 0) + time.perf_counter_ns() - t0
        _COUNTS[label] = _COUNTS.get(label, 0) + 1
        if label == "zero_filter" and result is None:
            _COUNTS["zero_filter_none"] = _COUNTS.get("zero_filter_none", 0) + 1
        return result

    _ORIG[(mod, name)] = fn
    setattr(mod, name, timed)
    return timed


def main():
    import numpy as np
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    import hvbta.allocators.assignments as A
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import PartialObservabilityState

    map_name, nr, nt, scenario, rep = "lak307d.map", 100, 100, "calm", 0
    mask_level, mask_policy = "low", "pessimistic"

    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    seed_tuple = (map_name, nr, nt, nc, "evaluate_suitability_balanced",
                  rep, scenario)
    schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF

    _seed_rngs(_derived_seed(schedule_seed, "entities"))
    robot_locs, robots = set(), []
    for i in range(nr):
        r = G.generate_random_robot_profile_strict(f"R{i+1}", grid, robot_locs)
        robots.append(r); robot_locs.add(r.location)
    task_locs, tasks = set(), []
    for i in range(nt):
        t = G.generate_random_task_description_strict(f"T{i+1}", grid, task_locs, [])
        tasks.append(t); task_locs.add(t.location)

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
    po_cond = PartialObservabilityState(schedule_seed, mask_level, mask_policy)
    sm_for_matrix = po_cond.masked_scorer(sm)
    matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYP)

    # ---- instrumentation (wrappers only; no core changes) ----
    V.POOL_STATS = []
    V.POOL_STATS_SINK = []
    _wrap(A, "generate_high_suitability_assignments", "generator")
    _wrap(A, "generate_random_assignments", "generator")
    _wrap(V, "_filter_zero_pairs", "zero_filter")
    _wrap(V, "_upper_bounds", "upper_bounds")
    _wrap(S, "apply_priority_weights", "weights")
    rule_wrapped = _wrap(V, "rank_assignments_borda", "rule")
    core = V.assign_tasks_with_voting

    def call_wrapper(*a, **kw):
        m = np.asarray(a[2])
        idx = len(V.POOL_STATS)
        t0 = time.perf_counter_ns()
        result = core(*a, **kw)
        dt = time.perf_counter_ns() - t0
        _CALLS.append({
            "total_s": dt / 1e9,
            "zero_frac": float(np.mean(m == 0)),
            "pos_robot_frac": float(np.mean(np.any(m > 0, axis=1))),
            "pos_task_frac": float(np.mean(np.any(m > 0, axis=0))),
            "n_robots": len(a[0]), "n_tasks": len(a[1]),
            "nc": a[3],
            "stall_k": kw.get("stall_k"),
            "seed": kw.get("seed_assignment") is not None,
            "extra": len(kw.get("extra_candidates") or []),
            "pool_idx": idx,
        })
        return result

    V.assign_tasks_with_voting = call_wrapper

    pretty = "rank_assignments_borda"
    _seed_rngs(_derived_seed(schedule_seed, "initial", pretty))
    out, score, length, per = V.assign_tasks_with_voting(
        robots, tasks, matrix, nc, rule_wrapped,
        tie_seed=_derived_seed(schedule_seed, "ties", "initial", pretty))

    prof = cProfile.Profile()
    prof.enable()
    benchmark_simulation(
        output=out, robots=copy.deepcopy(robots),
        tasks=copy.deepcopy(tasks),
        num_candidates=nc, voting_method=rule_wrapped, grid=grid,
        map_dict=map_dict, suitability_method=sm,
        suitability_matrix=matrix,
        max_time_steps=max_steps, add_tasks=True, add_robots=True,
        remove_robots=True, event_schedule=schedule,
        mask_level=mask_level, mask_policy=mask_policy, noise_sigma=None,
        schedule_seed=schedule_seed,
        run_seed=_derived_seed(schedule_seed, "run", pretty),
        planner="lacam", scenario=scenario,
        scheduled_event_count=0, horizon_binding_term="")
    prof.disable()

    # ---- report ----
    pool = V.POOL_STATS_SINK or V.POOL_STATS
    print(f"probe run: {map_name} {nr}x{nt} {scenario} rep{rep} "
          f"mask={mask_level}/{mask_policy} borda; "
          f"{len(_CALLS)} voting calls", flush=True)
    slow = sorted(_CALLS, key=lambda c: -c["total_s"])
    tot = sum(c["total_s"] for c in _CALLS)
    print(f"total alloc time in calls: {tot:.1f} s; "
          f"slowest 5: " +
          ", ".join(f"{c['total_s']:.1f}s" for c in slow[:5]), flush=True)
    print("\nslowest 5 calls (masked view at call time):")
    for c in slow[:5]:
        ps = pool[c["pool_idx"]] if c["pool_idx"] < len(pool) else {}
        print(f"  {c['total_s']:7.1f}s  zeros={c['zero_frac']:.3f} "
              f"pos_robots={c['pos_robot_frac']:.2f} pos_tasks={c['pos_task_frac']:.2f} "
              f"dims={c['n_robots']}x{c['n_tasks']} nc={c['nc']} "
              f"stall_k={c['stall_k']} seed={c['seed']} extra={c['extra']} "
              f"pool_full={ps.get('full')} pool_ranked={ps.get('ranked')}")
    fast = [c for c in _CALLS if c["total_s"] < 1.0]
    if fast:
        zf_slow = [c["zero_frac"] for c in slow[:20]]
        zf_fast = [c["zero_frac"] for c in fast]
        print(f"\nzero-entry fraction: slow(20) mean "
              f"{np.mean(zf_slow):.3f} vs fast(n={len(fast)}) mean "
              f"{np.mean(zf_fast):.3f}", flush=True)

    print("\nphase totals across all calls (wrapper-measured):")
    for label in ("generator", "zero_filter", "upper_bounds", "weights", "rule"):
        print(f"  {label:12s} {(_PHASES.get(label, 0))/1e9:8.1f} s  "
              f"({_PHASES.get(label, 0)/1e9/max(tot, 1e-9)*100:5.1f}% of alloc)  "
              f"n={_COUNTS.get(label, 0)}"
              + (f"  (None-returns: {_COUNTS.get('zero_filter_none', 0)})"
                 if label == "zero_filter" else ""))
    inline = tot - sum(_PHASES.get(l, 0) / 1e9
                       for l in ("generator", "zero_filter", "upper_bounds",
                                 "weights", "rule"))
    print(f"  {'inline':12s} {inline:8.1f} s  "
          f"({inline/max(tot, 1e-9)*100:5.1f}% of alloc)  "
          f"[pool assembly + diversity overlap checks + retry loop + stall loop]")

    s = io.StringIO()
    st = pstats.Stats(prof, stream=s)
    st.sort_stats("cumulative").print_stats(
        r"voting|assignments|suitability", 18)
    print("\ncProfile top functions in the voting path (cumulative):")
    for line in s.getvalue().split("\n"):
        if line.strip():
            print(line)

    out_dir = os.path.join("hvbta", "io", "results", "po_slowcall_probe")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "po_slowcall_probe.json"), "w",
              encoding="utf-8") as f:
        json.dump({"calls": _CALLS,
                   "phases_s": {k: v / 1e9 for k, v in _PHASES.items()},
                   "counts": _COUNTS,
                   "pool_stats": pool}, f, indent=2)
    print(f"\nwritten: {out_dir}/po_slowcall_probe.json", flush=True)


if __name__ == "__main__":
    main()