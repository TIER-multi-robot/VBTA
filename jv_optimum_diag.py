"""Item 8 (final scope): JV-optimum diagnostic on non-urgent voting calls.

WHAT IT MEASURES (report before running at full size): for every NON-URGENT
voting allocation call in the sampled noise-sweep groups, the ratio of the
voting winner's total suitability to the OPTIMAL assignment value, computed
offline with JV on the exact same unassigned x unassigned submatrix the
allocator saw (the allocator's masked/noisy view, recomputed deterministically
- the noisy scorer is seeded per (schedule_seed, robot, task, tick)). Reported
by unassigned-robot bin, per voting rule. Diagnostic only: no algorithm
change, no results files touched; the group re-runs are deterministic
reproductions of the noise sweep's call sequences (which is why this must run
IDLE - a LaCAM budget flip under load would change the call sequence).

Sampling (full size): lak102d all 4 nr x nt combos, lak307d 50x50 and
100x100; 4 scenarios, rep 0; sigma_obs in {0, 0.2}; 6 voting methods
(optimizers are skipped - they ARE the optimum baseline).

Usage:
  python jv_optimum_diag.py            # full sample (idle machine only)
  python jv_optimum_diag.py --pilot    # 2 groups only, validates plumbing
"""
import csv
import json
import os
import sys
import zlib

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

_RECORDS = []
_CUR = ["", "", "", "", "", ""]   # map, nr, nt, scenario, rep, sigma


def _install_wrappers():
    import numpy as np
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    from hvbta.simulation.main_simulation import REGIMES  # noqa: F401
    orig = V.reassign_robots_to_tasks

    def wrapped(robots, tasks, num_candidates, voting_method, suitability_source,
                unassigned_robots, unassigned_tasks, start_positions, goal_positions,
                map_size, **kw):
        is_non_urgent = kw.get("stall_k") is None and kw.get("score_threshold") is None
        is_voting = getattr(voting_method, "__name__", "").startswith(
            "rank_assignments")
        # PRE-call unassigned pools and the allocator's view of the
        # submatrix (post-call state would already be assigned).
        pre_urobots = [r for r in robots if not r.assigned] if (is_non_urgent and is_voting) else None
        pre_utasks = [t for t in tasks if not t.assigned] if pre_urobots is not None else None
        sub = None
        if pre_urobots and pre_utasks:
            if isinstance(suitability_source, tuple) and len(suitability_source) == 3:
                full_matrix, r_idx, t_idx = suitability_source
                from hvbta.allocators.optimizers import extract_submatrix
                sub = extract_submatrix(full_matrix, pre_urobots, pre_utasks,
                                       r_idx, t_idx)
            else:
                sub = np.clip(suitability_source(pre_urobots, pre_utasks,
                                                  map_size=map_size), 0.0, 1.0)
        result = orig(robots, tasks, num_candidates, voting_method, suitability_source,
                      unassigned_robots, unassigned_tasks, start_positions,
                      goal_positions, map_size, **kw)
        if sub is not None:
            ridx = {r.robot_id: i for i, r in enumerate(pre_urobots)}
            tidx = {t.task_id: j for j, t in enumerate(pre_utasks)}
            winner_pairs = [(ridx[rid], tidx[tid])
                            for rid, tid in result[0].items()
                            if rid in ridx and tid in tidx]
            winner_val = float(sum(sub[i][j] for i, j in winner_pairs))
            opt_pairs = O.jv_task_allocation(sub)
            opt_val = float(sum(sub[i][j] for i, j in opt_pairs))
            _RECORDS.append({
                "map": _CUR[0], "nr": _CUR[1], "nt": _CUR[2],
                "scenario": _CUR[3], "rep": _CUR[4], "sigma": _CUR[5],
                "method": getattr(voting_method, "__name__"),
                "n_robots": len(pre_urobots), "n_tasks": len(pre_utasks),
                "pool": num_candidates,
                "winner": winner_val, "optimum": opt_val,
                "ratio": (winner_val / opt_val) if opt_val > 0 else None,
            })
        return result

    V.reassign_robots_to_tasks = wrapped


def _run_group(map_name, nr, nt, scenario, rep, sigma, methods, bench_kwargs=None):
    import copy
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.optimizers as O
    import hvbta.simulation.main_simulation as MS
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import (
        PartialObservabilityState, noisy_scorer)

    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    seed_tuple = (map_name, nr, nt, nc, "evaluate_suitability_balanced",
                  rep, scenario)
    schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF
    _CUR[:] = [map_name, nr, nt, scenario, rep, sigma]

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
    extender = 1.5 if (nt / nr) > 10 else 1.2 if (nt / nr) > 5 else 1.0
    max_steps = max(200, horizon, int(HYP * 1.5 * extender))
    schedule = schedule[:max_steps]

    sm = S.evaluate_suitability_balanced
    O.PERTURBED_JV_SIGMA = 0.05
    po_cond = PartialObservabilityState(schedule_seed, "none", "none")
    sm_for_matrix = noisy_scorer(po_cond.masked_scorer(sm), sigma,
                                 schedule_seed, [0])
    matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYP)

    import hvbta.allocators.voting as V
    for pretty, method in methods:
        _seed_rngs(_derived_seed(schedule_seed, "initial", pretty))
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
            mask_level="none", mask_policy="none", noise_sigma=sigma,
            schedule_seed=schedule_seed,
            run_seed=_derived_seed(schedule_seed, "run", pretty),
            planner="lacam", scenario=scenario,
            scheduled_event_count=0, horizon_binding_term="")


def main():
    pilot = "--pilot" in sys.argv
    import hvbta.allocators.voting as V
    voting = [
        ("rank_assignments_borda", V.rank_assignments_borda),
        ("rank_assignments_approval", V.rank_assignments_approval),
        ("rank_assignments_majority_judgment", V.rank_assignments_majority_judgment),
        ("rank_assignments_cumulative_voting", V.rank_assignments_cumulative_voting),
        ("rank_assignments_condorcet_method", V.rank_assignments_condorcet_method),
        ("rank_assignments_range", V.rank_assignments_range),
    ]
    _install_wrappers()

    if pilot:
        jobs = [("lak102d.map", 10, 10, "calm", 0, 0.0)]
    else:
        jobs = ([(m, nr, nt, s, 0, sg)
                 for m, sizes in (("lak102d.map", (10, 20)),
                                   ("lak307d.map", (50, 100)))
                 for nr in sizes for nt in sizes
                 for s in ("calm", "task_churn", "fleet_churn", "overload")
                 for sg in (0.0, 0.2)])
    out_dir = os.path.join("hvbta", "io", "results",
                           "jv_optimum_diag" + ("_pilot" if pilot else ""))
    os.makedirs(out_dir, exist_ok=True)
    for i, (m, nr, nt, s, rep, sg) in enumerate(jobs, 1):
        _run_group(m, nr, nt, s, rep, sg, voting)
        print(f"  {i}/{len(jobs)} groups done ({m} {nr}x{nt} {s} sigma={sg}); "
              f"records: {len(_RECORDS)}", flush=True)

    import numpy as np
    bins = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
    print(f"\n=== JV-optimum diagnostic: {_RECORDS[0] and ''}"
          f"{len(_RECORDS)} non-urgent voting calls ===")
    print(f"  {'bin':>9s} {'n':>6s} {'ratio p50':>10s} {'p95':>7s} "
          f"{'min':>7s} {'=1.0':>6s}")
    summary = {}
    for lo, hi in bins:
        ratios = [r["ratio"] for r in _RECORDS
                  if r["ratio"] is not None and lo <= r["n_robots"] <= hi]
        if not ratios:
            continue
        a = np.asarray(ratios)
        line = (f"  {str(lo) + '-' + str(hi):>9s} {len(a):6d} "
                f"{np.percentile(a, 50):10.3f} {np.percentile(a, 95):7.3f} "
                f"{a.min():7.3f} {int((a == 1.0).sum()):6d}")
        print(line)
        summary[f"{lo}-{hi}"] = {
            "n": len(a), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()),
            "at_optimum": int((a == 1.0).sum()),
        }
    by_method = {}
    for pretty, _ in voting:
        ratios = [r["ratio"] for r in _RECORDS
                  if r["ratio"] is not None and r["method"] == pretty]
        if ratios:
            a = np.asarray(ratios)
            by_method[pretty] = {"n": len(a), "p50": float(np.percentile(a, 50)),
                                 "p95": float(np.percentile(a, 95))}
    print("per-method p50/p95:", {k: (v['p50'], v['p95'])
                                  for k, v in by_method.items()})
    with open(os.path.join(out_dir, "jv_optimum_records.json"), "w",
              encoding="utf-8") as f:
        json.dump({"records": _RECORDS, "by_bin": summary,
                   "by_method": by_method}, f, indent=2)
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()