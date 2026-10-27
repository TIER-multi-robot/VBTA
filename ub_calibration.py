"""Upper-bound calibration driver for the urgent-call stopping rule (4c).

Runs the v9 pilot configuration (lak102d [10,20]x[10,20], 4 scenarios, 5
mask conditions, 8 methods, 3 reps) PLUS one rep of the largest
robot x task configuration on one balanced map (den901d 100x100), with
V.UB_CALIBRATION enabled on every voting allocation call. Each record
captures the call's dimensions, the upper bounds (UB_robot / UB_task /
UB = min), the winner's coverage-normalized quality, and the positive
pair scores the allocator saw. NO completion numbers are read.

Usage:
  python ub_calibration.py run <outdir>          # run the calibration
  python ub_calibration.py report <outdir>       # write the report

Determinism: identical seeding to the serial sweep (phase4_sweep.run_group
scheme); the collector is passive (records only, never branches), so the
runs are bit-identical to the same config without the collector.
"""
import copy
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import zlib

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

N_WORKERS = max(1, (os.cpu_count() or 4) - 2)

# v9 pilot config + the large-map add-on. den901d chosen from the balanced
# large set (lak307d / den001d / den901d): the most obstacle-dense large
# map (50%), so feasibility structure - and therefore UB behavior at large
# call sizes - is stressed hardest; it is also the established timing
# reference for large-map runs.
CALIB_CONFIGS = [
    ("lak102d.map", [10, 20], [10, 20], 3),
    ("den901d.map", [100], [100], 1),
]

SCENARIOS = ["calm", "task_churn", "fleet_churn", "overload"]

VOTING_METHODS = [
    "rank_assignments_borda",
    "rank_assignments_approval",
    "rank_assignments_majority_judgment",
    "rank_assignments_cumulative_voting",
    "rank_assignments_condorcet_method",
    "rank_assignments_range",
]
OPT_METHODS = ["murtys_task_allocation", "perturbed_jv_task_allocation"]
ALL_METHODS = VOTING_METHODS + OPT_METHODS


def run_group(job):
    """Worker: all 5 conditions x 8 methods for one (map, size, scenario,
    rep) group, with the UB collector on. Returns (jsonl_path, n_records)."""
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import (
        PartialObservabilityState, PO_CONDITIONS,
    )

    map_name, nr, nt, scenario, rep = (job["map"], job["nr"], job["nt"],
                                       job["scenario"], job["rep"])
    out_path = job["out"]
    TRANSIENT_FLOOR, TARGET_EVENT_COUNT_K = 200, 125
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    sm_name = "evaluate_suitability_balanced"
    seed_tuple = (map_name, nr, nt, nc, sm_name, rep, scenario)
    schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF

    _seed_rngs(_derived_seed(schedule_seed, "entities"))
    robot_locs, robots = set(), []
    for idx in range(nr):
        r = G.generate_random_robot_profile_strict(f"R{idx+1}", grid, robot_locs)
        robots.append(r); robot_locs.add(r.location)
    task_locs, tasks = set(), []
    for idx in range(nt):
        t = G.generate_random_task_description_strict(f"T{idx+1}", grid, task_locs, [])
        tasks.append(t); task_locs.add(t.location)

    regime = REGIMES[scenario]
    provisional = max(TRANSIENT_FLOOR, int(HYP * 3))
    schedule = build_event_schedule(
        robots, tasks, grid, provisional,
        add_tasks=True, add_robots=True, remove_robots_flag=True, remove_tasks_flag=True,
        task_arrival_rate=regime["task_arrival_rate"],
        robot_arrival_rate=regime["robot_arrival_rate"],
        robot_departure_rate=regime["robot_departure_rate"],
        task_cancellation_rate=regime["task_cancellation_rate"],
        tasks_to_add=2, robots_to_add=1, robots_to_remove=1, tasks_to_remove=1,
        seed=schedule_seed,
    )
    horizon = horizon_for_event_count(schedule, TARGET_EVENT_COUNT_K)
    workload = max(1.0, nt / nr)
    extender = 1.5 if workload > 10 else 1.2 if workload > 5 else 1.0
    max_steps = max(TRANSIENT_FLOOR, horizon, int(HYP * 1.5 * extender))
    schedule = schedule[:max_steps]

    sm = S.evaluate_suitability_balanced
    O.PERTURBED_JV_SIGMA = 0.05

    records = []
    with open(out_path, "w", encoding="utf-8") as out_f:
        for mask_level, mask_policy in PO_CONDITIONS:
            po_cond = (
                PartialObservabilityState(schedule_seed, mask_level, mask_policy)
                if (mask_level is not None and mask_policy is not None) else None
            )
            sm_for_matrix = po_cond.masked_scorer(sm) if po_cond is not None else sm
            matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYP)

            for method_name in ALL_METHODS:
                is_voting = method_name in VOTING_METHODS
                method_fn = (getattr(V, method_name) if is_voting
                             else getattr(O, method_name))
                _seed_rngs(_derived_seed(schedule_seed, "initial", method_name))
                if method_name == "murtys_task_allocation":
                    out, score, length, per = O.assign_tasks_with_method(method_fn, matrix)
                elif method_name == "perturbed_jv_task_allocation":
                    init_fn = O.perturbed_jv_factory(
                        _derived_seed(schedule_seed, "run", method_name, "pjv", "initial"))
                    out, score, length, per = O.assign_tasks_with_method(init_fn, matrix)
                else:
                    out, score, length, per = V.assign_tasks_with_voting(
                        robots, tasks, matrix, nc, method_fn,
                        tie_seed=_derived_seed(schedule_seed, "ties", "initial", method_name))

                # Collector on for the whole (initial + run) call sequence.
                V.UB_CALIBRATION = [] if is_voting else None
                try:
                    tup = benchmark_simulation(
                        output=out, robots=copy.deepcopy(robots),
                        tasks=copy.deepcopy(tasks),
                        num_candidates=nc, voting_method=method_fn, grid=grid,
                        map_dict=map_dict, suitability_method=sm,
                        suitability_matrix=matrix,
                        max_time_steps=max_steps, add_tasks=True, add_robots=True,
                        remove_robots=True, event_schedule=schedule,
                        mask_level=mask_level, mask_policy=mask_policy,
                        schedule_seed=schedule_seed,
                        run_seed=_derived_seed(schedule_seed, "run", method_name),
                        planner="lacam", scenario=scenario, scheduled_event_count=0,
                        horizon_binding_term="",
                    )
                finally:
                    calls = V.UB_CALIBRATION or []
                    V.UB_CALIBRATION = None
                base = {"map": map_name, "nr": nr, "nt": nt,
                        "scenario": scenario, "rep": rep,
                        "mask_level": mask_level, "mask_policy": mask_policy,
                        "method": method_name}
                for c in calls:
                    rec = dict(base)
                    rec.update(c)
                    records.append(rec)
        for rec in records:
            out_f.write(json.dumps(rec) + "\n")
    return out_path, len(records)


def build_jobs(outdir):
    jobs = []
    for map_name, r_sizes, t_sizes, reps in CALIB_CONFIGS:
        for nr in r_sizes:
            for nt in t_sizes:
                for scenario in SCENARIOS:
                    for rep in range(reps):
                        jobs.append({
                            "map": map_name, "nr": nr, "nt": nt,
                            "scenario": scenario, "rep": rep, "cost": 0,
                            "out": os.path.join(
                                outdir,
                                f"cal_{map_name}_{nr}x{nt}_{scenario}_rep{rep}.jsonl"),
                        })
    return jobs


def run_cmd(outdir):
    res_dir = os.path.join("hvbta", "io", "results", outdir)
    os.makedirs(res_dir, exist_ok=True)
    jobs = build_jobs(res_dir)
    print(f"{len(jobs)} groups; workers={N_WORKERS}")
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(min(N_WORKERS, len(jobs))) as pool:
        for path, n in pool.imap_unordered(run_group, jobs):
            done += 1
            print(f"  [{done}/{len(jobs)}] {os.path.basename(path)} "
                  f"({n} call records, {time.perf_counter()-t0:.0f}s)", flush=True)
    print("DONE - records in", res_dir)


def _bin_label(n):
    for lo, hi in [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "129+"


def _stats(vals):
    import numpy as np
    a = np.asarray(vals, dtype=float)
    if not a.size:
        return None
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std()),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
    }


def report_cmd(outdir):
    import numpy as np
    res_dir = os.path.join("hvbta", "io", "results", outdir)
    records = []
    for fn in sorted(os.listdir(res_dir)):
        if fn.startswith("cal_") and fn.endswith(".jsonl"):
            with open(os.path.join(res_dir, fn), encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
    print(f"loaded {len(records)} call records")

    # ---- 1. pair-level suitability over all feasible pairs ----
    all_pairs = np.concatenate([np.asarray(r["pairs"]) for r in records]) \
        if records else np.array([])
    pair_stats = _stats(all_pairs)
    by_condition = {}
    for r in records:
        key = f'{r["mask_level"]}-{r["mask_policy"]}'
        by_condition.setdefault(key, []).extend(r["pairs"])
    pair_by_condition = {k: _stats(v) for k, v in by_condition.items()}

    # ---- 2/3. non-urgent full-pool calls ----
    nu = [r for r in records
          if not r["urgent"] and not r["budget_fired"] and not r["threshold_fired"]]
    wq = _stats([r["winner_quality"] for r in nu])
    ratio_robot = _stats([r["winner_quality"] / r["ub_robot"] for r in nu
                          if r["ub_robot"]])
    ratio_ub = _stats([r["winner_quality"] / r["ub"] for r in nu if r["ub"]])

    # bins of (unassigned robots, unassigned tasks) in the call
    bins = {}
    for r in nu:
        key = (_bin_label(r["n_robots"]), _bin_label(r["n_tasks"]))
        bins.setdefault(key, {"robot": [], "ub": []})
        if r["ub_robot"]:
            bins[key]["robot"].append(r["winner_quality"] / r["ub_robot"])
        if r["ub"]:
            bins[key]["ub"].append(r["winner_quality"] / r["ub"])
    bin_stats = {f"{rk}x{tk}": {"n": len(v["ub"]),
                                "ratio_ub_robot": _stats(v["robot"]),
                                "ratio_ub": _stats(v["ub"])}
                 for (rk, tk), v in sorted(bins.items())}

    # coverage of UB by the robot-side bound alone (how often UB == UB_robot)
    ub_binds_robot = sum(1 for r in nu if r["ub"] == r["ub_robot"])

    report = {
        "n_records": len(records),
        "n_non_urgent_full_pool": len(nu),
        "pair_suitability": pair_stats,
        "pair_suitability_by_condition": pair_by_condition,
        "winner_quality_non_urgent": wq,
        "ratio_winner_over_ub_robot": ratio_robot,
        "ratio_winner_over_ub": ratio_ub,
        "ub_binds_robot_share": (ub_binds_robot / len(nu)) if nu else None,
        "ratio_by_bins": bin_stats,
        "max_feasible_stats": _stats([r["max_feasible"] for r in nu]),
        "call_dim_stats": {
            "n_robots": _stats([r["n_robots"] for r in nu]),
            "n_tasks": _stats([r["n_tasks"] for r in nu]),
        },
    }
    out_json = os.path.join(res_dir, "ub_calibration_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    def fmt(s):
        if not s:
            return "n/a"
        return (f"n={s['n']} mean={s['mean']:.3f} sd={s['sd']:.3f} "
                f"p5={s['p5']:.3f} p25={s['p25']:.3f} p50={s['p50']:.3f} "
                f"p75={s['p75']:.3f} p95={s['p95']:.3f}")

    print("\n== 1. Pair-level suitability, all feasible pairs ==")
    print("  ", fmt(pair_stats))
    for k, v in pair_by_condition.items():
        print(f"   {k:18s}", fmt(v))
    print("\n== 2. Winner coverage-normalized quality, non-urgent full-pool calls ==")
    print("  ", fmt(wq))
    print("\n== 3. Winner / UB ratios, non-urgent full-pool calls ==")
    print("   UB_robot alone:", fmt(ratio_robot))
    print("   UB = min       :", fmt(ratio_ub))
    print(f"   UB binds on robot side: {report['ub_binds_robot_share']:.1%} of calls")
    print(f"   max_feasible:  ", fmt(report["max_feasible_stats"]))
    print("\n   by (unassigned robots x unassigned tasks) bins:")
    for k, v in bin_stats.items():
        print(f"   {k:12s} n={v['n']:5d}  ub: {fmt(v['ratio_ub'])}")
    print("\nreport written to", out_json)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "pilot_ub_calibration"
    if cmd == "run":
        run_cmd(outdir)
    elif cmd == "report":
        report_cmd(outdir)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
