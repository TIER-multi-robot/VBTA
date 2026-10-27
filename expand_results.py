"""Expand the pool driver's packed-Output CSV into the named-column schema.

The pool driver (phase4_sweep.py) writes ONE packed column "Output" - a
stringified 59-item list (main_simulation's 58-item return tuple +
execution_time) - plus the group identity. This script expands it into the
named-column schema the serial writer produces (CSV_COLUMNS.md; exact
header order verified against the serial CSV), plus:

  - a deterministic Run ID per group ("{map}_{nr}x{nt}_{scenario}_rep{rep}"):
    same config -> same Run ID, independent of worker completion order.
    (The serial writer's Run ID is an on-disk row counter and is NOT
    deterministic across resumes; it is not reproducible from the pool data.)
  - Map and Rep as explicit columns (the serial files encode the map in the
    stamped per-map filenames and have no Rep column).

The initial-quality prefix columns (Total Score, Mean_Assignment_Suitability,
Init_Jains/Gini/Median/IQR) are not carried by the pool CSV: they are
REGENERATED from the seeds, replicating the serial writer's exact path
(per-method initial-assignment RNG isolation; PO true-score rule in
_record_initial; the perturbed-JV initial factory; the all-zero guard).
Planning_Time_us is filled from the pool row's Init_Alloc_us (wall-clock).

Planner is inferred from the map (hyper maps -> "pp", everything else
"lacam") - the same rule sweep_report.py uses. Mask_Count follows the
serial writer (MASK_COUNTS lookup; -1 when the mask is absent).

Profiles: the parallel path never writes the profiles CSV. This script
regenerates it from the entities seed, one row per group, keyed by the
same Run ID, and reports the join check (every simulation row must match
exactly one profiles row).

The raw phase4_simulation.csv is kept as is; the expanded CSV and the
profiles CSV are written alongside it (phase4_expanded.csv /
phase4_profiles.csv).

Usage:
  python expand_results.py <results_dir>            # expand + profiles + join check
  python expand_results.py --verify <results_dir>   # + serial-group verification:
      re-runs the serial script for (lak102d 10x10, rep 0) and asserts that
      every expanded column matches the serial CSV exactly (excluding
      wall-clock timing columns and the serial's counter Run ID).

Schema stability: every row's Output length is asserted == 59 (58 main +
exec). If main_simulation's return tuple ever changes, this fails loudly -
that is the schema-version guard (no version number needed while the
producer code is frozen; the remaining sweeps run from commit 03ffbd6).
"""
import ast
import csv
import json
import multiprocessing as mp
import os
import sys
import zlib

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

OUTPUT_LEN = 59  # main_simulation's 58-item return tuple + execution_time

# Serial writer header, verified against verify_serial_noise_v4's CSV
# (84 columns: 14 prefix + 59 output + 10 config suffix + map size).
PREFIX_NAMES = [
    "Run ID", "Method", "Suitability Method", "Planner",
    "Num Robots", "Num Tasks", "Num Candidates",
    "Total Score", "Mean_Assignment_Suitability", "Planning_Time_us",
    "Init_Jains_Index", "Init_Gini", "Init_Median", "Init_IQR",
]
OUTPUT_NAMES = [
    "Task_Completion_Fraction", "total_reassignment_time",
    "Cumulative_Reassignment_Quality", "total_reassignments",
    "Total Time Steps", "Avg Reassignment Jains Index",
    "Attempted_Completion_Rate", "Tasks_Attempted", "Tasks_Spawned",
    "Completion_Step",
    "Throughput_Mean", "Throughput_Var",
    "Adaptation_Latency_Mean", "Adaptation_Latency_Max",
    "Workload_Variance", "Attempted_Workload_Variance",
    "Full_Schedule_Tasks_Spawned", "Full_Schedule_Robots_Max",
    "Percentage_Urgent_Mean", "Percentage_Urgent_Max",
    "Robot_Pressure_Fraction",
    "Avg_Reass_Gini", "Avg_Reass_Median", "Avg_Reass_IQR",
    "Mean_Task_Pool_Size", "Pool_Size_Trend",
    "Scheduled_Event_Count", "Horizon_Binding_Term",
    "Plan_Failure_Count", "Alternatives_Tried",
    "Mis_Assignment_Count", "Mis_Assignment_Rate",
    "Wasted_Travel_Steps", "Wasted_Travel_Fraction",
    "Masked_Starvation_Count", "Mean_Masked_Wait", "Reveal_Events",
    "Mis_Assignment_Recovery_Latency",
    "True_Assignment_Suitability", "Masked_View_Suitability",
    "Observed_View_Suitability",
    "Nav_Drop_Count", "Nav_Drop_Wasted_Travel",
    "Never_Attempted_Count", "All_Zero_Skip_Ticks",
    "Stuck_Assigned_Robot_Ticks",
    "Planned_Conflict_Count", "Safety_Nudge_Count",
    "Collision_Violation_Count", "Propagated_Wait_Count",
    "Mover_Conflict_Waits",
    "Run_Timed_Out",
    "Threshold_Fire_Rate", "Mean_Pool_Size_At_Rank",
    "Planner_Call_Count", "Planner_Total_ms", "Planner_Cap_Hits",
    "Planner_Dead_Tick",
    "Execution Time",
]
SUFFIX_NAMES = [
    "Scenario", "Task_Arrival_Rate", "Task_Cancellation_Rate",
    "Robot_Arrival_Rate", "Robot_Departure_Rate",
    "Mask_Level", "Mask_Count", "Mask_Policy",
    "Obs_Noise_Sigma", "Noise_Experiment", "Map Size",
]
EXPANDED_HEADER = ["Map", "Rep"] + PREFIX_NAMES + OUTPUT_NAMES + SUFFIX_NAMES
assert len(EXPANDED_HEADER) == 2 + len(PREFIX_NAMES) + OUTPUT_LEN + len(SUFFIX_NAMES)

# Wall-clock columns: filled but never compared (the serial values differ by
# construction). Deterministic columns must match bit-for-bit.
TIMING_COLS = {"Planning_Time_us", "total_reassignment_time",
               "Planner_Total_ms", "Execution Time"}

PROFILES_HEADER = ["Run_ID", "Map", "Num Robots", "Num Tasks",
                   "Suitability Method", "RobotProfiles", "TaskProfiles"]

SM_NAME = "evaluate_suitability_balanced"
HYPER_MAPS = {"hrt003d.map", "lak103c.map", "lak108c.map"}  # planner = pp


def _parse_output(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return eval(cell, {"nan": float("nan"), "inf": float("inf"),
                           "np": __import__("numpy"), "__builtins__": {}})


def make_run_id(map_name, nr, nt, scenario, rep):
    """Deterministic per-group Run ID, independent of worker order."""
    return f"{map_name}_{nr}x{nt}_{scenario}_rep{rep}"


def load_raw(dir_path):
    """Read the pool CSV; assert the Output tuple length on EVERY row."""
    csv_path = os.path.join(dir_path, "phase4_simulation.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"no phase4_simulation.csv in {dir_path}")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line_no, row in enumerate(r, 2):
            out = _parse_output(row["Output"])
            if len(out) != OUTPUT_LEN:
                raise SystemExit(
                    f"SCHEMA MISMATCH {csv_path}:{line_no}: Output has "
                    f"{len(out)} items, expected {OUTPUT_LEN} - the producer "
                    f"code changed; refuse to expand")
            rows.append({
                "map": row["Map"], "nr": int(row["Num Robots"]),
                "nt": int(row["Num Tasks"]), "scenario": row["Scenario"],
                "rep": int(row["Rep"]), "method": row["Method"],
                "out": out, "map_size": row["Map Size"],
                "init_alloc_us": float(row["Init_Alloc_us"]),
                "mask_level": row["Mask_Level"],
                "mask_policy": row["Mask_Policy"],
                "sigma": row["Obs_Noise_Sigma"],
            })
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Regeneration of the per-group entities and the initial-assignment prefix
# (replicates the serial writer's exact code path).
# ---------------------------------------------------------------------------
def _regen_group(key, conds):
    """key = (map, nr, nt, scenario, rep). conds = sorted list of
    (mask_level, mask_policy, sigma_str) present in the raw CSV for the
    group. Returns (profiles_row, prefix_map).

    prefix_map[(mask_level, mask_policy, sigma_str, method)] =
    (total_score, mean_suit, jains, gini, median, iqr) - deterministic.
    """
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    from hvbta.simulation.main_simulation import (
        _derived_seed, _seed_rngs, compute_all_fairness_metrics,
        suitability_all_zero,
    )
    from hvbta.pathfinding.CBS_helper import load_map
    from hvbta.partial_observability import (
        PartialObservabilityState, noisy_scorer, MASK_COUNTS,
    )

    map_name, nr, nt, scenario, rep = key
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    hyp = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    # Serial writer semantics: 'Map Size' holds the size class derived from
    # the map diagonal, not a cell count.
    map_class = "Small" if hyp < 55 else "Medium" if hyp < 105 else "Large"
    nc = max(nr, nt) * 6
    seed_tuple = (map_name, nr, nt, nc, SM_NAME, rep, scenario)
    schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF

    _seed_rngs(_derived_seed(schedule_seed, "entities"))
    robot_locs, robots = set(), []
    for idx in range(nr):
        r = G.generate_random_robot_profile_strict(f"R{idx+1}", grid, robot_locs)
        robots.append(r)
        robot_locs.add(r.location)
    task_locs, tasks = set(), []
    for idx in range(nt):
        t = G.generate_random_task_description_strict(f"T{idx+1}", grid, task_locs, [])
        tasks.append(t)
        task_locs.add(t.location)

    profiles_row = [
        make_run_id(map_name, nr, nt, scenario, rep), map_name, nr, nt,
        SM_NAME,
        json.dumps([r.strict_profile_name for r in robots]),
        json.dumps([t.strict_profile_name for t in tasks]),
    ]

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

    # Mirror run_group's hard-set (the analysis-plan amendment's 0.05):
    # without this the module default is None and the perturbed-JV initial
    # solve would run unperturbed, unlike every sweep run.
    O.PERTURBED_JV_SIGMA = 0.05

    sm = S.evaluate_suitability_balanced
    prefix = {}
    for mask_level, mask_policy, sigma_str in conds:
        sigma = float(sigma_str) if sigma_str else None
        po_cond = (PartialObservabilityState(schedule_seed, mask_level, mask_policy)
                   if mask_level else None)
        sm_for_matrix = po_cond.masked_scorer(sm) if po_cond is not None else sm
        if sigma is not None:
            sm_for_matrix = noisy_scorer(sm_for_matrix, sigma, schedule_seed, [0])
        matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, hyp)

        if suitability_all_zero(matrix):
            # Serial writer's 2.7 guard: empty initial assignment, score 0.
            for pretty, _ in methods:
                per_agent = []
                total = 0.0
                jains, gini, med, iqr = compute_all_fairness_metrics(per_agent)
                prefix[(mask_level, mask_policy, sigma_str, pretty)] = (
                    total, (total / len(per_agent) if per_agent else 0.0),
                    jains, gini, med, iqr)
            continue

        for pretty, method in methods:
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
            # Serial _record_initial: under PO conditions the reported
            # initial quality is computed on TRUE (un-masked / un-noised)
            # pair scores, not the allocator's view.
            if po_cond is None or not out:
                per_agent = per
                total = score
            else:
                pairs = out[0] if isinstance(out, tuple) else out
                per_agent = [S._balanced_pair(robots[r], tasks[t], map_size=hyp)
                             for r, t in pairs]
                total = sum(per_agent)
            jains, gini, med, iqr = compute_all_fairness_metrics(per_agent)
            prefix[(mask_level, mask_policy, sigma_str, pretty)] = (
                total, (total / len(per_agent) if per_agent else 0.0),
                jains, gini, med, iqr)
    return profiles_row, prefix, map_class


def _worker(payload):
    key, conds = payload
    return key, _regen_group(key, conds)


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------
def expand(dir_path):
    rows = load_raw(dir_path)
    from hvbta.simulation.main_simulation import REGIMES
    from hvbta.partial_observability import MASK_COUNTS

    # Deterministic expanded order: group key, then condition, then method.
    rows.sort(key=lambda r: (r["map"], r["nr"], r["nt"], r["scenario"],
                             r["rep"], r["mask_level"], r["mask_policy"],
                             r["sigma"], r["method"]))

    groups = sorted({(r["map"], r["nr"], r["nt"], r["scenario"], r["rep"])
                     for r in rows})
    run_ids = [make_run_id(*g) for g in groups]
    if len(set(run_ids)) != len(run_ids):
        raise SystemExit("Run ID collision across groups - refusing to write")

    group_conds = {}
    for r in rows:
        g = (r["map"], r["nr"], r["nt"], r["scenario"], r["rep"])
        group_conds.setdefault(g, set()).add(
            (r["mask_level"], r["mask_policy"], r["sigma"]))

    n_workers = max(1, (os.cpu_count() or 4) - 2)
    payloads = [(g, sorted(group_conds[g])) for g in groups]
    with mp.Pool(min(n_workers, len(groups))) as pool:
        regen = dict(pool.map(_worker, payloads))

    profiles_rows = [regen[g][0] for g in groups]
    expanded = []
    for r in rows:
        g = (r["map"], r["nr"], r["nt"], r["scenario"], r["rep"])
        profiles_row, prefix, map_class = regen[g]
        mask_level = r["mask_level"]
        sigma_str = r["sigma"]
        sigma = float(sigma_str) if sigma_str else None
        pretty = r["method"]
        key = (mask_level, r["mask_policy"], sigma_str, pretty)
        if key not in prefix:
            raise SystemExit(f"missing regenerated prefix for {g} {key}")
        total, mean_suit, jains, gini, med, iqr = prefix[key]
        rates = REGIMES[r["scenario"]]
        row = {
            "Map": r["map"], "Rep": r["rep"],
            "Run ID": make_run_id(*g),
            "Method": pretty,
            "Suitability Method": SM_NAME,
            "Planner": "pp" if r["map"] in HYPER_MAPS else "lacam",
            "Num Robots": r["nr"], "Num Tasks": r["nt"],
            "Num Candidates": max(r["nr"], r["nt"]) * 6,
            "Total Score": total, "Mean_Assignment_Suitability": mean_suit,
            "Planning_Time_us": r["init_alloc_us"],
            "Init_Jains_Index": jains, "Init_Gini": gini,
            "Init_Median": med, "Init_IQR": iqr,
            "Scenario": r["scenario"],
            "Task_Arrival_Rate": rates["task_arrival_rate"],
            "Task_Cancellation_Rate": rates["task_cancellation_rate"],
            "Robot_Arrival_Rate": rates["robot_arrival_rate"],
            "Robot_Departure_Rate": rates["robot_departure_rate"],
            "Mask_Level": mask_level,
            "Mask_Count": (MASK_COUNTS.get(mask_level, -1) if mask_level else -1),
            "Mask_Policy": r["mask_policy"],
            "Obs_Noise_Sigma": sigma_str,
            "Noise_Experiment": (1 if sigma is not None else 0),
            "Map Size": map_class,
        }
        for name, value in zip(OUTPUT_NAMES, r["out"]):
            row[name] = value
        expanded.append(row)

    out_csv = os.path.join(dir_path, "phase4_expanded.csv")
    prof_csv = os.path.join(dir_path, "phase4_profiles.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EXPANDED_HEADER)
        w.writeheader()
        for row in expanded:
            w.writerow(row)
    with open(prof_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(PROFILES_HEADER)
        for row in profiles_rows:
            w.writerow(row)

    # ---- join check: (Run ID, Map) sim <-> profiles, both directions ----
    prof_keys = {(row[0], row[1]) for row in profiles_rows}
    sim_keys = {(row["Run ID"], row["Map"]) for row in expanded}
    unmatched_sim = len(expanded) - sum(1 for row in expanded
                                        if (row["Run ID"], row["Map"]) in prof_keys)
    orphan_prof = sorted(prof_keys - sim_keys)
    print(f"raw rows: {len(rows)}; groups: {len(groups)}")
    print(f"expanded -> {out_csv} ({len(expanded)} rows x {len(EXPANDED_HEADER)} cols)")
    print(f"profiles -> {prof_csv} ({len(profiles_rows)} rows)")
    print(f"join check: simulation rows without a profiles row: {unmatched_sim}"
          f" (must be 0); orphan profiles rows: {len(orphan_prof)}")
    if unmatched_sim or orphan_prof:
        raise SystemExit("JOIN CHECK FAILED")
    print("join check: PASS (every simulation row matches exactly one profiles row)")
    return expanded, profiles_rows


# ---------------------------------------------------------------------------
# Verification: one group through the serial script vs the expanded pool rows
# ---------------------------------------------------------------------------
def _float_eq(a, b):
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))
    except (ValueError, TypeError):
        return str(a) == str(b)


def verify(dir_path):
    """Run the serial script for (lak102d 10x10, rep 0, noise conditions),
    then compare every expanded column against the serial CSV."""
    import subprocess

    serial_dir = "verify_serial_expand_v5"
    env = dict(os.environ)
    env["PATH"] = r"C:\msys64\ucrt64\bin;" + env.get("PATH", "")
    env["PYTHONPATH"] = "."
    env["HVBTA_OUTDIR"] = serial_dir
    env["HVBTA_REPS"] = "1"
    env["HVBTA_SIZES"] = "10"
    env["HVBTA_NOISE"] = "1"
    env["HVBTA_PJV_SIGMA"] = "0.05"
    env.pop("HVBTA_PO", None)
    env.pop("HVBTA_PLANNER", None)
    env.pop("HVBTA_STALL", None)
    env.pop("HVBTA_NOISE_SIGMAS", None)
    res_dir = os.path.join("hvbta", "io", "results", serial_dir)
    if not os.path.isdir(res_dir):
        print("running the serial script for lak102d 10x10 (fresh v5 dir)...")
        r = subprocess.run(
            [sys.executable, "-m", "hvbta.simulation.main_simulation"],
            env=env, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise SystemExit(f"serial run failed:\n{r.stderr[-4000:]}")

    serial_rows = {}
    import glob
    for path in sorted(glob.glob(os.path.join(res_dir, "*simulation*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["Method"], row["Scenario"], row["Obs_Noise_Sigma"])
                serial_rows[key] = row

    raw = load_raw(dir_path)
    pool_group = [r for r in raw
                  if (r["map"], r["nr"], r["nt"], r["scenario"], r["rep"])
                  == ("lak102d.map", 10, 10, r["scenario"], 0)]
    group_keys = sorted({(r["map"], r["nr"], r["nt"], r["scenario"], r["rep"])
                         for r in pool_group})
    print(f"pool rows for the verification group(s): {len(pool_group)} "
          f"({len(group_keys)} groups); serial rows: {len(serial_rows)}")
    if len(pool_group) != len(serial_rows) or not pool_group:
        raise SystemExit("row-count mismatch between pool group and serial run")

    by_group = {}
    for r in pool_group:
        by_group.setdefault((r["map"], r["nr"], r["nt"], r["scenario"],
                              r["rep"]), []).append(r)
    regen = {}
    map_classes = {}
    for g in group_keys:
        conds = sorted({(r["mask_level"], r["mask_policy"], r["sigma"])
                         for r in by_group[g]})
        _, prefix, map_classes[g] = _regen_group(g, conds)
        regen[g] = prefix

    from hvbta.simulation.main_simulation import REGIMES
    from hvbta.partial_observability import MASK_COUNTS

    checked = mismatches = 0
    for r in sorted(pool_group,
                    key=lambda r: (r["scenario"], r["sigma"], r["method"])):
        g = (r["map"], r["nr"], r["nt"], r["scenario"], r["rep"])
        prefix = regen[g]
        total, mean_suit, jains, gini, med, iqr = prefix[
            (r["mask_level"], r["mask_policy"], r["sigma"], r["method"])]
        rates = REGIMES[r["scenario"]]
        exp = dict(zip(OUTPUT_NAMES, r["out"]))
        exp.update({
            "Map": r["map"], "Rep": r["rep"],
            "Method": r["method"], "Suitability Method": SM_NAME,
            "Planner": "lacam",
            "Num Robots": r["nr"], "Num Tasks": r["nt"],
            "Num Candidates": max(r["nr"], r["nt"]) * 6,
            "Total Score": total, "Mean_Assignment_Suitability": mean_suit,
            "Init_Jains_Index": jains, "Init_Gini": gini,
            "Init_Median": med, "Init_IQR": iqr,
            "Scenario": r["scenario"],
            "Task_Arrival_Rate": rates["task_arrival_rate"],
            "Task_Cancellation_Rate": rates["task_cancellation_rate"],
            "Robot_Arrival_Rate": rates["robot_arrival_rate"],
            "Robot_Departure_Rate": rates["robot_departure_rate"],
            "Mask_Level": r["mask_level"],
            "Mask_Count": (MASK_COUNTS.get(r["mask_level"], -1)
                           if r["mask_level"] else -1),
            "Mask_Policy": r["mask_policy"],
            "Obs_Noise_Sigma": r["sigma"],
            "Noise_Experiment": (1 if r["sigma"] else 0),
            "Map Size": map_classes[g],
        })
        skey = (r["method"], r["scenario"], r["sigma"])
        if skey not in serial_rows:
            raise SystemExit(f"no serial row for {skey}")
        srow = serial_rows[skey]
        checked += 1
        for col, val in exp.items():
            if col in TIMING_COLS or col in ("Run ID", "Map", "Rep"):
                continue  # wall-clock / serial-counter / pool-only columns
            if col not in srow:
                raise SystemExit(f"serial CSV lacks column {col!r}")
            if not _float_eq(val, srow[col]):
                mismatches += 1
                if mismatches <= 10:
                    print(f"  MISMATCH {skey} col {col}: "
                          f"expanded={val!r} serial={srow[col]!r}")
    print(f"verification: {checked} rows x "
          f"{len(EXPANDED_HEADER) - len(TIMING_COLS) - 3} compared columns; "
          f"mismatches: {mismatches}")
    if mismatches or checked == 0:
        raise SystemExit("VERIFICATION FAILED")
    print("VERIFICATION PASSED: expanded pool rows == serial CSV "
          "(excluding wall-clock timing columns and the serial Run ID counter)")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        verify(args[0] if args else ".")
    else:
        args = sys.argv[1:]
        expand(args[0] if args else ".")
