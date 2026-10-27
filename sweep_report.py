"""Post-sweep report: row counts, invariant checks, allocation time per call.

Standing report run after EVERY sweep (user requirement, 2026-10-26):
  - row counts by method / map
  - invariant checks: Planned_Conflict_Count, Safety_Nudge_Count,
    Collision_Violation_Count, Mover_Conflict_Waits (excepted for the
    PP-stress sweep), Stuck_Assigned_Robot_Ticks all 0;
    Mis_Assignment_Count 0 at mask none AND under pessimistic policies;
    Run_Timed_Out 0 (the 600 s safety stop must never fire);
    Planner_Cap_Hits summary (deterministic planner-work cap firings).
  - Planning_Time_us per allocation call by method:
    (initial alloc time + total_reassignment_time) / (1 + reassignments).

Handles both CSV formats:
  - the serial sweep's stamped 80-column files (header names, one or more
    '*simulation*.csv' per map in the directory)
  - the pool driver's phase4_simulation.csv (Output = bracketed tuple)

Usage: python sweep_report.py <results_dir> [--pp-stress]
"""
import ast
import csv
import glob
import os
import sys

import numpy as np

OUT_IDX = {
    "completion": 0, "reassign_time": 1, "reassignments": 3, "ticks": 4,
    "failures": 28, "alts": 29, "mis": 30,
    "stuck": 45, "conflicts": 46, "nudge": 47, "violations": 48,
    "prop": 49, "mover": 50, "timedout": 51,
    "plcalls": 54, "plms": 55, "cap_hits": 56, "exec": 57,
}


def _parse_output(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return eval(cell, {"nan": float("nan"), "inf": float("inf"),
                           "np": np, "__builtins__": {}})


def load_rows(dir_path):
    """Returns (rows, format). Each row: dict with identity + named metrics."""
    # pool format first
    pool = os.path.join(dir_path, "phase4_simulation.csv")
    if os.path.exists(pool):
        rows = []
        with open(pool, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                o = _parse_output(r["Output"])
                rows.append({
                    "map": r["Map"], "nr": int(r["Num Robots"]),
                    "nt": int(r["Num Tasks"]), "scenario": r["Scenario"],
                    "rep": int(r["Rep"]), "method": r["Method"],
                    "planner": "pp" if r["Map"] in
                               ("hrt003d.map", "lak103c.map", "lak108c.map") else "lacam",
                    "mask_level": r.get("Mask_Level") or "none",
                    "mask_policy": r.get("Mask_Policy") or "none",
                    "init_alloc_us": float(r.get("Init_Alloc_us") or 0.0),
                    **{k: o[i] for k, i in OUT_IDX.items()},
                })
        return rows, "phase4"
    rows = []
    for path in sorted(glob.glob(os.path.join(dir_path, "*simulation*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "map": r.get("Map") or os.path.basename(path),
                    "nr": int(r["Num Robots"]), "nt": int(r["Num Tasks"]),
                    "scenario": r["Scenario"], "rep": -1, "method": r["Method"],
                    "planner": r["Planner"],
                    "mask_level": r.get("Mask_Level") or "",
                    "mask_policy": r.get("Mask_Policy") or "",
                    "init_alloc_us": float(r.get("Planning_Time_us") or 0.0),
                    "reassign_time": float(r["total_reassignment_time"]),
                    "reassignments": int(r["total_reassignments"]),
                    "ticks": int(r["Total Time Steps"]),
                    "failures": int(r["Plan_Failure_Count"]),
                    "alts": int(r["Alternatives_Tried"]),
                    "mis": int(r["Mis_Assignment_Count"]),
                    "stuck": int(r["Stuck_Assigned_Robot_Ticks"]),
                    "conflicts": int(r["Planned_Conflict_Count"]),
                    "nudge": int(r["Safety_Nudge_Count"]),
                    "violations": int(r["Collision_Violation_Count"]),
                    "prop": int(r["Propagated_Wait_Count"]),
                    "mover": int(r["Mover_Conflict_Waits"]),
                    "timedout": int(r["Run_Timed_Out"]),
                    "plcalls": int(r["Planner_Call_Count"]),
                    "plms": float(r["Planner_Total_ms"]),
                    "cap_hits": int(r["Planner_Cap_Hits"]),
                    "exec": float(r["Execution Time"]),
                    "mask_ok": (r.get("Mask_Level") or "") in ("", "none"),
                })
        return rows, "serial"
    return [], "none"


def report(dir_path, pp_stress=False):
    rows, fmt = load_rows(dir_path)
    if not rows:
        raise SystemExit(f"no CSVs found in {dir_path}")
    print(f"=== sweep report: {dir_path} (format: {fmt}) ===")
    print(f"rows: {len(rows)}")
    print("by map:", dict(sorted(
        (m, sum(1 for r in rows if r["map"] == m)) for m in {r["map"] for r in rows})))

    # ---- invariant checks ----
    print("\n--- invariant checks ---")
    ok = True
    for k in ("conflicts", "nudge", "violations", "stuck"):
        bad = sum(1 for r in rows if r[k])
        print(f"  {k:12s}: nonzero on {bad} rows{'  <-- VIOLATION' if bad else ''}")
        ok = ok and not bad
    bad = sum(1 for r in rows if r["mover"])
    exempt = pp_stress
    print(f"  mover_waits  : nonzero on {bad} rows"
          f"{' (PP-stress exempt)' if exempt and bad else ''}"
          f"{'  <-- VIOLATION' if bad and not exempt else ''}")
    ok = ok and (not bad or exempt)
    # Mis_Assignment_Count must be 0 at mask none and under pessimistic
    mis_bad = sum(1 for r in rows if r["mis"] and
                  ((r.get("mask_level") in ("", "none"))
                   or (r.get("mask_policy") == "pessimistic")))
    print(f"  mis_assign   : nonzero on {mis_bad} rows at none/pessimistic"
          f"{'  <-- VIOLATION' if mis_bad else ''}")
    ok = ok and not mis_bad
    timed = sum(1 for r in rows if r["timedout"])
    print(f"  run_timed_out: {timed} rows hit the 600 s safety stop"
          f"{'  <-- VIOLATION' if timed else ''}")
    ok = ok and not timed
    cap_rows = sum(1 for r in rows if r["cap_hits"])
    cap_total = sum(r["cap_hits"] for r in rows)
    print(f"  planner cap hits: {cap_total} on {cap_rows} rows "
          f"(deterministic planner-work caps; expected 0 on LaCAM sweeps)")
    print(f"  propagated waits: {sum(r['prop'] for r in rows)} total "
          f"(informational)")

    # ---- allocation time per call by method (4f) ----
    print("\n--- allocation time per call by method ---")
    print(f"  {'method':40s} {'calls/run':>9s} {'us/call mean':>12s} "
          f"{'us/call p50':>11s} {'us/call p95':>11s}")
    for m in sorted({r["method"] for r in rows}):
        per_call = []
        calls = []
        for r in rows:
            if r["method"] != m:
                continue
            n_calls = 1 + int(r["reassignments"])
            t_us = r["init_alloc_us"] + float(r["reassign_time"])
            per_call.append(t_us / n_calls)
            calls.append(n_calls)
        a = np.asarray(per_call)
        print(f"  {m:40s} {np.mean(calls):9.1f} {a.mean():12.1f} "
              f"{np.percentile(a, 50):11.1f} {np.percentile(a, 95):11.1f}")

    # ---- planner call stats (context) ----
    print("\n--- planner calls (context) ---")
    for m in sorted({r["method"] for r in rows}):
        g = [r for r in rows if r["method"] == m]
        mc = np.mean([r["plms"] / max(1, r["plcalls"]) for r in g])
        print(f"  {m:40s} calls/run={np.mean([r['plcalls'] for r in g]):6.0f} "
              f"ms/call={mc:8.1f} fails/run={np.mean([r['failures'] for r in g]):6.1f}")

    print(f"\nOVERALL: {'ALL INVARIANTS PASS' if ok else 'INVARIANT VIOLATIONS PRESENT'}")
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pp = "--pp-stress" in sys.argv
    report(args[0] if args else ".", pp_stress=pp)
