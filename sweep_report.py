"""Post-sweep report: row counts, invariant checks, allocation time per call.

Standing report run after EVERY sweep (user requirement, 2026-10-26):
  - row counts by method / map
  - invariant checks: Planned_Conflict_Count, Safety_Nudge_Count,
    Collision_Violation_Count, Mover_Conflict_Waits (excepted for the
    PP-stress sweep), Stuck_Assigned_Robot_Ticks all 0;
    Mis_Assignment_Count 0 at mask none AND under pessimistic policies;
    Run_Timed_Out 0 (the 600 s safety stop must never fire);
    Planner_Cap_Hits summary (deterministic planner-work cap firings);
    planner-dead runs per method (PLANNER_FAILED_CALL_BUDGET exhausted:
    count, dead tick mean/median, mean completion dead vs other runs).
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
    "plcalls": 54, "plms": 55, "cap_hits": 56,
    # dead_tick (57) and exec (58) handled explicitly below: pre-2026-10-27
    # CSVs have a 58-item tuple (no Planner_Dead_Tick) and must still load.
}


def _parse_output(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return eval(cell, {"nan": float("nan"), "inf": float("inf"),
                           "np": np, "__builtins__": {}})


def _load_named(paths):
    """Named-column format: the serial stamped files AND the unified pool
    CSV (2026-10-27+; same schema plus Map/Rep columns)."""
    rows = []
    for path in paths:
        base = os.path.basename(path)
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                got_map = r.get("Map") or ""
                rows.append({
                    "map": got_map or base,
                    "nr": int(r["Num Robots"]), "nt": int(r["Num Tasks"]),
                    "scenario": r["Scenario"],
                    "rep": int(r.get("Rep") or -1),
                    "method": r["Method"],
                    "planner": (r.get("Planner") or
                                ("pp" if (got_map or base) in
                                 ("hrt003d.map", "lak103c.map", "lak108c.map")
                                 else "lacam")),
                    "mask_level": r.get("Mask_Level") or "none",
                    "mask_policy": r.get("Mask_Policy") or "none",
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
                    "dead_tick": (int(r["Planner_Dead_Tick"])
                                  if r.get("Planner_Dead_Tick") else None),
                    "exec": float(r["Execution Time"]),
                    "mask_ok": (r.get("Mask_Level") or "") in ("", "none"),
                })
    return rows


def load_rows(dir_path):
    """Returns (rows, format). Each row: dict with identity + named metrics.

    Formats, in detection order:
      - "phase4": the pool driver's phase4_simulation.csv. Pre-2026-10-27
        dirs pack the output tuple into one stringified "Output" column;
        the unified format (2026-10-27+) is the full named-column schema
        and is loaded via _load_named.
      - "serial": the serial sweep's stamped per-map files.
    """
    pool = os.path.join(dir_path, "phase4_simulation.csv")
    if os.path.exists(pool):
        with open(pool, newline="", encoding="utf-8") as f:
            first = next(csv.reader(f), [])
        if "Output" in first:
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
                        # 59-item tuple (2026-10-27+): dead_tick = o[57],
                        # exec = o[58]. Legacy 58-item tuple: no
                        # Planner_Dead_Tick.
                        "dead_tick": (int(o[57]) if len(o) >= 59 else None),
                        "exec": float(o[-1]),
                        **{k: o[i] for k, i in OUT_IDX.items()},
                    })
            return rows, "phase4"
        return _load_named([pool]), "phase4"
    paths = sorted(glob.glob(os.path.join(dir_path, "*simulation*.csv")))
    if not paths:
        return [], "none"
    return _load_named(paths), "serial"


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

    # ---- planner-dead runs (PLANNER_FAILED_CALL_BUDGET exhausted) ----
    # Per method: how many runs went planner-dead, the tick it happened
    # (mean / median over dead runs), and mean completion for dead vs
    # other runs.
    print("\n--- planner-dead runs (failed-call budget exhausted) ---")
    unknown = sum(1 for r in rows if r.get("dead_tick") is None)
    if unknown == len(rows):
        print("  n/a: these CSVs predate the Planner_Dead_Tick column")
    else:
        if unknown:
            print(f"  note: {unknown} legacy rows without Planner_Dead_Tick"
                  " are counted as not dead")
        any_dead = any((r.get("dead_tick") or -1) >= 0 for r in rows)
        if not any_dead:
            print("  none: no run exhausted PLANNER_FAILED_CALL_BUDGET")
        else:
            print(f"  {'method':40s} {'dead':>5s} {'/runs':>6s} "
                  f"{'tick mean':>9s} {'tick p50':>9s} "
                  f"{'compl dead':>10s} {'compl other':>11s}")
            for m in sorted({r["method"] for r in rows}):
                g = [r for r in rows if r["method"] == m]
                dead = [r for r in g if (r.get("dead_tick") or -1) >= 0]
                other = [r for r in g if (r.get("dead_tick") or -1) < 0]
                if dead:
                    ticks = [r["dead_tick"] for r in dead]
                    cd = float(np.mean([r["completion"] for r in dead]))
                    tm = float(np.mean(ticks))
                    tp = float(np.percentile(ticks, 50))
                else:
                    ticks, cd, tm, tp = [], float("nan"), float("nan"), float("nan")
                co = (float(np.mean([r["completion"] for r in other]))
                      if other else float("nan"))
                print(f"  {m:40s} {len(dead):5d} {len(g):6d} "
                      f"{tm:9.1f} {tp:9.1f} {cd:10.3f} {co:11.3f}")

    print(f"\nOVERALL: {'ALL INVARIANTS PASS' if ok else 'INVARIANT VIOLATIONS PRESENT'}")
    return ok


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pp = "--pp-stress" in sys.argv
    report(args[0] if args else ".", pp_stress=pp)
