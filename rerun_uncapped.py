"""Uncapped re-run of wall-stop truncated groups (2026-10-28 amendment).

Re-runs every driver group (map, nr, nt, scenario, rep) whose source sweep
contains a Run_Timed_Out = 1 row, all 8 methods, identical code and seeds
(calls phase4_sweep.run_group unchanged), with the per-run wall stop raised
via HVBTA_RUN_WALL_LIMIT_S (env read once at main_simulation import).

Rows of AFFECTED condition-level groups (mask level/policy/sigma, i.e. the
grouping the analysis pairs on) are written to a fresh output directory.
The re-computed rows of unaffected conditions are compared against the
source sweep on all shared non-timing columns before being discarded
(count reported). Every re-run row whose source twin was NOT truncated is
also compared the same way - the identity check the amendment records.

Usage:
  python rerun_uncapped.py run <sweep> <source_dir> <outdir> <wall_limit_s>
  python rerun_uncapped.py verify <source_dir> <outdir>

<sweep> is a phase4_sweep sweep name ("po", "balanced", ...). Source and
output live under hvbta/io/results/. Resume: completed driver groups in
the output CSV are skipped, per-group comparison results accumulate in
<outdir>/comparisons.jsonl.
"""

import csv
import json
import os
import sys
import time
import multiprocessing as mp

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

TIMING_COLS = {"Planning_Time_us", "total_reassignment_time",
               "Planner_Total_ms", "Execution Time", "Planner_Timeout_Count"}
KEY_COLS = ("Map", "Num Robots", "Num Tasks", "Scenario", "Rep",
            "Mask_Level", "Mask_Policy", "Obs_Noise_Sigma", "Method")


def results_csv(dir_name):
    return os.path.join("hvbta", "io", "results", dir_name,
                        "phase4_simulation.csv")


def load_source(source_dir):
    """Source rows keyed by the full (group, condition, method) key."""
    path = results_csv(source_dir)
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames
        src = {tuple(r[c] for c in KEY_COLS): r for r in rdr}
    if len(src) != sum(1 for _ in open(path, encoding="utf-8")) - 1:
        raise SystemExit("duplicate keys in source CSV")
    return header, src


def group_estimate(src_rows):
    """Serial seconds for one driver group: measured exec for non-truncated
    rows, linear tick extrapolation (600 s x t_ref/t_stop) for truncated."""
    t_ref = max(int(r["Total Time Steps"]) for r in src_rows
                if r["Run_Timed_Out"] == "0")
    est = 0.0
    for r in src_rows:
        if r["Run_Timed_Out"] == "1":
            est += 600.0 * t_ref / int(r["Total Time Steps"])
        else:
            est += float(r["Execution Time"]) / 1e9
    return est


def compare_row(new_row, src_row, shared_cols, col_idx):
    """Columns (shared, non-timing) where the new row differs from source."""
    return [c for c in shared_cols
            if str(new_row[col_idx[c]]) != src_row[c]]


def cmd_run(sweep, source_dir, outdir, wall_limit_s):
    os.environ["HVBTA_RUN_WALL_LIMIT_S"] = str(int(wall_limit_s))
    import phase4_sweep as P
    from hvbta.simulation.main_simulation import RUN_WALL_LIMIT_S
    print(f"HVBTA_RUN_WALL_LIMIT_S={int(wall_limit_s)} -> "
          f"RUN_WALL_LIMIT_S={RUN_WALL_LIMIT_S}", flush=True)
    if sweep not in P.SWEEP_CONFIGS:
        raise SystemExit(f"unknown sweep: {sweep}")

    src_header, src = load_source(source_dir)
    affected_conds = {k[:8] for k, r in src.items() if r["Run_Timed_Out"] == "1"}
    by_drv = {}
    for k in src:
        by_drv.setdefault(k[:5], []).append(k)
    affected_drv = {d for d, keys in by_drv.items()
                    if any(k[:8] in affected_conds for k in keys)}
    shared_cols = [c for c in src_header
                   if c in P._COL_IDX and c not in TIMING_COLS]
    col_idx = P._COL_IDX

    res_dir = os.path.join("hvbta", "io", "results", outdir)
    csv_path = os.path.join(res_dir, "phase4_simulation.csv")
    prof_path = os.path.join(res_dir, "phase4_profiles.csv")
    jsonl_path = os.path.join(res_dir, "comparisons.jsonl")
    if os.path.exists(csv_path):
        done_drv = set()
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_drv.add(tuple(row[c] for c in KEY_COLS[:5]))
        mode = "a"
        print(f"resuming: {len(done_drv)} driver groups already in "
              f"{outdir}", flush=True)
    else:
        done_drv = set()
        os.makedirs(res_dir, exist_ok=True)
        mode = "w"

    jobs = []
    for drv in affected_drv:
        if drv in done_drv:
            continue
        m, nr, nt, sc, rep = drv
        jobs.append({"sweep": sweep, "map": m, "nr": int(nr), "nt": int(nt),
                     "scenario": sc, "rep": int(rep),
                     "cost": group_estimate([src[k] for k in by_drv[drv]]),
                     "planner": P.SWEEP_PLANNER.get(sweep, "lacam")})
    jobs.sort(key=lambda j: -j["cost"])
    n_written_target = sum(1 for k in src if k[:8] in affected_conds)
    print(f"{len(affected_drv)} affected driver groups ({len(affected_conds)} "
          f"condition groups x 8 methods = {n_written_target} rows to write); "
          f"{len(jobs)} to run now; workers={P.N_WORKERS}", flush=True)
    print(f"estimated serial {sum(j['cost'] for j in jobs)/3600:.1f} h; "
          f"longest group {max(j['cost'] for j in jobs):.0f} s", flush=True)

    t0 = time.perf_counter()
    n_cmp = n_mis = n_disc = n_actually_written = 0
    with open(csv_path, mode, newline="", encoding="utf-8") as f, \
         open(prof_path, "a" if mode == "a" else "w", newline="",
              encoding="utf-8") as pf, \
         open(jsonl_path, "a", encoding="utf-8") as jf:
        w = csv.writer(f)
        pw = csv.writer(pf)
        if mode == "w":
            w.writerow(P.ROW_HEADER)
            pw.writerow(P.PROFILES_HEADER)
        with mp.Pool(P.N_WORKERS) as pool:
            for i, (rows, prof_row) in enumerate(
                    pool.imap_unordered(P.run_group, jobs), 1):
                rec = {"group": prof_row[0], "written": 0,
                       "compared": 0, "mismatches": []}
                for r in rows:
                    key = tuple(str(r[col_idx[c]]) for c in KEY_COLS)
                    s_row = src.get(key)
                    if s_row is None:
                        raise SystemExit(f"re-run row has no source twin: {key}")
                    keep = key[:8] in affected_conds
                    if s_row["Run_Timed_Out"] == "0":
                        diffs = compare_row(r, s_row, shared_cols, col_idx)
                        n_cmp += 1
                        rec["compared"] += 1
                        if diffs:
                            n_mis += 1
                            rec["mismatches"].append(
                                {"row": key, "cols": diffs})
                        elif not keep:
                            n_disc += 1
                    if keep:
                        w.writerow(r)
                        n_actually_written += 1
                        rec["written"] += 1
                pw.writerow(prof_row)
                f.flush()
                jf.write(json.dumps(rec) + "\n")
                jf.flush()
                if i % 5 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} groups "
                          f"({(time.perf_counter()-t0)/60:.0f} min, "
                          f"cmp {n_cmp}, mis {n_mis})", flush=True)

    print(f"DONE: wrote {n_actually_written} rows this session "
          f"(target {n_written_target}); compared {n_cmp} non-truncated "
          f"rows vs source ({n_disc} unaffected-condition rows identical "
          f"and discarded); mismatches {n_mis}", flush=True)
    if n_mis:
        print("MISMATCHES PRESENT - do not use the amendment wording "
              "until investigated", flush=True)


def cmd_verify(source_dir, outdir):
    import phase4_sweep as P
    src_header, src = load_source(source_dir)
    affected_conds = {k[:8] for k, r in src.items() if r["Run_Timed_Out"] == "1"}
    with open(results_csv(outdir), newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        new_header = rdr.fieldnames
        new_rows = {tuple(r[c] for c in KEY_COLS): r for r in rdr}
    shared_cols = [c for c in src_header
                   if c in new_header and c not in TIMING_COLS]
    expected = {k for k in src if k[:8] in affected_conds}
    missing = expected - set(new_rows)
    extra = set(new_rows) - expected
    stop_hits = [k for k, r in new_rows.items() if r["Run_Timed_Out"] != "0"]
    mism = []
    n_cmp = 0
    for k, r in new_rows.items():
        s_row = src.get(k)
        if s_row is None:
            mism.append((k, ["NO SOURCE TWIN"]))
            continue
        if s_row["Run_Timed_Out"] == "0":
            n_cmp += 1
            diffs = [c for c in shared_cols if r[c] != s_row[c]]
            if diffs:
                mism.append((k, diffs))
    max_exec = max(float(r["Execution Time"]) for r in new_rows.values()) / 1e9
    print(f"verify {outdir} vs {source_dir}:")
    print(f"  rows written: {len(new_rows)} (expected {len(expected)})")
    print(f"  missing: {len(missing)}  extra: {len(extra)}")
    print(f"  rows hitting the raised stop (Run_Timed_Out=1): {len(stop_hits)}")
    print(f"  non-truncated identity: {n_cmp - len(mism)} identical, "
          f"{len(mism)} mismatching (shared non-timing cols = {len(shared_cols)})")
    print(f"  max Execution Time in new rows: {max_exec:.0f} s")
    if len(new_rows) == len(expected) and not missing and not extra \
            and not stop_hits and not mism:
        print("  OVERALL: PASS")
    else:
        print("  OVERALL: FAIL")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "run" and len(sys.argv) == 6:
        cmd_run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) >= 2 and sys.argv[1] == "verify" and len(sys.argv) == 4:
        cmd_verify(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()