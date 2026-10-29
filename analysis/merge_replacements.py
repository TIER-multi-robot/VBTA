"""Merge original sweeps with valid uncapped replacement rows (2026-10-28).

For each sweep (po, balanced): start from the ORIGINAL results directory's
phase4_simulation.csv; replace every Run_Timed_Out = 1 row with its valid
replacement - a row with Planner_Timeout_Count = 0 for the same
(map, nr, nt, scenario, rep, condition, method) key, taken from
phase4_final_po_uncapped (v1, PO only) where v1's copy is valid, otherwise
from rerun_targeted/ - and write the merged file to a NEW directory
(phase4_final_po_merged / phase4_final_balanced_merged). A Source_Dir column
records each row's provenance. Existing results directories are never
modified.

The 2026-10-28 validity rule: a row is valid iff Planner_Timeout_Count = 0
(the LaCAM 1 s wall-clock budget is the only load-dependent code path; rows
with zero budget hits reproduce exactly). PO source rows predate the
Planner_Timeout_Count column; they get "0", justified by Plan_Failure_Count
= 0 on every source row (verified at merge time).

Assertions (each reported):
  1. every group is complete with all 8 methods (group = map, nr, nt,
     scenario, rep, condition; the set must equal the source's);
  2. 0 rows with Run_Timed_Out = 1 in the merged file;
  3. 0 rows with Planner_Timeout_Count > 0 in the merged file.

Usage:
  python analysis/merge_replacements.py [po|balanced|all]   (default: all)
"""

import csv
import os
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
os.chdir(REPO)

KEY_COLS = ("Map", "Num Robots", "Num Tasks", "Scenario", "Rep",
            "Mask_Level", "Mask_Policy", "Obs_Noise_Sigma", "Method")
N_METHODS = 8


def results_csv(dir_name, fname="phase4_simulation.csv"):
    return os.path.join("hvbta", "io", "results", dir_name, fname)


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return rdr.fieldnames, list(rdr)


def row_key(r):
    return tuple(r[c] for c in KEY_COLS)


def valid_by_key(rows):
    """key -> row, only rows with PTC = 0 and Run_Timed_Out = 0; asserts all
    valid rows for a key are identical (determinism)."""
    out = {}
    for r in rows:
        if r["Planner_Timeout_Count"] != "0" or r["Run_Timed_Out"] != "0":
            continue
        k = row_key(r)
        if k in out and out[k] != r:
            raise SystemExit(f"two non-identical valid rows for {k}")
        out[k] = r
    return out


def merge(sweep):
    src_dir = f"phase4_final_{sweep}"
    out_dir = f"phase4_final_{sweep}_merged"
    src_fn, src_rows = load(results_csv(src_dir))

    import phase4_sweep as P
    out_fn = P.ROW_HEADER + ["Source_Dir"]
    ptc_in_src = "Planner_Timeout_Count" in src_fn

    targ_fn, targ_rows = load(results_csv("rerun_targeted"))
    targ = valid_by_key(targ_rows)
    v1 = {}
    if sweep == "po":
        v1_fn, v1_rows = load(results_csv("phase4_final_po_uncapped"))
        v1 = valid_by_key(v1_rows)

    n_repl_v1 = n_repl_t = 0
    out_rows = []
    for r in src_rows:
        k = row_key(r)
        if r["Run_Timed_Out"] != "1":
            row = dict(r)
            if not ptc_in_src:
                if r["Plan_Failure_Count"] != "0":
                    raise SystemExit(f"original PO row has failures: {k}")
                row["Planner_Timeout_Count"] = "0"
            else:
                row["Planner_Timeout_Count"] = r["Planner_Timeout_Count"]
            row["Source_Dir"] = src_dir
            out_rows.append(row)
            continue
        if sweep == "po" and k in v1:
            rep, src_of_rep = v1[k], "phase4_final_po_uncapped"
            n_repl_v1 += 1
        elif k in targ:
            rep, src_of_rep = targ[k], "rerun_targeted"
            n_repl_t += 1
        else:
            raise SystemExit(f"no valid replacement for truncated row {k}")
        row = {c: rep[c] for c in P.ROW_HEADER}
        row["Source_Dir"] = src_of_rep
        out_rows.append(row)

    bad = [row_key(r) for r in out_rows if r["Planner_Timeout_Count"] != "0"]
    assert not bad, f"merged rows with PTC>0: {bad[:5]}"
    timed_out = [row_key(r) for r in out_rows if r["Run_Timed_Out"] == "1"]
    assert not timed_out, f"merged rows with Run_Timed_Out=1: {timed_out[:5]}"

    src_groups = {}
    for r in src_rows:
        src_groups.setdefault(row_key(r)[:8], []).append(r["Method"])
    out_groups = {}
    for r in out_rows:
        out_groups.setdefault(row_key(r)[:8], []).append(r["Method"])
    incomplete = []
    for g, methods in src_groups.items():
        om = out_groups.get(g, [])
        if sorted(om) != sorted(methods) or len(om) != N_METHODS:
            incomplete.append(g)
    assert not incomplete, f"incomplete groups: {incomplete[:5]}"
    assert set(out_groups) == set(src_groups), "group sets differ"

    os.makedirs(os.path.join("hvbta", "io", "results", out_dir), exist_ok=True)
    with open(results_csv(out_dir), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fn)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    shutil.copyfile(results_csv(src_dir, "phase4_profiles.csv"),
                   results_csv(out_dir, "phase4_profiles.csv"))

    n_trunc = sum(1 for r in src_rows if r["Run_Timed_Out"] == "1")
    print(f"[{sweep}] rows: {len(out_rows)} (source {len(src_rows)}; "
          f"{n_trunc} truncated replaced: {n_repl_v1} from "
          f"phase4_final_po_uncapped, {n_repl_t} from rerun_targeted)")
    print(f"[{sweep}] Source_Dir counts:",
          {d: sum(1 for r in out_rows if r["Source_Dir"] == d)
           for d in sorted({r["Source_Dir"] for r in out_rows})})
    print(f"[{sweep}] checks: groups complete with 8 methods "
          f"({len(out_groups)} groups): PASS; Run_Timed_Out=1 rows: "
          f"{len(timed_out)}: PASS; Planner_Timeout_Count>0 rows: "
          f"{len(bad)}: PASS")
    return len(out_rows), len(timed_out), len(bad), len(out_groups)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    sweeps = ["po", "balanced"] if arg == "all" else [arg]
    for s in sweeps:
        merge(s)


if __name__ == "__main__":
    main()