"""Item 2 verification: pool vs serial bit-identity, noise + PO paths,
all 8 methods, one small group each (lak102d 10x10, 4 scenarios, rep 0).

Runs phase4_sweep.run_group in-process for sweep=noise and sweep=po and
compares every output-tuple field against the serial __main__ CSVs,
excluding wall-clock timing columns (output idx 1 total_reassignment_time,
55 Planner_Total_ms, 57 exec time). Planner_Call_Count (54) and
Planner_Cap_Hits (56) are deterministic and ARE compared.
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import phase4_sweep as P

SCENARIOS = ["calm", "task_churn", "fleet_churn", "overload"]
TIMING_IDX = {1, 55, 57}   # 0-based within the 58-item output tuple
N_OUT = 58                 # 57 main + exec


def load_serial(dir_path, mode):
    """Returns {(method, scenario, cond): [58 output fields]}."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(dir_path, "*simulation*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                fields = list(r.values())
                out = fields[14:14 + N_OUT]
                if mode == "noise":
                    cond = float(r["Obs_Noise_Sigma"])
                else:
                    cond = (r["Mask_Level"], r["Mask_Policy"])
                rows[(r["Method"], r["Scenario"], cond)] = out
    return rows


def main():
    ok = True
    for mode, sweep, serial_dir in [
            ("noise", "noise", "verify_serial_noise_v3"),
            ("po", "po", "verify_serial_po_v3")]:
        serial = load_serial(os.path.join("hvbta", "io", "results", serial_dir), mode)
        print(f"\n=== {mode.upper()} path: {len(serial)} serial rows loaded ===")
        checked = mismatches = 0
        for scenario in SCENARIOS:
            job = {"sweep": sweep, "map": "lak102d.map", "nr": 10, "nt": 10,
                   "scenario": scenario, "rep": 0, "cost": 0, "planner": "lacam"}
            for row in P.run_group(job):
                method, out = row[5], row[6]
                out = [x.item() if hasattr(x, "item") else x for x in out]
                if mode == "noise":
                    cond = float(row[11])
                else:
                    cond = (row[9] or "none", row[10] or "none")
                key = (method, scenario, cond)
                if key not in serial:
                    print(f"  MISSING serial row for {key}")
                    ok = False
                    continue
                s = serial[key]
                checked += 1
                diffs = []
                for i in range(N_OUT):
                    if i in TIMING_IDX:
                        continue
                    a, b = out[i], s[i]
                    try:
                        if abs(float(a) - float(b)) > 1e-9:
                            diffs.append((i, a, b))
                    except (ValueError, TypeError):
                        if str(a) != str(b):
                            diffs.append((i, a, b))
                if diffs:
                    mismatches += 1
                    print(f"  MISMATCH {key}: {len(diffs)} fields, "
                          f"first 5: {[(i, a, b) for i, a, b in diffs[:5]]}")
        print(f"  checked {checked} rows (8 methods x 5 conditions x 4 scenarios"
              f" = 160 expected); mismatches: {mismatches}")
        ok = ok and checked == 160 and mismatches == 0
    print("\nRESULT:", "POOL == SERIAL on all rows (excl. timing)"
          if ok else "MISMATCHES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
