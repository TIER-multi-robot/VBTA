"""SECONDARY quality/fairness re-test on the v2 data (mapping approved).

The plan: "Quality and fairness claims (completion, Jain's index,
Workload_Variance) are re-tested on the v2 data as secondary results."

Per sweep, paired Wilcoxon (Murty's minus each of the 6 voting rules)
on:
  Task_Completion_Fraction, Avg Reassignment Jains Index,
  Workload_Variance
Holm across the 6 comparisons within each (sweep, metric, reporting
level); per scenario and pooled; standard verdicts; positive median
diff = Murty's higher (Jain's/fairness: higher = more fair).

Sweeps and restrictions (approved):
  noise      phase4_final_noise/phase4_expanded.csv, sigma_obs = 0 only
  po         phase4_final_po_merged, no-mask condition only
  balanced   phase4_final_balanced_merged
  constrained phase4_final_constrained
  hyper      phase4_final_hyper
  full       phase4_final_full_r2

Usage: python analysis/secondary_quality_fairness.py
"""

import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from paired_stats import (group_key, index_rows, load, paired_diffs,
                          report_family, verdict, wilcoxon_result)

SWEEPS = [
    ("noise (sigma_obs=0)",
     "hvbta/io/results/phase4_final_noise/phase4_expanded.csv",
     lambda r: r["Obs_Noise_Sigma"] in ("0.0", "0")),
    ("po (no-mask)",
     "hvbta/io/results/phase4_final_po_merged/phase4_simulation.csv",
     lambda r: r["Mask_Level"] == "none"),
    ("balanced",
     "hvbta/io/results/phase4_final_balanced_merged/phase4_simulation.csv",
     None),
    ("constrained",
     "hvbta/io/results/phase4_final_constrained/phase4_simulation.csv",
     None),
    ("hyper",
     "hvbta/io/results/phase4_final_hyper/phase4_simulation.csv",
     None),
    ("full R2",
     "hvbta/io/results/phase4_final_full_r2/phase4_simulation.csv",
     None),
]
GCOLS = ("Map", "Num Robots", "Num Tasks", "Scenario", "Rep")
SCENARIOS = ("calm", "task_churn", "fleet_churn", "overload")
MURTY = "murtys_task_allocation"
VOTING = ("rank_assignments_borda", "rank_assignments_approval",
          "rank_assignments_majority_judgment",
          "rank_assignments_cumulative_voting",
          "rank_assignments_condorcet_method", "rank_assignments_range")
LABEL = {"rank_assignments_borda": "Borda",
         "rank_assignments_approval": "Approval",
         "rank_assignments_majority_judgment": "Majority Judgment",
         "rank_assignments_cumulative_voting": "Cumulative",
         "rank_assignments_condorcet_method": "Condorcet",
         "rank_assignments_range": "Range"}
METRICS = ("Task_Completion_Fraction", "Avg Reassignment Jains Index",
           "Workload_Variance")


def run_family(index, groups, metric):
    results = []
    for rule in VOTING:
        d = paired_diffs(index, groups, metric, MURTY, rule)
        r = wilcoxon_result(d, f"secondary:{metric}:{rule}")
        r["label"] = f"Murty's vs {LABEL[rule]}"
        results.append(r)
    return results


def main():
    print("SECONDARY quality/fairness re-test (mapping approved) - "
          "labeled secondary, not part of the preregistered primary "
          "hypotheses")
    for sweep_name, path, keep in SWEEPS:
        rows = load(path)
        if keep is not None:
            rows = [r for r in rows if keep(r)]
        index = index_rows(rows, GCOLS)
        groups = sorted({group_key(r, GCOLS) for r in rows})
        print()
        print("#" * 72)
        print(f"# sweep: {sweep_name} ({len(rows)} rows, "
              f"{len(groups)} groups)")
        print("#" * 72)
        for metric in METRICS:
            fam = run_family(index, groups, metric)
            report_family(metric, f"Pooled (n = {len(groups)})", fam)
            for sc in SCENARIOS:
                sub = [g for g in groups if g[3] == sc]
                if not sub:
                    continue
                report_family(metric, f"Scenario: {sc} (n = {len(sub)})",
                              run_family(index, sub, metric))


if __name__ == "__main__":
    main()