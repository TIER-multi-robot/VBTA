"""Preregistered H-PO analysis (H-PO1, H-PO2, H-PO3) - mapping approved.

Implements docs/ANALYSIS_PLAN.md H-PO1/2/3 exactly, per the approved
data mapping. Data: phase4_final_po_merged (the PO sweep with its 279
wall-stop truncated rows replaced by valid uncapped re-runs; the
merged analysis file per the 2026-10-28 amendment). The pairing unit
and contrasts per hypothesis:

H-PO1 "Voting's advantage over Murty's grows as masking increases."
  Metric: Task_Completion_Fraction (the primary metric). Advantage
  D(cond) = completion(rule) - completion(Murty's) per condition-level
  group. Contrast per policy (optimistic, pessimistic):
  D(high) - D(none) per (map, nr, nt, scenario, rep) driver group;
  positive = the advantage grew from no masking to the high mask.
  Family: the 6 voting rules per policy (Holm); reported per scenario
  and pooled. The low level is reported descriptively (table of mean D
  per condition), not tested - the hypothesis speaks of increasing
  masking, tested at the strongest level.

H-PO2 "Optimistic masking raises mis-assignment rate and wasted travel;
  pessimistic masking raises starvation above the Never_Attempted
  baseline."
  Contrasts (per method, paired within driver group):
    Mis_Assignment_Rate:   low-optimistic vs none, high-optimistic vs none
    Wasted_Travel_Fraction: low-optimistic vs none, high-optimistic vs none
    Masked_Starvation_Count - Never_Attempted_Count (derived per row):
      low-pessimistic vs none, high-pessimistic vs none
  Family: the 8 methods per (metric, contrast) - Holm. Reported pooled
  (the claim is about the masking conditions, not per-scenario; the
  per-scenario breakdown is printed descriptively for the medians).

H-PO3 "Voting and perturbed JV have lower
  Mis_Assignment_Recovery_Latency than Murty's."
  Data restricted to the optimistic-mask groups (mis-assignments occur
  there; the none/pessimistic conditions have Mis_Assignment_Count = 0
  and the latency column is the -1 sentinel). Comparisons: Murty's vs
  each of the 6 voting rules and vs perturbed JV (7), paired per
  condition-level group; pairs where EITHER side is the -1 sentinel
  are dropped (dropped count reported). Family: the 7 comparisons
  (Holm); reported per scenario and pooled. Positive median
  (Murty's minus the other) = the other has lower latency = supported.

Conventions: analysis/paired_stats.py (two-sided Wilcoxon, median +
bootstrap 95% CI, zero-pair counts, Holm, standard verdicts).
"""

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from paired_stats import (group_key, index_rows, load, paired_diffs,
                          report_family, verdict, wilcoxon_result)

DATA = "hvbta/io/results/phase4_final_po_merged/phase4_simulation.csv"
DCOLS = ("Map", "Num Robots", "Num Tasks", "Scenario", "Rep")
CCOLS = DCOLS + ("Mask_Level", "Mask_Policy")
GCOLS = CCOLS + ("Obs_Noise_Sigma",)
SCENARIOS = ("calm", "task_churn", "fleet_churn", "overload")
MURTY = "murtys_task_allocation"
PJV = "perturbed_jv_task_allocation"
VOTING = ("rank_assignments_borda", "rank_assignments_approval",
          "rank_assignments_majority_judgment",
          "rank_assignments_cumulative_voting",
          "rank_assignments_condorcet_method", "rank_assignments_range")
ALL_METHODS = VOTING + (MURTY, PJV)
LABEL = {"rank_assignments_borda": "Borda",
         "rank_assignments_approval": "Approval",
         "rank_assignments_majority_judgment": "Majority Judgment",
         "rank_assignments_cumulative_voting": "Cumulative",
         "rank_assignments_condorcet_method": "Condorcet",
         "rank_assignments_range": "Range",
         "murtys_task_allocation": "Murty's",
         "perturbed_jv_task_allocation": "Perturbed JV"}
CONDITIONS = (("none", "none"), ("low", "optimistic"),
             ("low", "pessimistic"), ("high", "optimistic"),
             ("high", "pessimistic"))


def starv(r):
    return float(r["Masked_Starvation_Count"]) - float(r["Never_Attempted_Count"])


def hpo1(rows, index):
    """D(high, pol) - D(none) per voting rule, per policy."""
    print("=" * 70)
    print("H-PO1: advantage growth with masking (Task_Completion_Fraction)")
    print("=" * 70)
    by_d_m = {}   # (driver, cond, rule) -> completion
    by_d_mu = {}
    for r in rows:
        d = group_key(r, DCOLS)
        c = (r["Mask_Level"], r["Mask_Policy"])
        if r["Method"] == MURTY:
            by_d_mu[(d, c)] = float(r["Task_Completion_Fraction"])
        elif r["Method"] in VOTING:
            by_d_m[(d, c, r["Method"])] = float(r["Task_Completion_Fraction"])
    drivers = sorted({d for d, c, m in by_d_m})
    # descriptive: mean D per condition per rule
    print("descriptive mean advantage D (rule - Murty's) per condition:")
    hdr = f"{'rule':<18}" + "".join(f"{l:>18}" for l in
                                    ("none", "low-opt", "low-pess",
                                     "high-opt", "high-pess"))
    print(hdr)
    for rule in VOTING:
        cells = []
        for c in CONDITIONS:
            ds = [by_d_m[(d, c, rule)] - by_d_mu[(d, c)]
                  for d in drivers
                  if (d, c, rule) in by_d_m and (d, c) in by_d_mu]
            cells.append(np.mean(ds) if ds else float("nan"))
        print(f"{LABEL[rule]:<18}" + "".join(f"{v:>+18.4f}" for v in cells))
    print()
    for pol in ("optimistic", "pessimistic"):
        for name, dgroups in [("Pooled", drivers)] + \
                [(f"Scenario: {sc}", [d for d in drivers if d[3] == sc])
                 for sc in SCENARIOS]:
            results = []
            for rule in VOTING:
                diffs = []
                for d in dgroups:
                    hi = (d, ("high", pol))
                    no = (d, ("none", "none"))
                    if (d, hi, rule) in by_d_m and (d, no, rule) in by_d_m \
                            and hi in by_d_mu and no in by_d_mu:
                        diffs.append(
                            (by_d_m[(d, hi, rule)] - by_d_mu[hi])
                            - (by_d_m[(d, no, rule)] - by_d_mu[no]))
                r = wilcoxon_result(np.array(diffs),
                                    f"hpo1:{pol}:{rule}")
                r["label"] = f"D(high-{pol}) - D(none): {LABEL[rule]}"
                results.append(r)
            report_family(f"H-PO1 {pol}", f"{name} (n = {len(dgroups)})",
                          results)


def hpo2(rows):
    print("=" * 70)
    print("H-PO2: masking harms (mis-assignment, wasted travel, starvation)")
    print("=" * 70)
    by_dm = {}
    for r in rows:
        by_dm[(group_key(r, DCOLS), r["Mask_Level"] + "-" + r["Mask_Policy"],
               r["Method"])] = r
    drivers = sorted({d for d, c, m in by_dm})

    def contrast(methods, value, cond_a, cond_b, family_title):
        results = []
        for m in methods:
            diffs = []
            for d in drivers:
                ra = by_dm.get((d, cond_a, m))
                rb = by_dm.get((d, cond_b, m))
                if ra is None or rb is None:
                    continue
                diffs.append(value(ra) - value(rb))
            r = wilcoxon_result(np.array(diffs), f"hpo2:{family_title}:{m}")
            r["label"] = f"{LABEL[m]}"
            results.append(r)
        report_family(family_title, f"Pooled (n drivers = {len(drivers)})",
                      results)

    for metric_title, value in (
            ("Mis_Assignment_Rate",
             lambda r: float(r["Mis_Assignment_Rate"])),
            ("Wasted_Travel_Fraction",
             lambda r: float(r["Wasted_Travel_Fraction"]))):
        contrast(ALL_METHODS, value, "low-optimistic", "none-none",
                 f"H-PO2a {metric_title}: low-opt - none")
        contrast(ALL_METHODS, value, "high-optimistic", "none-none",
                 f"H-PO2a {metric_title}: high-opt - none")
    contrast(ALL_METHODS, starv, "low-pessimistic", "none-none",
             "H-PO2b starvation (Masked_Starvation - Never_Attempted): "
             "low-pess - none")
    contrast(ALL_METHODS, starv, "high-pessimistic", "none-none",
             "H-PO2b starvation (Masked_Starvation - Never_Attempted): "
             "high-pess - none")


def hpo3(rows):
    print("=" * 70)
    print("H-PO3: Mis_Assignment_Recovery_Latency vs Murty's "
          "(optimistic-mask groups only; -1 sentinel pairs dropped)")
    print("=" * 70)
    print("SEMANTICS OF -1 (from partial_observability.finalize): "
          "recovery_latency_mean = mean(recovery_latencies) if any were "
          "recorded, else -1. recovery_latencies only records tasks that "
          "were RE-ASSIGNED after a mis-assignment. So -1 = no recovery "
          "events recorded, which is EITHER no mis-assignment at all "
          "(Mis_Assignment_Count == 0) OR mis-assignments that were never "
          "recovered before the run ended (Mis_Assignment_Count > 0). "
          "Disambiguated per row via Mis_Assignment_Count; pairs with a "
          "-1 on either side are dropped for the test either way, with "
          "the never-recovered-any count reported per method and flagged.")
    print()
    opt = [r for r in rows
           if (r["Mask_Level"], r["Mask_Policy"]) in
           (("low", "optimistic"), ("high", "optimistic"))]
    index = index_rows(opt, GCOLS)
    groups = sorted({group_key(r, GCOLS) for r in opt})
    others = VOTING + (PJV,)
    never_recovered = {m: 0 for m in others + (MURTY,)}
    no_mis = {m: 0 for m in others + (MURTY,)}
    for r in opt:
        m = r["Method"]
        if m in never_recovered and float(r["Mis_Assignment_Recovery_Latency"]) == -1:
            if float(r["Mis_Assignment_Count"]) > 0:
                never_recovered[m] += 1
            else:
                no_mis[m] += 1

    def run(groups_subset):
        results = []
        for m in others:
            d, dropped = [], 0
            for g in groups_subset:
                ra = index.get((g, MURTY))
                rb = index.get((g, m))
                if ra is None or rb is None:
                    continue
                va, vb = float(ra["Mis_Assignment_Recovery_Latency"]), \
                    float(rb["Mis_Assignment_Recovery_Latency"])
                if va == -1 or vb == -1:
                    dropped += 1
                    continue
                d.append(va - vb)
            r = wilcoxon_result(np.array(d), f"hpo3:{m}")
            r["label"] = (f"Murty's vs {LABEL[m]} "
                          f"(dropped {dropped} sentinel pairs)")
            results.append(r)
        return results

    print("sentinel accounting over all optimistic-mask runs per method:")
    for m in others + (MURTY,):
        print(f"  {LABEL[m]:<18} runs with latency -1 and "
              f"mis-assignments (never recovered): {never_recovered[m]:>4}"
              f"   runs with latency -1 and no mis-assignment: "
              f"{no_mis[m]:>4}"
              + ("   *** FLAG" if never_recovered[m] else ""))
    print()
    report_family("H-PO3", f"Pooled (n groups = {len(groups)})",
                  run(groups))
    for sc in SCENARIOS:
        sub = [g for g in groups if g[3] == sc]
        report_family("H-PO3", f"Scenario: {sc} (n = {len(sub)})",
                      run(sub))
    print()


def main():
    rows = load(DATA)
    print("H-PO (preregistered) -- H-PO1, H-PO2, H-PO3 "
          "[mapping approved]")
    print(f"data: {DATA} ({len(rows)} rows)")
    print()
    index = index_rows(rows, GCOLS)
    hpo1(rows, index)
    hpo2(rows)
    hpo3(rows)


if __name__ == "__main__":
    main()