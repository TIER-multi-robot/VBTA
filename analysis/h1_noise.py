"""Preregistered H1 analysis: noise robustness on the phase4 noise sweep.

Implements docs/ANALYSIS_PLAN.md H1 exactly, on the phase4_final_noise
expanded CSV: per-group Task_Completion_Fraction degradation from
sigma_obs = 0 to sigma_obs = 0.3; Murty's K-best vs each ordinal/median
rule (Borda, Majority Judgment, Condorcet); two-sided Wilcoxon
signed-rank on the per-group paired differences (zero pairs dropped by
the default 'wilcox' handling; the dropped count is reported); median
paired difference with a percentile bootstrap 95% CI (10,000 resamples,
zlib.adler32-seeded per level/comparison); Holm-Bonferroni across the
three comparisons within each reporting level (pooled and each
scenario); alpha = 0.05. Also prints the descriptive degradation curve
(mean completion at each sigma level, all eight methods). No other
subgroup analyses.
"""

import csv
import sys
import zlib
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

DEFAULT_CSV = "hvbta/io/results/phase4_final_noise/phase4_expanded.csv"
SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.3)
SIGMA_LO = 0.0
SIGMA_HI = 0.3
ALPHA = 0.05
BOOT_RESAMPLES = 10_000
SCENARIOS = ("calm", "task_churn", "fleet_churn", "overload")

MURTY = "murtys_task_allocation"
RULES = (
    ("Borda", "rank_assignments_borda"),
    ("Majority Judgment", "rank_assignments_majority_judgment"),
    ("Condorcet", "rank_assignments_condorcet_method"),
)
DISPLAY_ORDER = (
    ("Murty's K-best", MURTY),
    ("Perturbed JV", "perturbed_jv_task_allocation"),
    ("Borda", "rank_assignments_borda"),
    ("Approval", "rank_assignments_approval"),
    ("Majority Judgment", "rank_assignments_majority_judgment"),
    ("Cumulative", "rank_assignments_cumulative_voting"),
    ("Condorcet", "rank_assignments_condorcet_method"),
    ("Range", "rank_assignments_range"),
)
ALL_METHODS = tuple(m for _, m in DISPLAY_ORDER)


def canon_sigma(s):
    for t in SIGMAS:
        if abs(s - t) < 1e-9:
            return t
    raise SystemExit(f"unexpected Obs_Noise_Sigma value: {s!r}")


def load(path):
    comp = {}
    sigmas_by_gm = defaultdict(set)
    methods_by_gs = defaultdict(set)
    scenarios = set()
    timed_out = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            g = (row["Map"], row["Num Robots"], row["Num Tasks"],
                 row["Scenario"], row["Rep"])
            sigma = canon_sigma(float(row["Obs_Noise_Sigma"]))
            method = row["Method"]
            key = g + (sigma, method)
            if key in comp:
                raise SystemExit(f"duplicate row key: {key}")
            if row["Run_Timed_Out"] != "0":
                timed_out += 1
            comp[key] = float(row["Task_Completion_Fraction"])
            sigmas_by_gm[g + (method,)].add(sigma)
            methods_by_gs[g + (sigma,)].add(method)
            scenarios.add(row["Scenario"])
    if timed_out:
        raise SystemExit(f"{timed_out} rows have Run_Timed_Out=1; aborting")
    groups = sorted({k[:5] for k in comp})
    if scenarios != set(SCENARIOS):
        raise SystemExit(f"unexpected scenario set: {sorted(scenarios)}")
    for g in groups:
        for method in ALL_METHODS:
            if sigmas_by_gm[g + (method,)] != set(SIGMAS):
                raise SystemExit(f"group {g} method {method}: sigma set incomplete")
        for sigma in SIGMAS:
            if methods_by_gs[g + (sigma,)] != set(ALL_METHODS):
                raise SystemExit(f"group {g} sigma {sigma}: method set incomplete")
    n_rows = len(groups) * len(SIGMAS) * len(ALL_METHODS)
    if len(comp) != n_rows:
        raise SystemExit(f"row count {len(comp)} != expected {n_rows}")
    return comp, groups


def degradation(comp, g, method):
    return comp[g + (SIGMA_LO, method)] - comp[g + (SIGMA_HI, method)]


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, max(prev, (m - rank) * ps[i]))
        adj[i] = a
        prev = a
    return adj


def verdict(median, adj_p):
    if adj_p is None or adj_p >= ALPHA:
        return "not supported"
    if median > 0:
        return "SUPPORTED"
    if median < 0:
        return "not supported (reverse)"
    return "not supported (median=0)"


def analyze_level(name, groups, comp):
    rows = []
    for label, method in RULES:
        d = np.array(
            [degradation(comp, g, MURTY) - degradation(comp, g, method)
             for g in groups],
            dtype=float,
        )
        n = len(d)
        n_zero = int(np.count_nonzero(d == 0.0))
        median = float(np.median(d))
        seed = zlib.adler32(f"h1_noise:{name}:{label}".encode())
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(BOOT_RESAMPLES, n))
        boot = np.median(d[idx], axis=1)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        w_stat = p_raw = None
        try:
            res = wilcoxon(d, alternative="two-sided")
            w_stat, p_raw = float(res.statistic), float(res.pvalue)
        except ValueError:
            pass
        rows.append({
            "label": label, "n": n, "n_zero": n_zero, "median": median,
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "w": w_stat, "p": p_raw,
        })
    valid = [i for i, r in enumerate(rows) if r["p"] is not None]
    adj = holm([rows[i]["p"] for i in valid]) if valid else []
    adj_map = {i: a for i, a in zip(valid, adj)}
    print(f"=== {name} (n groups = {len(groups)}) ===")
    header = (f"{'comparison':<28}{'n':>5}{'zeros':>7}{'median diff':>13}"
              f"{'bootstrap 95% CI':>24}{'W':>12}{'raw p':>12}{'Holm p':>12}  verdict")
    print(header)
    for i, r in enumerate(rows):
        if r["p"] is not None:
            p_str = f"{r['p']:.4g}"
            adj_p = adj_map.get(i)
            adj_str = f"{adj_p:.4g}" if adj_p is not None else "n/a"
        else:
            p_str = adj_str = "n/a"
            adj_p = None
        w_str = f"{r['w']:.1f}" if r["w"] is not None else "n/a"
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        print(f"Murty's vs {r['label']:<18}{r['n']:>5}{r['n_zero']:>7}"
              f"{r['median']:+13.4f}{ci:>24}{w_str:>12}{p_str:>12}{adj_str:>12}"
              f"  {verdict(r['median'], adj_p)}")
    print()


def curve(comp, groups):
    print("=== Descriptive: mean Task_Completion_Fraction by sigma_obs "
          f"(pooled, all {len(groups)} groups) ===")
    cols = "".join(f"{'sigma=' + format(s, '.2f'):>10}" for s in SIGMAS)
    print(f"{'method':<20}{cols}{'drop(0->0.3)':>14}")
    for label, method in DISPLAY_ORDER:
        means = [np.mean([comp[g + (s, method)] for g in groups]) for s in SIGMAS]
        cells = "".join(f"{m:>10.4f}" for m in means)
        print(f"{label:<20}{cells}{means[0] - means[-1]:>+14.4f}")
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    comp, groups = load(path)
    print("H1 (preregistered) -- noise robustness")
    print(f"data: {path} ({len(comp)} rows = {len(groups)} groups x "
          f"{len(SIGMAS)} sigmas x {len(ALL_METHODS)} methods)")
    print(f"degradation(group, method) = completion(sigma={SIGMA_LO}) "
          f"- completion(sigma={SIGMA_HI})")
    print("paired difference = degradation(Murty's) - degradation(rule); "
          "positive = Murty's degrades more (H1 direction)")
    print("test: two-sided Wilcoxon signed-rank, zero_method wilcox "
          "(zero pairs dropped, count reported)")
    print(f"effect: median paired difference, percentile bootstrap 95% CI, "
          f"{BOOT_RESAMPLES} resamples, zlib.adler32-seeded")
    print("multiplicity: Holm-Bonferroni across the 3 comparisons within "
          f"each reporting level; alpha = {ALPHA}")
    print("supported = Holm p < 0.05 AND median difference > 0; "
          "significant reverse direction reported as 'not supported (reverse)'")
    print()
    analyze_level("Pooled", groups, comp)
    for sc in SCENARIOS:
        analyze_level(f"Scenario: {sc}", [g for g in groups if g[3] == sc], comp)
    curve(comp, groups)


if __name__ == "__main__":
    main()