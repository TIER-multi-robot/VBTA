"""Shared paired-comparison machinery for the preregistered analyses
(H2 / H3 / H-PO), matching the analysis/h1_noise.py conventions exactly:

  - two-sided Wilcoxon signed-rank on per-group paired differences
    (zero pairs dropped by the default 'wilcox' handling; dropped count
    reported);
  - median paired difference with a percentile bootstrap 95% CI
    (10,000 resamples, zlib.adler32-seeded per level/comparison);
  - Holm-Bonferroni within each family; alpha = 0.05;
  - verdicts: supported / not supported / not supported (reverse).
"""

import csv
import zlib

import numpy as np
from scipy.stats import wilcoxon

ALPHA = 0.05
BOOT_RESAMPLES = 10_000


def load(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def group_key(r, cols):
    return tuple(r[c] for c in cols)


def index_rows(rows, gcols):
    """(group_key, Method) -> row."""
    return {(group_key(r, gcols), r["Method"]): r for r in rows}


def paired_diffs(index, groups, metric, method_a, method_b):
    """Per-group paired differences value(a) - value(b), groups where both
    methods exist."""
    d = []
    for g in groups:
        ra = index.get((g, method_a))
        rb = index.get((g, method_b))
        if ra is None or rb is None:
            continue
        d.append(float(ra[metric]) - float(rb[metric]))
    return np.array(d, dtype=float)


def wilcoxon_result(d, seed_label):
    n = len(d)
    n_zero = int(np.count_nonzero(d == 0.0))
    median = float(np.median(d))
    rng = np.random.default_rng(zlib.adler32(f"paired:{seed_label}".encode()))
    if n:
        idx = rng.integers(0, n, size=(BOOT_RESAMPLES, n))
        boot = np.median(d[idx], axis=1)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    else:
        ci_lo = ci_hi = float("nan")
    w_stat = p_raw = None
    if n:
        try:
            res = wilcoxon(d, alternative="two-sided")
            w_stat, p_raw = float(res.statistic), float(res.pvalue)
        except ValueError:
            pass
    return {"n": n, "n_zero": n_zero, "median": median,
            "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "w": w_stat, "p": p_raw}


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


def report_family(title, level_name, results, verdict_fn=verdict):
    """results: list of dicts with keys label, n, n_zero, median, ci_lo,
    ci_hi, w, p (from wilcoxon_result) - Holm applied across all of them
    (the family), one table printed. verdict_fn maps (median, adj_p) to the
    verdict string (the standard rule by default)."""
    valid = [i for i, r in enumerate(results) if r["p"] is not None]
    adj = holm([results[i]["p"] for i in valid]) if valid else []
    adj_map = {i: a for i, a in zip(valid, adj)}
    print(f"=== {title} | {level_name} ===")
    print(f"{'comparison':<34}{'n':>5}{'zeros':>7}{'median diff':>13}"
          f"{'bootstrap 95% CI':>24}{'W':>12}{'raw p':>12}{'Holm p':>12}"
          f"  verdict")
    for i, r in enumerate(results):
        if r["p"] is not None:
            p_str = f"{r['p']:.4g}"
            adj_p = adj_map.get(i)
            adj_str = f"{adj_p:.4g}" if adj_p is not None else "n/a"
        else:
            p_str = adj_str = "n/a"
            adj_p = None
        w_str = f"{r['w']:.1f}" if r["w"] is not None else "n/a"
        ci = (f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
              if r["n"] else "n/a")
        print(f"{r['label']:<34}{r['n']:>5}{r['n_zero']:>7}"
              f"{r['median']:+13.4f}{ci:>24}{w_str:>12}{p_str:>12}"
              f"{adj_str:>12}  {verdict_fn(r['median'], adj_p)}")
    print()