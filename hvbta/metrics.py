import numpy as np
from typing import List
from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.optimize import linear_sum_assignment


def calculate_jains_index(scores: List[float]) -> float:
    """
    Calculate Jain's Fairness Index for a list of allocation scores.
    
    Jain's Index formula: J = (sum(x_i))^2 / (n * sum(x_i^2))
    
    Properties:
    - Returns 1.0 for perfectly fair allocations (all scores equal)
    - Returns 1/n for maximally unfair allocations (one agent gets everything)
    - Range: [1/n, 1]
    
    Parameters:
        scores: List of individual agent allocation scores (suitability scores).
                Only includes assigned agents (unassigned agents are excluded).
    
    Returns:
        Jain's fairness index value in range [0, 1], or 1.0 if empty/all zeros.
    """
    if not scores or len(scores) == 0:
        return 1.0  # No allocations = trivially fair
    
    n = len(scores)
    sum_scores = sum(scores)
    sum_sq_scores = sum(s ** 2 for s in scores)
    
    if sum_sq_scores == 0:
        return 1.0  # All zeros = perfectly "fair" (everyone got equally nothing)
    
    return (sum_scores ** 2) / (n * sum_sq_scores)


def calculate_threshold_metrics(scores: List[float]) -> dict:
    """
    Calculate threshold fairness metrics for Approval voting analysis.
    
    Uses two thresholds:
    - GOOD_ENOUGH (0.5): Minimum acceptable suitability
    - GOOD (0.7): Desirable suitability level
    
    Parameters:
        scores: List of individual agent allocation scores.
    
    Returns:
        Dictionary with:
        - below_ge_frac: Fraction of agents below "good enough" threshold (0.5)
        - below_good_frac: Fraction of agents below "good" threshold (0.7)
        - deficit_all_ge: Mean deficit from 0.5 threshold, averaged over ALL agents
        - deficit_below_ge: Mean deficit from 0.5 threshold, averaged over those BELOW only
        - deficit_all_good: Mean deficit from 0.7 threshold, averaged over ALL agents  
        - deficit_below_good: Mean deficit from 0.7 threshold, averaged over those BELOW only
    """
    THRESHOLD_GOOD_ENOUGH = 0.5
    THRESHOLD_GOOD = 0.7
    
    if not scores or len(scores) == 0:
        return {
            "below_ge_frac": 0.0,
            "below_good_frac": 0.0,
            "deficit_all_ge": 0.0,
            "deficit_below_ge": 0.0,
            "deficit_all_good": 0.0,
            "deficit_below_good": 0.0
        }
    
    n = len(scores)
    
    # Good Enough threshold (0.5)
    below_ge = [s for s in scores if s < THRESHOLD_GOOD_ENOUGH]
    below_ge_count = len(below_ge)
    below_ge_frac = below_ge_count / n
    deficit_all_ge = sum(max(0, THRESHOLD_GOOD_ENOUGH - s) for s in scores) / n
    deficit_below_ge = sum(max(0, THRESHOLD_GOOD_ENOUGH - s) for s in below_ge) / below_ge_count if below_ge_count > 0 else 0.0
    
    # Good threshold (0.7)
    below_good = [s for s in scores if s < THRESHOLD_GOOD]
    below_good_count = len(below_good)
    below_good_frac = below_good_count / n
    deficit_all_good = sum(max(0, THRESHOLD_GOOD - s) for s in scores) / n
    deficit_below_good = sum(max(0, THRESHOLD_GOOD - s) for s in below_good) / below_good_count if below_good_count > 0 else 0.0
    
    return {
        "below_ge_frac": below_ge_frac,
        "below_good_frac": below_good_frac,
        "deficit_all_ge": deficit_all_ge,
        "deficit_below_ge": deficit_below_ge,
        "deficit_all_good": deficit_all_good,
        "deficit_below_good": deficit_below_good
    }

def calculate_inequality_metrics(scores: List[float]) -> dict:
    """
    Calculate min-max and inequality fairness metrics for Majority Judgment analysis.
    
    Uses O(n log n) Gini coefficient calculation via sorted cumulative sum.
    
    Parameters:
        scores: List of individual agent allocation scores.
    
    Returns:
        Dictionary with:
        - score_range: max - min score (spread between best and worst)
        - min_max_ratio: min/max ratio (closer to 1 = fairer)
        - gini: Gini coefficient [0=perfect equality, 1=max inequality]
        - cv: Coefficient of variation (std/mean, normalized dispersion)
    """
    if not scores or len(scores) == 0:
        return {
            "score_range": 0.0,
            "min_max_ratio": 1.0,
            "gini": 0.0,
            "cv": 0.0
        }
    
    n = len(scores)
    min_s = min(scores)
    max_s = max(scores)
    mean_s = sum(scores) / n
    
    # Range
    score_range = max_s - min_s
    
    # Min/Max ratio (1.0 if max is 0 to avoid division by zero)
    min_max_ratio = min_s / max_s if max_s > 0 else 1.0
    
    # Gini coefficient - O(n log n) via sorted cumulative sum
    # Formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    # where x_i are sorted in ascending order and i is 1-indexed
    sorted_scores = sorted(scores)
    cumsum = sum((i + 1) * s for i, s in enumerate(sorted_scores))
    total = sum(sorted_scores)
    if total > 0:
        gini = (2 * cumsum) / (n * total) - (n + 1) / n
        gini = max(0.0, gini)  # Ensure non-negative due to floating point
    else:
        gini = 0.0
    
    # Coefficient of variation
    if mean_s > 0:
        variance = sum((s - mean_s) ** 2 for s in scores) / n
        std_dev = variance ** 0.5
        cv = std_dev / mean_s
    else:
        cv = 0.0
    
    return {
        "score_range": score_range,
        "min_max_ratio": min_max_ratio,
        "gini": gini,
        "cv": cv
    }

def calculate_robustness_metrics(scores: List[float]) -> dict:
    """
    Calculate outlier robustness metrics for comparing median vs mean behavior.
    
    These metrics help demonstrate why Majority Judgment (median-based) is more
    robust than mean-based methods to extreme outlier scores.
    
    Parameters:
        scores: List of individual agent allocation scores.
    
    Returns:
        Dictionary with:
        - median: Median score (resistant to outliers)
        - mean: Mean score (sensitive to outliers)
        - med_mean_gap: |median - mean| (large gap indicates skewed distribution)
        - iqr: Interquartile range (Q3 - Q1, robust spread measure)
    """
    if not scores or len(scores) == 0:
        return {
            "median": 0.0,
            "mean": 0.0,
            "med_mean_gap": 0.0,
            "iqr": 0.0
        }
    
    n = len(scores)
    sorted_scores = sorted(scores)
    
    # Mean
    mean_s = sum(scores) / n
    
    # Median
    if n % 2 == 1:
        median_s = sorted_scores[n // 2]
    else:
        median_s = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    
    # Median-Mean gap
    med_mean_gap = abs(median_s - mean_s)
    
    # Interquartile Range (IQR = Q3 - Q1)
    def percentile(data, p):
        """Calculate percentile using linear interpolation."""
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])
    
    q1 = percentile(sorted_scores, 25)
    q3 = percentile(sorted_scores, 75)
    iqr = q3 - q1
    
    return {
        "median": median_s,
        "mean": mean_s,
        "med_mean_gap": med_mean_gap,
        "iqr": iqr
    }

def compare_matrices(M_llm: np.ndarray, M_rule: np.ndarray) -> dict:
    R, T = M_rule.shape
    flat_llm, flat_rule = M_llm.ravel(), M_rule.ravel()

    # Calibration
    mae = np.mean(np.abs(flat_llm - flat_rule))
    rmse = np.sqrt(np.mean((flat_llm - flat_rule) ** 2))
    bias = np.mean(flat_llm - flat_rule)
    pearson = pearsonr(flat_llm, flat_rule).statistic

    # Ranking
    # take the spearman rank correlation coefficient across each task
    rho_per_task = np.nanmean([spearmanr(M_llm[:, j], M_rule[:, j]).statistic for j in range(T)])
    # take the spearman rank correlation coefficient across each robot
    rho_per_robot = np.nanmean([spearmanr(M_llm[i, :], M_rule[i, :]).statistic for i in range(R)])

    # Decisions
    # check the top one robot ranking is agreed upon per task
    top1 = np.mean(M_llm.argmax(axis=0) == M_rule.argmax(axis=0))
    # Hungarian assignment overlap, Jaccard index
    r_llm, c_llm = linear_sum_assignment(-M_llm)
    r_rule, c_rule = linear_sum_assignment(-M_rule)
    pairs_llm = set(zip(r_llm, c_llm))
    pairs_rule = set(zip(r_rule, c_rule))
    jaccard = len(pairs_llm & pairs_rule) / max(len(pairs_llm | pairs_rule), 1)

    # Efficiency: LLM's chosen assignments evaluated under rule based matrix
    llm_score_under_rule = M_rule[r_llm, c_llm].sum()
    rule_optimum = M_rule[r_rule, c_rule].sum()
    efficiency = llm_score_under_rule / rule_optimum if rule_optimum > 0 else 0.0

    # Feasability agreement
    eps = 0.05
    feas_agree = np.mean((M_llm > eps) == (M_rule > eps))

    return dict(mae=mae, rmse=rmse, bias=bias, pearson=pearson, 
                rho_per_task=rho_per_task, rho_per_robot=rho_per_robot,
                top1=top1, hungarian_jaccard=jaccard, efficiency=efficiency,
                feasability_agree=feas_agree)