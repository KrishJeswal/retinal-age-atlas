import numpy as np
from scipy import stats

from rra.analysis.bootstrap import cluster_bootstrap, percentile_ci


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC, P(score_pos > score_neg) + 0.5 P(tie); NaN if a class is empty."""
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_with_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    replicates = cluster_bootstrap(lambda i: auc(scores[i], labels[i]), clusters, n_boot, seed)
    low, high = percentile_ci(replicates, level)
    return {
        "auc": auc(scores, labels),
        "ci_low": low,
        "ci_high": high,
        "n_pos": int(labels.sum()),
        "n_neg": int((~labels).sum()),
    }
