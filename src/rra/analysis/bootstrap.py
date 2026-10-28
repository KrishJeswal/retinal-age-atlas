from collections.abc import Callable

import numpy as np


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Replicates of `statistic(index)` where each replicate resamples whole clusters (patients)
    with replacement; `index` selects the rows of the resampled clusters. Eyes of one patient are
    correlated, so resampling eyes would understate the variance."""
    rng = np.random.default_rng(seed)
    _, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    members = np.split(order, np.flatnonzero(np.diff(inverse[order])) + 1)
    n = len(members)
    return np.array(
        [
            statistic(np.concatenate([members[i] for i in rng.integers(0, n, n)]))
            for _ in range(n_boot)
        ]
    )


def percentile_ci(replicates: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval; NaN replicates (e.g. a resample with no positives) are dropped."""
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(replicates, [tail, 100 - tail])
    return float(low), float(high)
