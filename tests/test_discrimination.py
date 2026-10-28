import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from rra.analysis.bootstrap import cluster_bootstrap, percentile_ci
from rra.analysis.discrimination import auc, auc_with_ci


@given(
    st.lists(st.tuples(st.integers(0, 5), st.booleans()), min_size=2, max_size=40).filter(
        lambda rows: 0 < sum(label for _, label in rows) < len(rows)
    )
)
def test_auc_equals_pairwise_probability(rows):
    scores = np.array([s for s, _ in rows], dtype=float)
    labels = np.array([lab for _, lab in rows])
    pos, neg = scores[labels], scores[~labels]
    brute = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (
        len(pos) * len(neg)
    )
    assert auc(scores, labels) == pytest.approx(brute)


def test_auc_empty_class_is_nan():
    assert np.isnan(auc(np.array([1.0, 2.0]), np.array([True, True])))


@pytest.mark.parametrize("d", [0.0, 0.5, 1.0, 2.0])
def test_auc_recovers_planted_binormal_separation(d):
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(d, 1, 20000), rng.normal(0, 1, 20000)])
    labels = np.repeat([True, False], 20000)
    assert auc(scores, labels) == pytest.approx(stats.norm.cdf(d / np.sqrt(2)), abs=0.01)


def test_cluster_bootstrap_resamples_whole_clusters():
    clusters = np.array([7, 7, 3, 3, 3, 9])
    together = cluster_bootstrap(
        lambda i: float(
            np.count_nonzero(i == 0) == np.count_nonzero(i == 1)
            and np.count_nonzero(i == 2) == np.count_nonzero(i == 4)
        ),
        clusters,
        n_boot=200,
    )
    assert together.min() == 1.0


def test_cluster_bootstrap_is_seeded():
    clusters = np.arange(30) // 2
    first = cluster_bootstrap(lambda i: i.sum(), clusters, n_boot=50, seed=4)
    again = cluster_bootstrap(lambda i: i.sum(), clusters, n_boot=50, seed=4)
    assert np.array_equal(first, again)


def test_percentile_ci_ignores_nan():
    assert percentile_ci(np.array([np.nan, *np.arange(101.0)]), 0.9) == pytest.approx((5, 95))


def clustered_null(rng, n_patients=100):
    """Two eyes per patient sharing a patient effect in the score and a patient-level label:
    eye-level resampling undercovers here, patient-level resampling should not."""
    label = np.repeat(rng.random(n_patients) < 0.3, 2)
    score = np.repeat(rng.normal(size=n_patients), 2) + rng.normal(scale=0.3, size=2 * n_patients)
    return score, label, np.repeat(np.arange(n_patients), 2)


def test_auc_ci_holds_nominal_coverage_under_clustered_null():
    rng = np.random.default_rng(2)
    covered = []
    for sim in range(200):
        score, label, clusters = clustered_null(rng)
        result = auc_with_ci(score, label, clusters, n_boot=300, seed=sim)
        covered.append(result["ci_low"] <= 0.5 <= result["ci_high"])
    assert 0.90 <= np.mean(covered) <= 0.99


def test_auc_ci_detects_planted_effect():
    rng = np.random.default_rng(3)
    label = np.repeat(rng.random(150) < 0.4, 2)
    score = np.repeat(rng.normal(size=150), 2) + 1.2 * label
    result = auc_with_ci(score, label, np.repeat(np.arange(150), 2), n_boot=500)
    assert result["ci_low"] > 0.5
    assert result["ci_low"] <= stats.norm.cdf(1.2 / np.sqrt(2)) <= result["ci_high"]
