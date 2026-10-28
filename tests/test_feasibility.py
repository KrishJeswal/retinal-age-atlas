import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.audit.feasibility import (
    MIXED,
    fold_assignment,
    fold_feasibility,
    patient_clusters,
    strata,
)
from rra.audit.quality import yellowness_slopes


def test_patient_cluster_agrees_or_is_mixed():
    clusters = pl.DataFrame(
        {
            "patient_id": [0, 0, 1, 1, 2, 2],
            "eye": ["left", "right"] * 3,
            "camera_cluster": ["a", "a", "a", "b", "b", None],
        }
    )
    result = dict(patient_clusters(clusters).sort("patient_id").rows())
    assert result == {0: "a", 1: MIXED, 2: "b"}


@given(
    st.lists(st.sampled_from(["30/a", "40/a", "40/b", "50/c"]), min_size=1, max_size=200),
    st.integers(2, 6),
    st.integers(0, 3),
)
def test_folds_balance_every_stratum_and_the_totals(labels, folds, seed):
    stratified = pl.DataFrame({"patient_id": range(len(labels)), "stratum": labels})
    assigned = fold_assignment(stratified, folds, seed).join(stratified, on="patient_id")
    assert assigned.height == len(labels)
    per_stratum = assigned.group_by("stratum", "fold").len()
    for (_,), part in per_stratum.group_by("stratum"):
        counts = part["len"].to_list() + [0] * (folds - part.height)
        assert max(counts) - min(counts) <= 1
    totals = assigned.group_by("fold").len()["len"].to_list() + [0] * (
        folds - assigned["fold"].n_unique()
    )
    assert max(totals) - min(totals) <= 1


def test_fold_assignment_is_seeded():
    stratified = pl.DataFrame({"patient_id": range(50), "stratum": ["x"] * 50})
    assert fold_assignment(stratified, seed=1).equals(fold_assignment(stratified, seed=1))
    assert not fold_assignment(stratified, seed=1).equals(fold_assignment(stratified, seed=2))


def test_fold_feasibility_counts_and_sparse_strata():
    patients = pl.DataFrame(
        {
            "patient_id": range(23),
            "age": [45] * 20 + [72] * 3,
            "camera_cluster": ["a"] * 20 + ["b"] * 3,
        }
    )
    per_bin, sparse = fold_feasibility(patients, folds=5)
    assert per_bin.filter(pl.col("bin_start") == 40).drop("bin_start").row(0) == (4,) * 5
    assert per_bin.drop("bin_start").sum_horizontal().sum() == 23
    assert sparse.rows() == [("70/b", 3)]


def test_strata_without_camera_use_unknown():
    patients = pl.DataFrame({"patient_id": [0], "age": [61], "camera_cluster": [None]})
    assert strata(patients.cast({"camera_cluster": pl.String}))["stratum"].item() == "60/unknown"


def yellow_cohort(rng, n_patients, slope, clusters=("a", "b")):
    age = rng.uniform(30, 80, n_patients)
    cluster = rng.choice(clusters, n_patients)
    patient_effect = rng.normal(scale=2, size=n_patients)
    rows = {
        "patient_id": np.repeat(np.arange(n_patients), 2),
        "age": np.repeat(age, 2),
        "camera_cluster": np.repeat(cluster, 2),
        "lab_b": np.repeat(slope * age + patient_effect, 2) + rng.normal(size=2 * n_patients),
    }
    return pl.DataFrame(rows)


def test_yellowness_slope_recovers_planted_effect():
    result = yellowness_slopes(yellow_cohort(np.random.default_rng(0), 400, 0.1), n_boot=300)
    overall = result.filter(pl.col("camera_cluster") == "all").row(0, named=True)
    assert overall["slope_per_year"] == pytest.approx(0.1, abs=0.02)
    assert overall["ci_low"] <= 0.1 <= overall["ci_high"]
    assert result["camera_cluster"].to_list() == ["all", "a", "b"]


def test_yellowness_ci_covers_zero_at_nominal_rate_under_null():
    rng = np.random.default_rng(1)
    covered = []
    for sim in range(150):
        result = yellowness_slopes(yellow_cohort(rng, 80, 0.0, ("a",)), n_boot=200, seed=sim)
        overall = result.filter(pl.col("camera_cluster") == "all").row(0, named=True)
        covered.append(overall["ci_low"] <= 0 <= overall["ci_high"])
    assert 0.89 <= np.mean(covered) <= 0.99
