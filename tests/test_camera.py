import numpy as np
import polars as pl
import pytest

from rra.audit.camera import (
    OTHER,
    camera_clusters,
    cluster_association,
    cluster_prevalence,
    cramers_v,
)


def features(groups: list[tuple[int, int, str, int]]) -> pl.DataFrame:
    rows = []
    for width, height, shape, n in groups:
        rows += [(width, height, shape)] * n
    return pl.DataFrame(
        {
            "patient_id": np.arange(len(rows)) // 2,
            "eye": ["left", "right"] * (len(rows) // 2) + ["left"] * (len(rows) % 2),
            "decode_ok": True,
            "native_width": [r[0] for r in rows],
            "native_height": [r[1] for r in rows],
            "fov_shape": [r[2] for r in rows],
        }
    )


def test_small_clusters_merge_into_other():
    table = features(
        [(2592, 1728, "full", 60), (2592, 1728, "cut_vertical", 50), (640, 480, "full", 10)]
    )
    counts = dict(camera_clusters(table).group_by("camera_cluster").len().rows())
    assert counts == {"2592x1728/full": 60, "2592x1728/cut_vertical": 50, OTHER: 10}


def test_unmeasured_eyes_get_no_cluster():
    table = features([(2592, 1728, "full", 60)]).with_columns(
        decode_ok=pl.Series([False] + [True] * 59)
    )
    clusters = camera_clusters(table)
    assert clusters["camera_cluster"].null_count() == 1
    assert clusters["camera_cluster"].drop_nulls().unique().to_list() == ["2592x1728/full"]


def test_prevalence_per_cluster():
    clusters = camera_clusters(features([(1, 1, "full", 4), (2, 2, "full", 2)]), min_eyes=1)
    categories = clusters.select("patient_id", "eye").with_columns(
        cat_dr=pl.Series([True, True, False, False, True, False])
    )
    prevalence = dict(
        cluster_prevalence(clusters, categories).select("camera_cluster", "prev_dr").rows()
    )
    assert prevalence == {"1x1/full": 0.5, "2x2/full": 0.5}


def test_cramers_v_known_tables():
    assert cramers_v(np.array([[50, 0], [0, 50]]))["cramers_v"] == pytest.approx(1.0)
    assert cramers_v(np.array([[25, 25], [25, 25]]))["cramers_v"] == pytest.approx(0.0)
    assert np.isnan(cramers_v(np.array([[10, 0], [5, 0]]))["chi2"])


def simulate(rng, prevalences, n_per_cluster):
    rows = []
    for k, p in enumerate(prevalences):
        rows.append(np.column_stack([np.full(n_per_cluster, k), rng.random(n_per_cluster) < p]))
    data = np.vstack(rows)
    return np.array(
        [
            [np.sum((data[:, 0] == k) & (data[:, 1] == v)) for v in (0, 1)]
            for k in range(len(prevalences))
        ]
    )


def test_cramers_v_recovers_planted_association():
    prevalences = np.array([0.1, 0.3, 0.5])
    mean = prevalences.mean()
    planted = np.sqrt(prevalences.var() / (mean * (1 - mean)))
    table = simulate(np.random.default_rng(0), prevalences, 20000)
    assert cramers_v(table)["cramers_v"] == pytest.approx(planted, abs=0.01)


def test_chi_square_holds_nominal_type_one_error():
    rng = np.random.default_rng(1)
    p = [cramers_v(simulate(rng, [0.2, 0.2, 0.2, 0.2], 200))["p"] for _ in range(1000)]
    assert 0.03 <= np.mean(np.array(p) < 0.05) <= 0.07


def test_cluster_association_one_row_per_category():
    clusters = camera_clusters(features([(1, 1, "full", 100), (2, 2, "full", 100)]))
    categories = clusters.select("patient_id", "eye").with_columns(
        cat_dr=pl.Series([True] * 100 + [False] * 100),
        cat_normal=pl.Series([True, False] * 100),
    )
    result = dict(cluster_association(clusters, categories).select("category", "cramers_v").rows())
    assert result["dr"] == pytest.approx(1.0)
    assert result["normal"] == pytest.approx(0.0)
