import numpy as np
import polars as pl
import pytest

from rra.audit.quality import (
    QUALITY_DIRECTION,
    crossfit_logistic_score,
    low_quality_keyword,
    normal_scores,
    quality_score,
    validate_quality,
)


def cohort(rng, n_patients: int, effect: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Latent badness drives every metric (in its quality direction) with strength `effect` and
    drives the keyword; effect 0 is the null. Two eyes per patient share part of the latent."""
    n = 2 * n_patients
    latent = np.repeat(rng.normal(size=n_patients), 2) * 0.5 + rng.normal(size=n)
    metrics = {
        name: np.exp(sign * effect * latent + rng.normal(size=n))
        for name, sign in QUALITY_DIRECTION.items()
    }
    keys = {
        "patient_id": np.repeat(np.arange(n_patients), 2),
        "eye": ["left", "right"] * n_patients,
    }
    flagged = latent + rng.normal(scale=0.5, size=n) > 1.2
    features = pl.DataFrame({**keys, **metrics})
    categories = pl.DataFrame(
        {
            **keys,
            "cat_quality": flagged & (rng.random(n) < 0.7),
            "cat_ungradable": flagged & (rng.random(n) >= 0.7),
        }
    )
    return features, categories


def test_normal_scores_are_centred_ranks_with_nan_kept():
    scores = normal_scores(np.array([3.0, 1.0, np.nan, 2.0, 2.0]))
    assert np.isnan(scores[2])
    assert scores[3] == scores[4] == pytest.approx(0.0)
    assert scores[0] == pytest.approx(-scores[1])


def test_score_is_higher_for_worse_metrics():
    features = pl.DataFrame(
        {
            "patient_id": [0, 0, 1],
            "eye": ["left", "right", "left"],
            "sharpness": [10.0, 1.0, 5.0],
            "contrast": [0.3, 0.1, 0.2],
            "illum_cv": [0.1, 0.5, 0.3],
            "clip_high": [0.0, 0.2, 0.1],
            "clip_low": [0.0, 0.2, 0.1],
        }
    )
    assert quality_score(features)["quality_score"].to_list() == pytest.approx(
        [-0.967, 0.967, 0.0], abs=1e-3
    )


def test_score_is_invariant_to_monotone_transforms_and_keeps_nulls():
    features, _ = cohort(np.random.default_rng(0), 50, 1.0)
    features = features.with_columns(
        pl.when(pl.int_range(pl.len()) == 3)
        .then(None)
        .otherwise(pl.col("sharpness"))
        .alias("sharpness")
    )
    transformed = features.with_columns(pl.col("sharpness").log(), pl.col("clip_high").sqrt())
    original, again = (
        quality_score(features)["quality_score"],
        quality_score(transformed)["quality_score"],
    )
    assert original.null_count() == 1
    assert original[3] is None
    assert original.to_numpy() == pytest.approx(again.to_numpy(), nan_ok=True)


def test_low_quality_target_is_quality_or_ungradable():
    categories = pl.DataFrame(
        {
            "patient_id": [0, 0, 1],
            "eye": ["left", "right", "left"],
            "cat_quality": [True, False, False],
            "cat_ungradable": [False, True, False],
        }
    )
    assert low_quality_keyword(categories)["low_quality_kw"].to_list() == [True, True, False]


def test_validation_recovers_planted_quality_signal():
    features, categories = cohort(np.random.default_rng(1), 400, 1.0)
    result = validate_quality(features, categories, n_boot=300)
    assert result["auc"] > 0.8
    assert result["ci_low"] > 0.75
    assert result["logistic_auc"] > 0.8
    assert result["passes"]


def test_validation_under_the_null_is_chance():
    rng = np.random.default_rng(2)
    fixed, logistic = [], []
    for sim in range(20):
        features, categories = cohort(rng, 200, 0.0)
        result = validate_quality(features, categories, n_boot=200, seed=sim)
        fixed.append(result["auc"])
        logistic.append(result["logistic_auc"])
        assert not result["passes"]
    assert np.mean(fixed) == pytest.approx(0.5, abs=0.03)
    assert np.mean(logistic) == pytest.approx(0.5, abs=0.04)


def test_logistic_score_is_out_of_fold_by_patient():
    features, categories = cohort(np.random.default_rng(3), 100, 1.0)
    target = low_quality_keyword(categories)
    first = crossfit_logistic_score(features, target, seed=0)
    assert first.height == features.height
    shuffled = crossfit_logistic_score(features, target, seed=1)
    assert not np.allclose(first["logistic_score"], shuffled["logistic_score"])


def test_logistic_drops_constant_metrics_and_reports_unfittable_as_nan():
    features, categories = cohort(np.random.default_rng(4), 100, 1.0)
    target = low_quality_keyword(categories)
    constant = features.with_columns(clip_low=pl.lit(0.0))
    assert crossfit_logistic_score(constant, target)["logistic_score"].is_not_nan().all()
    separable = target.with_columns(
        low_quality_kw=pl.Series(features["sharpness"] < features["sharpness"].median())
    )
    assert crossfit_logistic_score(features, separable)["logistic_score"].is_nan().all()
