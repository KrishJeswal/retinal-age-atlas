import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import stats

from rra.analysis.power import (
    WILCOXON_ARE,
    contrast_eye_groups,
    contrast_mde_table,
    d6_demotions,
    effective_n,
    h2_mde_table,
    mde_logistic,
    mde_matched,
    mde_paired,
    mde_two_group,
    tier_mde_table,
    z_sum,
)

ALPHA = 0.05
SIMS = 1000


def within(rate: float, target: float, sims: int = SIMS) -> bool:
    """Rate within 3.5 binomial SEs of target."""
    return abs(rate - target) <= 3.5 * np.sqrt(target * (1 - target) / sims)


def test_z_sum_textbook_value():
    assert z_sum(0.05, 0.8) == pytest.approx(1.95996 + 0.84162, abs=1e-4)


def test_textbook_example_from_project_md():
    mde = z_sum(0.05) * 5 * np.sqrt(1 / 1100 + 1 / 100)
    assert mde_two_group(np.ones(1100), np.ones(100), 5, 0.0, 0.05) == pytest.approx(mde)
    assert mde == pytest.approx(1.47, abs=0.01)


@given(st.lists(st.integers(1, 2), min_size=1, max_size=50), st.floats(0, 1), st.floats(0, 1))
def test_effective_n_falls_with_icc_between_independent_and_fully_correlated(sizes, a, b):
    sizes = np.array(sizes)
    low, high = sorted((a, b))
    assert effective_n(sizes, high) <= effective_n(sizes, low) + 1e-9
    assert sizes.sum() ** 2 / (sizes**2).sum() - 1e-9 <= effective_n(sizes, high)
    assert effective_n(sizes, 0.0) == pytest.approx(sizes.sum())


def test_effective_n_equal_clusters_is_n_over_deff():
    assert effective_n(np.full(100, 2), 0.5) == pytest.approx(200 / 1.5)
    assert effective_n(np.full(100, 2), 0.0) == pytest.approx(200)
    assert effective_n(np.full(100, 2), 1.0) == pytest.approx(100)


def clustered_group(rng, sizes, sigma, icc, shift):
    patient = np.repeat(rng.normal(scale=sigma * np.sqrt(icc), size=len(sizes)), sizes)
    eye = rng.normal(scale=sigma * np.sqrt(1 - icc), size=sizes.sum())
    return shift + patient + eye, np.repeat(np.arange(len(sizes)), sizes)


def cluster_robust_var(values, patients):
    """Sandwich variance of an eye-level mean with patient clusters."""
    sums = np.bincount(patients, values)
    counts = np.bincount(patients)
    return ((sums - counts * values.mean()) ** 2).sum() / len(values) ** 2


def two_group_rejection_rate(effect, sizes_a, sizes_b, sigma, icc, seed):
    rng = np.random.default_rng(seed)
    critical = stats.norm.ppf(1 - ALPHA / 2)
    rejected = 0
    for _ in range(SIMS):
        a, pa = clustered_group(rng, sizes_a, sigma, icc, effect)
        b, pb = clustered_group(rng, sizes_b, sigma, icc, 0.0)
        se = np.sqrt(cluster_robust_var(a, pa) + cluster_robust_var(b, pb))
        rejected += abs(a.mean() - b.mean()) / se > critical
    return rejected / SIMS


SIZES_A = np.array([2] * 300 + [1] * 100)
SIZES_B = np.array([2] * 80 + [1] * 60)


def test_two_group_mde_gives_nominal_power_with_clustered_eyes():
    mde = mde_two_group(SIZES_A, SIZES_B, 5.0, 0.5, ALPHA)
    assert within(two_group_rejection_rate(mde, SIZES_A, SIZES_B, 5.0, 0.5, seed=0), 0.8)


def test_two_group_test_holds_type_one_error():
    assert within(two_group_rejection_rate(0.0, SIZES_A, SIZES_B, 5.0, 0.5, seed=1), ALPHA)


def test_ignoring_clustering_would_understate_the_mde():
    naive = mde_two_group(np.ones(SIZES_A.sum()), np.ones(SIZES_B.sum()), 5.0, 0.0, ALPHA)
    mde = mde_two_group(SIZES_A, SIZES_B, 5.0, 0.5, ALPHA)
    assert two_group_rejection_rate(naive, SIZES_A, SIZES_B, 5.0, 0.5, seed=2) < 0.7
    assert mde > naive


def test_paired_mde_gives_nominal_power():
    rng = np.random.default_rng(3)
    n, sd = 120, 3.8
    mde = mde_paired(n, sd, ALPHA)
    diffs = rng.normal(mde, sd, size=(SIMS, n))
    t = diffs.mean(axis=1) / (diffs.std(axis=1, ddof=1) / np.sqrt(n))
    assert within(np.mean(np.abs(t) > stats.t.ppf(1 - ALPHA / 2, n - 1)), 0.8)


@pytest.mark.parametrize(("effect_scale", "target"), [(1.0, 0.8), (0.0, ALPHA)])
def test_wilcoxon_mde_with_are_gives_nominal_power_and_size(effect_scale, target):
    rng = np.random.default_rng(4)
    n, sd = 120, 3.8
    effect = effect_scale * mde_paired(n, sd, ALPHA, efficiency=WILCOXON_ARE)
    diffs = rng.normal(effect, sd, size=(SIMS, n))
    p = stats.wilcoxon(diffs, axis=1).pvalue
    rate = np.mean(p < ALPHA)
    if effect_scale:
        assert 0.8 - 0.02 <= rate <= 0.9
    else:
        assert within(rate, target)


def test_matched_mde_gives_at_least_nominal_power():
    rng = np.random.default_rng(5)
    n, sigma = 150, 5.0
    mde = mde_matched(n, 3, sigma, ALPHA)
    cases = rng.normal(mde, sigma, size=(SIMS, n))
    controls = rng.normal(0, sigma, size=(SIMS, 3 * n))
    se = np.sqrt(cases.var(axis=1, ddof=1) / n + controls.var(axis=1, ddof=1) / (3 * n))
    z = (cases.mean(axis=1) - controls.mean(axis=1)) / se
    assert within(np.mean(np.abs(z) > stats.norm.ppf(1 - ALPHA / 2)), 0.8)


def test_logistic_mde_gives_nominal_power():
    rng = np.random.default_rng(6)
    n, rate = 1500, 0.3
    beta = mde_logistic(n, rate, ALPHA)
    intercept = np.log(rate / (1 - rate))
    rejected = 0
    sims = 400
    for _ in range(sims):
        x = rng.normal(size=n)
        y = rng.random(n) < 1 / (1 + np.exp(-(intercept + beta * x)))
        rejected += abs(stats.pearsonr(x, y).statistic) * np.sqrt(n) > stats.norm.ppf(1 - ALPHA / 2)
    assert within(rejected / sims, 0.8, sims)


def toy_categories() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows = []
    for patient in range(40):
        for eye in ("left", "right"):
            rows.append(
                {
                    "patient_id": patient,
                    "eye": eye,
                    "cat_dr": 10 <= patient < 20 or (patient == 20 and eye == "left"),
                    "cat_glaucoma": 18 <= patient < 30,
                    "cat_hr": patient == 31,
                    "dr_severity": "mild_npdr" if 10 <= patient < 14 else None,
                }
            )
    categories = pl.DataFrame(rows)
    pools = pl.DataFrame({"patient_id": range(40), "pool_main": [p < 10 for p in range(40)]})
    return categories, pools


def test_contrast_groups_and_overlap():
    groups = contrast_eye_groups(*toy_categories())
    table = contrast_mde_table(groups, sigmas=(5.0,), iccs=(0.5,))
    row = {r["contrast"]: r for r in table.to_dicts()}
    assert (row["N-D"]["n_eyes_a"], row["N-D"]["n_eyes_b"], row["N-D"]["n_patients_b"]) == (
        20,
        21,
        11,
    )
    assert row["D-G"]["overlap_patients"] == 3
    assert (row["D-G"]["n_patients_a"], row["D-G"]["n_patients_b"]) == (8, 9)
    assert row["N-D"]["alpha"] == pytest.approx(0.025 / 3)
    assert row["N-H"]["family"] == "exploratory"
    assert row["N-H"]["alpha"] == pytest.approx(0.025)


def test_contrast_table_sweeps_and_mde_grows_with_sigma_and_icc():
    table = contrast_mde_table(contrast_eye_groups(*toy_categories()))
    assert table.height == 4 * 3 * 3
    nd = table.filter(pl.col("contrast") == "N-D").sort("sigma", "icc")["mde_years"].to_numpy()
    assert np.all(np.diff(nd.reshape(3, 3), axis=1) > 0)
    assert np.all(np.diff(nd.reshape(3, 3), axis=0) > 0)


def test_h2_table_full_alpha_in_fixed_sequence():
    table = h2_mde_table({"G": 100, "D": 200})
    assert table["disease"].to_list() == ["G", "G", "D", "D"]
    assert (table["alpha"] == 0.025).all()
    assert table["mde_years"][0] == pytest.approx(
        mde_paired(100, 3.4, 0.025, efficiency=WILCOXON_ARE)
    )


def test_tier_table_units_and_alpha_split():
    table = tier_mde_table(contrast_eye_groups(*toy_categories()), tier_c_cases=30)
    by_tier = table.filter((pl.col("sigma") == 5.0) & (pl.col("icc") == 0.5)).to_dicts()
    tiers = {r["tier"]: r for r in by_tier}
    assert tiers["A"]["unit"] == "odds_ratio_per_year"
    assert tiers["A"]["mde"] > 1
    assert tiers["A"]["alpha"] == tiers["B"]["alpha"] == pytest.approx(0.025)
    assert tiers["C"]["alpha"] == pytest.approx(0.05)
    assert tiers["B"]["n"] == 8
    assert tiers["C"]["mde"] == pytest.approx(mde_matched(30, 3, 5.0, 0.05))


def test_d6_demotes_when_mde_exceeds_plausible_effect():
    table = contrast_mde_table(contrast_eye_groups(*toy_categories()), sigmas=(5.0,), iccs=(0.5,))
    mde = dict(table.select("contrast", "mde_years").rows())
    plausible = {"N-D": mde["N-D"] + 1, "N-G": mde["N-G"] - 1, "D-G": mde["D-G"] + 0.1}
    decisions = dict(d6_demotions(table, plausible).select("contrast", "demoted").rows())
    assert decisions == {"N-D": False, "N-G": True, "D-G": False}


def test_d6_refuses_an_unfiltered_sweep():
    table = contrast_mde_table(contrast_eye_groups(*toy_categories()))
    with pytest.raises(ValueError, match="one row per contrast"):
        d6_demotions(table, {"N-D": 1.0, "N-G": 1.0, "D-G": 1.0})
