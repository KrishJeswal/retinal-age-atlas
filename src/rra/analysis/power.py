import numpy as np
import polars as pl
from scipy import stats

FAMILY_ALPHA = 0.025
TIER_ALPHA = 0.05
POWER = 0.8
H1_CONTRASTS = (("N", "D"), ("N", "G"), ("D", "G"))
EXPLORATORY_CONTRASTS = (("N", "H"),)
H2_SEQUENCE = ("G", "D")
WILCOXON_ARE = 3 / np.pi
SIGMA_SWEEP = (4.0, 5.0, 6.0)
ICC_SWEEP = (0.3, 0.5, 0.7)
PAIRED_SD_SWEEP = (3.4, 4.1)
TIER_C_RATIO = 3


def z_sum(alpha: float, power: float = POWER) -> float:
    """z_{1-alpha/2} + z_{power}: the two-sided MDE multiplier."""
    return float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))


def effective_n(cluster_sizes: np.ndarray, icc: float) -> float:
    """Eyes worth of independent information in a mean over clustered eyes with exchangeable
    correlation `icc`: Var(mean) = sigma^2 sum m_i (1 + (m_i - 1) icc) / N^2, so
    n_eff = N^2 / sum m_i (1 + (m_i - 1) icc). Equal sizes m give N / DEFF, DEFF = 1 + (m-1) icc."""
    sizes = np.asarray(cluster_sizes, dtype=float)
    return float(sizes.sum() ** 2 / (sizes * (1 + (sizes - 1) * icc)).sum())


def mde_two_group(
    sizes_a: np.ndarray,
    sizes_b: np.ndarray,
    sigma: float,
    icc: float,
    alpha: float,
    power: float = POWER,
) -> float:
    """Smallest true difference in eye-level means detected with `power` by a two-sided z-test,
    eyes clustered within patients. Assumes no patient is in both groups."""
    return (
        z_sum(alpha, power)
        * sigma
        * np.sqrt(1 / effective_n(sizes_a, icc) + 1 / effective_n(sizes_b, icc))
    )


def mde_paired(
    n_pairs: int, sd_diff: float, alpha: float, power: float = POWER, efficiency: float = 1.0
) -> float:
    """Paired within-patient difference. `efficiency` < 1 scales n for a less efficient test
    (Wilcoxon signed-rank: ARE 3/pi under normality)."""
    return z_sum(alpha, power) * sd_diff / np.sqrt(n_pairs * efficiency)


def mde_matched(
    n_cases: int, ratio: int, sigma: float, alpha: float, power: float = POWER
) -> float:
    """1:k matched case-control mean difference, controls from distinct patients. Ignores the
    variance matching removes, so it is conservative."""
    return z_sum(alpha, power) * sigma * np.sqrt(1 / n_cases + 1 / (ratio * n_cases))


def mde_logistic(n: float, event_rate: float, alpha: float, power: float = POWER) -> float:
    """Tier A: smallest log odds ratio per SD of a normal predictor (Hsieh 1989 large-sample
    form, no other covariates): (z_a + z_b) / sqrt(n p (1 - p))."""
    return z_sum(alpha, power) / np.sqrt(n * event_rate * (1 - event_rate))


def cluster_sizes(groups: pl.DataFrame, group: str) -> np.ndarray:
    return groups.filter(pl.col("group") == group).group_by("patient_id").len()["len"].to_numpy()


def contrast_eye_groups(categories: pl.DataFrame, pools: pl.DataFrame) -> pl.DataFrame:
    """Long (patient_id, eye, group) for the H1 contrasts and Tier B. N = both eyes of main-pool
    patients; D, G, H = eyes carrying that keyword category (eye-level, so bilateral disease
    gives two eyes from one patient); D_mild = mild-NPDR eyes (Tier B)."""
    healthy = categories.join(
        pools.filter(pl.col("pool_main")).select("patient_id"), on="patient_id"
    ).select("patient_id", "eye", pl.lit("N").alias("group"))
    diseased = [
        categories.filter(pl.col(f"cat_{category}")).select(
            "patient_id", "eye", pl.lit(group).alias("group")
        )
        for group, category in (("D", "dr"), ("G", "glaucoma"), ("H", "hr"))
    ]
    mild = categories.filter(pl.col("dr_severity") == "mild_npdr").select(
        "patient_id", "eye", pl.lit("D_mild").alias("group")
    )
    return pl.concat([healthy, *diseased, mild])


def contrast_mde_table(
    groups: pl.DataFrame,
    sigmas: tuple[float, ...] = SIGMA_SWEEP,
    iccs: tuple[float, ...] = ICC_SWEEP,
    family_alpha: float = FAMILY_ALPHA,
    power: float = POWER,
) -> pl.DataFrame:
    """G / D6: MDE in years per contrast over the sigma x ICC sweep. H1 contrasts use the
    Holm worst case alpha / 3 (the first Holm step); exploratory contrasts the unadjusted
    family alpha, for reference only. Patients in both groups are dropped from both sides and
    counted in `overlap_patients`."""
    rows = []
    contrasts = [(c, "h1") for c in H1_CONTRASTS] + [
        (c, "exploratory") for c in EXPLORATORY_CONTRASTS
    ]
    for (a, b), family in contrasts:
        alpha = family_alpha / len(H1_CONTRASTS) if family == "h1" else family_alpha
        in_a = groups.filter(pl.col("group") == a)
        in_b = groups.filter(pl.col("group") == b)
        overlap = in_a.join(in_b.select("patient_id").unique(), on="patient_id", how="semi")[
            "patient_id"
        ].unique()
        both = groups.filter(~pl.col("patient_id").is_in(overlap.implode()))
        sizes_a, sizes_b = cluster_sizes(both, a), cluster_sizes(both, b)
        for sigma in sigmas:
            for icc in iccs:
                mde = mde_two_group(sizes_a, sizes_b, sigma, icc, alpha, power)
                rows.append(
                    {
                        "contrast": f"{a}-{b}",
                        "family": family,
                        "n_eyes_a": int(sizes_a.sum()),
                        "n_patients_a": len(sizes_a),
                        "n_eyes_b": int(sizes_b.sum()),
                        "n_patients_b": len(sizes_b),
                        "overlap_patients": overlap.len(),
                        "alpha": alpha,
                        "sigma": sigma,
                        "icc": icc,
                        "mde_years": mde,
                        "mde_sd": mde / sigma,
                    }
                )
    return pl.DataFrame(rows)


def h2_mde_table(
    pair_counts: dict[str, int],
    sd_diffs: tuple[float, ...] = PAIRED_SD_SWEEP,
    family_alpha: float = FAMILY_ALPHA,
    power: float = POWER,
    efficiency: float = WILCOXON_ARE,
) -> pl.DataFrame:
    """H2 fixed sequence (G then D): each step at the full family alpha. `pair_counts` maps
    disease group to the number of unilateral patients."""
    return pl.DataFrame(
        [
            {
                "disease": group,
                "n_pairs": pair_counts[group],
                "alpha": family_alpha,
                "sd_diff": sd,
                "mde_years": mde_paired(pair_counts[group], sd, family_alpha, power, efficiency),
            }
            for group in H2_SEQUENCE
            for sd in sd_diffs
        ]
    )


def d6_demotions(table: pl.DataFrame, plausible: dict[str, float]) -> pl.DataFrame:
    """D6: an H1 contrast is demoted when its MDE exceeds the plausible published effect.
    `table` must already be the pre-registered sigma/ICC row per contrast."""
    h1 = table.filter(pl.col("family") == "h1")
    if h1["contrast"].is_duplicated().any():
        raise ValueError("one row per contrast: filter the sweep to the pre-registered sigma/ICC")
    return h1.select(
        "contrast",
        "mde_years",
        pl.col("contrast").replace_strict(plausible).alias("plausible_years"),
    ).with_columns(demoted=pl.col("mde_years") > pl.col("plausible_years"))


def tier_mde_table(
    groups: pl.DataFrame,
    tier_c_cases: int,
    sigmas: tuple[float, ...] = SIGMA_SWEEP,
    iccs: tuple[float, ...] = ICC_SWEEP,
    alpha: float = TIER_ALPHA,
    power: float = POWER,
) -> pl.DataFrame:
    """G8 tier MDEs over the sweep. A and B use alpha / 2 (Holm worst case within the A/B pair),
    C the full alpha. A: odds ratio per year of gap (log OR per SD / sigma), on effective eyes.
    B: mild-NPDR vs healthy eyes in years. C: 1:3 matched difference in years (one eye each)."""
    sizes_n, sizes_d, sizes_b = (cluster_sizes(groups, g) for g in ("N", "D", "D_mild"))
    rows = []
    for sigma in sigmas:
        for icc in iccs:
            n_n, n_d = effective_n(sizes_n, icc), effective_n(sizes_d, icc)
            log_or_sd = mde_logistic(n_n + n_d, n_d / (n_n + n_d), alpha / 2, power)
            rows += [
                {
                    "tier": "A",
                    "n": int(sizes_d.sum()),
                    "alpha": alpha / 2,
                    "sigma": sigma,
                    "icc": icc,
                    "mde": float(np.exp(log_or_sd / sigma)),
                    "unit": "odds_ratio_per_year",
                },
                {
                    "tier": "B",
                    "n": int(sizes_b.sum()),
                    "alpha": alpha / 2,
                    "sigma": sigma,
                    "icc": icc,
                    "mde": mde_two_group(sizes_b, sizes_n, sigma, icc, alpha / 2, power),
                    "unit": "years",
                },
                {
                    "tier": "C",
                    "n": tier_c_cases,
                    "alpha": alpha,
                    "sigma": sigma,
                    "icc": icc,
                    "mde": mde_matched(tier_c_cases, TIER_C_RATIO, sigma, alpha, power),
                    "unit": "years",
                },
            ]
    return pl.DataFrame(rows)
