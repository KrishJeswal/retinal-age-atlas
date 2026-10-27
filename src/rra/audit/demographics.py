import polars as pl

from rra.audit.integrity import normalised_sex
from rra.data.odir import LABELS

REFERENCE_GROUP = "healthy_main"
POOL_GROUPS = {
    "pool_strict": "healthy_strict",
    "pool_main": "healthy_main",
    "pool_low_quality": "healthy_low_quality",
}


def patient_table(eyes: pl.DataFrame) -> pl.DataFrame:
    """One row per patient with at least one included eye; sex normalised to male/female."""
    return (
        eyes.group_by("patient_id")
        .agg(
            pl.col("age").first(),
            normalised_sex().first().alias("sex"),
            *[pl.col(LABELS).first()],
        )
        .sort("patient_id")
    )


def group_membership(patients: pl.DataFrame, pools: pl.DataFrame) -> pl.DataFrame:
    """Long (patient_id, group): the three D1 pools plus one group per patient label.
    Groups overlap (ODIR labels are multi-label; pools are nested)."""
    pool_rows = [
        pools.filter(pl.col(column)).select("patient_id", pl.lit(group).alias("group"))
        for column, group in POOL_GROUPS.items()
    ]
    label_rows = [
        patients.filter(pl.col(label) == 1).select("patient_id", pl.lit(label).alias("group"))
        for label in LABELS
    ]
    return pl.concat(pool_rows + label_rows)


def demographics_table(
    patients: pl.DataFrame, membership: pl.DataFrame, reference: str = REFERENCE_GROUP
) -> pl.DataFrame:
    """Age and sex per group, with standardised mean differences against `reference`:
    SMD = (m_g - m_ref) / sqrt((s_g^2 + s_ref^2) / 2); for a proportion, s^2 = p(1 - p)."""
    female = (pl.col("sex") == "female").cast(pl.Float64)
    summary = (
        membership.join(patients, on="patient_id")
        .group_by("group", maintain_order=True)
        .agg(
            pl.len().cast(pl.Int64).alias("n"),
            pl.col("age").mean().alias("age_mean"),
            pl.col("age").std().alias("age_sd"),
            pl.col("age").median().alias("age_median"),
            pl.col("age").quantile(0.25, interpolation="linear").alias("age_q1"),
            pl.col("age").quantile(0.75, interpolation="linear").alias("age_q3"),
            pl.col("age").min().alias("age_min"),
            pl.col("age").max().alias("age_max"),
            pl.col("age").skew(bias=False).alias("age_skew"),
            female.mean().alias("p_female"),
        )
    )
    ref = summary.filter(pl.col("group") == reference)
    if ref.is_empty():
        return summary.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("age_smd"),
            pl.lit(None, dtype=pl.Float64).alias("female_smd"),
        )
    ref_age, ref_sd, ref_p = ref.select("age_mean", "age_sd", "p_female").row(0)
    p = pl.col("p_female")
    return summary.with_columns(
        ((pl.col("age_mean") - ref_age) / ((pl.col("age_sd") ** 2 + ref_sd**2) / 2).sqrt()).alias(
            "age_smd"
        ),
        ((p - ref_p) / ((p * (1 - p) + ref_p * (1 - ref_p)) / 2).sqrt()).alias("female_smd"),
    )


def age_bins(patients: pl.DataFrame, membership: pl.DataFrame, width: int = 5) -> pl.DataFrame:
    """Counts per group and age bin [start, start + width), for the healthy-skew check."""
    return (
        membership.join(patients, on="patient_id")
        .with_columns(((pl.col("age") // width) * width).alias("bin_start"))
        .group_by("group", "bin_start")
        .len()
        .rename({"len": "n"})
        .with_columns(pl.col("n").cast(pl.Int64))
        .sort("group", "bin_start")
    )


def matching_candidates(
    cases: pl.DataFrame, controls: pl.DataFrame, caliper: float, exact: tuple[str, ...]
) -> pl.DataFrame:
    """Per case, the number of controls within ±caliper years of age and equal on `exact`
    (with replacement, so an upper bound on what G7 matching can achieve)."""
    pairs = cases.select("patient_id", "age", *exact).join(
        controls.select(pl.col("patient_id").alias("control_id"), "age", *exact),
        on=list(exact) if exact else None,
        how="inner" if exact else "cross",
        suffix="_control",
    )
    within = pairs.filter((pl.col("age") - pl.col("age_control")).abs() <= caliper)
    return (
        cases.select("patient_id")
        .join(within.group_by("patient_id").len(), on="patient_id", how="left")
        .select("patient_id", pl.col("len").fill_null(0).cast(pl.Int64).alias("n_candidates"))
        .sort("patient_id")
    )


def matching_feasibility(
    cases: pl.DataFrame,
    controls: pl.DataFrame,
    caliper: float = 2,
    ratio: int = 3,
    exact: tuple[str, ...] = ("sex",),
) -> pl.DataFrame:
    """G7 defaults: 1:3, age caliper of 2 yr, exact sex. Camera cluster joins `exact` once the
    image audit exists. `controls_needed` vs `n_controls` is only a crude capacity check;
    the G7 matcher (without replacement) gives the real answer."""
    counts = matching_candidates(cases, controls, caliper, exact)["n_candidates"]
    return pl.DataFrame(
        {
            "n_cases": [cases.height],
            "n_controls": [controls.height],
            "candidates_min": [counts.min() if len(counts) else None],
            "candidates_median": [float(counts.median()) if len(counts) else None],
            "cases_with_ratio": [int((counts >= ratio).sum())],
            "controls_needed": [ratio * cases.height],
        }
    )
