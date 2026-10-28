import numpy as np
import polars as pl

FOLDS = 5
STRATUM_AGE_WIDTH = 10
MIXED = "mixed"


def patient_clusters(clusters: pl.DataFrame) -> pl.DataFrame:
    """One camera cluster per patient for patient-level stratification and matching: the
    cluster of the measured eyes if they agree, else "mixed"."""
    return (
        clusters.drop_nulls("camera_cluster")
        .group_by("patient_id")
        .agg(pl.col("camera_cluster").unique())
        .select(
            "patient_id",
            pl.when(pl.col("camera_cluster").list.len() == 1)
            .then(pl.col("camera_cluster").list.first())
            .otherwise(pl.lit(MIXED))
            .alias("camera_cluster"),
        )
    )


def strata(patients: pl.DataFrame, age_width: int = STRATUM_AGE_WIDTH) -> pl.DataFrame:
    """patient_id + stratum = age bin x camera cluster (G: folds stratified by both)."""
    return patients.select(
        "patient_id",
        pl.concat_str(
            ((pl.col("age") // age_width) * age_width).cast(pl.String),
            pl.lit("/"),
            pl.col("camera_cluster").fill_null("unknown"),
        ).alias("stratum"),
    )


def fold_assignment(stratified: pl.DataFrame, folds: int = FOLDS, seed: int = 0) -> pl.DataFrame:
    """Patient-level stratified folds: patients are shuffled within each stratum and dealt
    round-robin, the dealer position carrying over between strata, so every stratum is split
    within +-1 across folds and the folds' total sizes differ by at most 1. Both eyes of a
    patient share the fold by construction."""
    rng = np.random.default_rng(seed)
    ordered = stratified.sort("stratum", "patient_id")
    ids, labels = ordered["patient_id"].to_numpy(), ordered["stratum"].to_numpy()
    fold = np.empty(len(ids), dtype=np.int64)
    position = 0
    for stratum in np.unique(labels):
        members = np.flatnonzero(labels == stratum)
        shuffled = rng.permutation(members)
        fold[shuffled] = (position + np.arange(len(members))) % folds
        position += len(members)
    return pl.DataFrame({"patient_id": ids, "fold": fold}).sort("patient_id")


def fold_feasibility(
    patients: pl.DataFrame,
    folds: int = FOLDS,
    seed: int = 0,
    age_width: int = STRATUM_AGE_WIDTH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """H: healthy patients per age bin x fold under the stratified assignment, and the strata
    too small to place a patient in every fold (candidates for merging before splits are made)."""
    stratified = strata(patients, age_width)
    assigned = fold_assignment(stratified, folds, seed).join(patients, on="patient_id")
    per_bin = (
        assigned.with_columns(((pl.col("age") // age_width) * age_width).alias("bin_start"))
        .group_by("bin_start", "fold")
        .len()
        .pivot(on="fold", index="bin_start", values="len", sort_columns=True)
        .fill_null(0)
        .rename(lambda c: c if c == "bin_start" else f"fold_{c}")
        .sort("bin_start")
    )
    sparse = (
        stratified.group_by("stratum")
        .len()
        .rename({"len": "n_patients"})
        .filter(pl.col("n_patients") < folds)
        .sort("stratum")
    )
    return per_bin, sparse
