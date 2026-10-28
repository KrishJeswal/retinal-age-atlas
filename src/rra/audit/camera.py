import numpy as np
import polars as pl
from scipy import stats

from rra.data.manifest import KEY

MIN_CLUSTER_EYES = 50
OTHER = "other"


def camera_clusters(features: pl.DataFrame, min_eyes: int = MIN_CLUSTER_EYES) -> pl.DataFrame:
    """D4: camera proxy = native resolution x FOV-shape class. Clusters with fewer than
    `min_eyes` eyes merge into "other" (the threshold was fixed before seeing ODIR counts).
    Null for eyes the sweep could not measure."""
    measured = pl.col("decode_ok").fill_null(False) & pl.col("fov_shape").is_not_null()
    raw = pl.concat_str(
        pl.col("native_width"), pl.lit("x"), pl.col("native_height"), pl.lit("/"), "fov_shape"
    )
    return (
        features.select(*KEY, pl.when(measured).then(raw).alias("camera_raw"))
        .with_columns(size=pl.len().over("camera_raw"))
        .select(
            *KEY,
            "camera_raw",
            pl.when(pl.col("camera_raw").is_null())
            .then(None)
            .when(pl.col("size") < min_eyes)
            .then(pl.lit(OTHER))
            .otherwise(pl.col("camera_raw"))
            .alias("camera_cluster"),
        )
    )


def cluster_prevalence(clusters: pl.DataFrame, categories: pl.DataFrame) -> pl.DataFrame:
    """Per cluster: eyes, patients and the share of eyes carrying each keyword category."""
    flags = [c for c in categories.columns if c.startswith("cat_")]
    return (
        clusters.drop_nulls("camera_cluster")
        .join(categories.select(*KEY, *flags), on=KEY)
        .group_by("camera_cluster")
        .agg(
            pl.len().alias("n_eyes"),
            pl.col("patient_id").n_unique().alias("n_patients"),
            *[pl.col(f).mean().alias(f"prev_{f.removeprefix('cat_')}") for f in flags],
        )
        .sort("n_eyes", descending=True)
    )


def cramers_v(table: np.ndarray) -> dict[str, float]:
    """Pearson chi-square (no continuity correction) and Cramer's V for an r x c count table.
    Rows or columns with no counts are dropped; NaN if fewer than two remain on either side."""
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if min(table.shape) < 2:
        return {"chi2": np.nan, "dof": np.nan, "p": np.nan, "cramers_v": np.nan}
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    v = np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1)))
    return {"chi2": float(chi2), "dof": float(dof), "p": float(p), "cramers_v": float(v)}


def cluster_association(clusters: pl.DataFrame, categories: pl.DataFrame) -> pl.DataFrame:
    """E: per category, cluster x (category present) chi-square and Cramer's V over eyes.
    Descriptive: both eyes of a patient usually share camera and disease, so the eye-level p
    is anti-conservative; V is the quantity reported."""
    data = clusters.drop_nulls("camera_cluster").join(categories, on=KEY)
    rows = []
    for flag in [c for c in categories.columns if c.startswith("cat_")]:
        table = (
            data.pivot(
                on=flag,
                index="camera_cluster",
                values="patient_id",
                aggregate_function="len",
            )
            .fill_null(0)
            .drop("camera_cluster")
            .to_numpy()
        )
        rows.append(
            {"category": flag.removeprefix("cat_"), "n_eyes": data.height, **cramers_v(table)}
        )
    return pl.DataFrame(rows)
