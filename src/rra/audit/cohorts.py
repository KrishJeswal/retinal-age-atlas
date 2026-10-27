import polars as pl

from rra.audit.keywords import BENIGN, only_categories
from rra.data.manifest import KEY

POOL_CATEGORIES = {
    "strict": ("normal",),
    "main": BENIGN,
    "low_quality": (*BENIGN, "quality"),
}


def healthy_pools(eyes: pl.DataFrame, categories: pl.DataFrame) -> pl.DataFrame:
    """D1 pools per patient: N label and both eyes present with only the pool's categories.
    Nested by construction: strict ⊆ main ⊆ low_quality. `main` is the training pool."""
    per_eye = (
        eyes.select(*KEY, "N")
        .join(categories, on=KEY)
        .select(
            "patient_id",
            "N",
            *[only_categories(allowed).alias(pool) for pool, allowed in POOL_CATEGORIES.items()],
        )
    )
    return (
        per_eye.group_by("patient_id")
        .agg(
            *[
                ((pl.col("N") == 1).all() & pl.col(pool).all() & (pl.len() == 2)).alias(
                    f"pool_{pool}"
                )
                for pool in POOL_CATEGORIES
            ]
        )
        .sort("patient_id")
    )
