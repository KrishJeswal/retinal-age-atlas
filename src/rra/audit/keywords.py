import polars as pl

from rra.data.manifest import KEY
from rra.data.odir import LABELS


def normalised_terms(keywords: pl.Expr) -> pl.Expr:
    """NFKC folds the full-width comma (ODIR's separator) to ',' along with full-width
    spaces and letters, so one split handles both comma forms."""
    return (
        keywords.str.normalize("NFKC")
        .str.to_lowercase()
        .str.split(",")
        .list.eval(pl.element().str.replace_all(r"\s+", " ").str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.unique(maintain_order=True)
    )


def eye_terms(eyes: pl.DataFrame) -> pl.DataFrame:
    """One row per (patient_id, eye, term). Eyes with null or empty keywords contribute no rows."""
    return (
        eyes.select(*KEY, normalised_terms(pl.col("keywords")).alias("term"))
        .explode("term", empty_as_null=True)
        .drop_nulls("term")
    )


def vocabulary(eyes: pl.DataFrame) -> pl.DataFrame:
    """Per-term counts for the D2 mapping review. `with_<label>` counts eyes carrying the term
    whose patient has that label, to show how readers' labels relate to each term."""
    return (
        eye_terms(eyes)
        .join(eyes.select(*KEY, *LABELS), on=KEY)
        .group_by("term")
        .agg(
            pl.len().alias("n_eyes"),
            pl.col("patient_id").n_unique().alias("n_patients"),
            (pl.col("eye") == "left").sum().alias("n_left"),
            (pl.col("eye") == "right").sum().alias("n_right"),
            *[pl.col(label).cast(pl.Int64).sum().alias(f"with_{label}") for label in LABELS],
        )
        .with_columns(pl.exclude("term").cast(pl.Int64))
        .sort(["n_eyes", "term"], descending=[True, False])
    )
