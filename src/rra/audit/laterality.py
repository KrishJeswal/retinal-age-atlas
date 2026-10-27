import polars as pl

from rra.audit.keywords import CATEGORIES, benign_eye

LATERALITY = pl.Enum(["none", "left_only", "right_only", "bilateral", "single_eye"])
DISEASES = tuple(c for c in CATEGORIES if c not in ("normal", "artefact"))


def eye_pairs(categories: pl.DataFrame) -> pl.DataFrame:
    """One row per patient with `<column>_left` / `<column>_right`. A patient with an excluded
    eye has nulls on that side."""
    per_eye = categories.with_columns(benign_eye().alias("benign"))
    sides = [
        per_eye.filter(pl.col("eye") == eye)
        .drop("eye")
        .rename(lambda c, eye=eye: c if c == "patient_id" else f"{c}_{eye}")
        for eye in ("left", "right")
    ]
    return sides[0].join(sides[1], on="patient_id", how="full", coalesce=True).sort("patient_id")


def laterality(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return (
        pl.when(left.is_null() | right.is_null())
        .then(pl.lit("single_eye"))
        .when(left & right)
        .then(pl.lit("bilateral"))
        .when(left)
        .then(pl.lit("left_only"))
        .when(right)
        .then(pl.lit("right_only"))
        .otherwise(pl.lit("none"))
        .cast(LATERALITY)
    )


def laterality_table(categories: pl.DataFrame) -> pl.DataFrame:
    """Patients per category by laterality (C). `single_eye` = the other eye was excluded."""
    pairs = eye_pairs(categories)
    rows = []
    for c in DISEASES:
        side = laterality(pl.col(f"cat_{c}_left"), pl.col(f"cat_{c}_right"))
        rows.append(
            pairs.select(
                pl.lit(c).alias("category"),
                *[(side == s).sum().cast(pl.Int64).alias(s) for s in LATERALITY.categories],
            )
        )
    return pl.concat(rows)


def both_eyes() -> pl.Expr:
    return pl.col("n_terms_left").is_not_null() & pl.col("n_terms_right").is_not_null()


def unilateral(category: str, benign_fellow: bool) -> pl.Expr:
    left, right = pl.col(f"cat_{category}_left"), pl.col(f"cat_{category}_right")
    if benign_fellow:
        one_sided = (left & pl.col("benign_right")) | (right & pl.col("benign_left"))
    else:
        one_sided = left ^ right
    return both_eyes() & one_sided


def tier_counts(categories: pl.DataFrame) -> pl.DataFrame:
    """Sample sizes for the go/no-go tiers and H2, before matching and power. Paired counts
    need both eyes. Tier C cases: DR in one eye, benign fellow (the fellow enters Tier C).
    H2: disease in exactly one eye; `_benign_fellow` also requires the fellow to be benign."""
    pairs = eye_pairs(categories)

    def patients(expr: pl.Expr) -> int:
        return pairs.select(expr.sum()).item()

    counts = {
        ("tier_a_dr_eyes", "eyes"): categories["cat_dr"].sum(),
        ("tier_b_mild_npdr_eyes", "eyes"): (categories["dr_severity"] == "mild_npdr").sum(),
        ("tier_c_cases", "patients"): patients(unilateral("dr", benign_fellow=True)),
    }
    for category, short in (("glaucoma", "g"), ("dr", "d")):
        for suffix, benign in (("unilateral", False), ("unilateral_benign_fellow", True)):
            counts[(f"h2_{short}_{suffix}", "patients")] = patients(unilateral(category, benign))
    return pl.DataFrame(
        {
            "count": [name for name, _ in counts],
            "unit": [unit for _, unit in counts],
            "n": [int(v) for v in counts.values()],
        }
    )


def unilateral_patients(
    categories: pl.DataFrame, category: str, benign_fellow: bool
) -> pl.DataFrame:
    """Patient IDs with `category` in exactly one eye; Tier C cases are ("dr", True)."""
    pairs = eye_pairs(categories)
    return pairs.filter(unilateral(category, benign_fellow)).select("patient_id")
