import polars as pl

from rra.data.manifest import KEY

MIN_AGE = 18
MAX_PLAUSIBLE_AGE = 100
SEXES = ("male", "female")
IMAGE_EXTENSIONS = r"\.(jpe?g|png)$"

REASONS = (
    "age_missing",
    "age_implausible",
    "under_18",
    "sex_uncoded",
    "file_missing",
    "duplicate_file",
    "filename_mismatch",
    "file_not_found",
    "ungradable",
)
REASON_DTYPE = pl.Enum(REASONS)


def normalised_sex() -> pl.Expr:
    return pl.col("sex").str.strip_chars().str.to_lowercase()


def metadata_exclusions(
    eyes: pl.DataFrame,
    categories: pl.DataFrame,
    image_files: set[str] | None = None,
) -> pl.DataFrame:
    """One row per (patient_id, eye, reason); an eye can fail several checks. Patient-level
    reasons (age, sex) are recorded on both eyes. `image_files=None` skips the existence
    check, so the audit can run from the annotation sheet alone."""
    expected_stem = pl.concat_str(pl.col("patient_id").cast(pl.String), pl.lit("_"), pl.col("eye"))
    stem = pl.col("file").str.strip_chars().str.to_lowercase()
    checks = {
        "age_missing": pl.col("age").is_null(),
        "age_implausible": (pl.col("age") < 0) | (pl.col("age") > MAX_PLAUSIBLE_AGE),
        "under_18": pl.col("age").is_between(0, MIN_AGE, closed="left"),
        "sex_uncoded": ~normalised_sex().is_in(SEXES).fill_null(False),
        "file_missing": stem.is_null() | (stem == ""),
        "duplicate_file": stem.is_not_null() & stem.is_duplicated(),
        "filename_mismatch": stem.is_not_null()
        & (stem != "")
        & (
            ~stem.str.contains(IMAGE_EXTENSIONS)
            | (stem.str.replace(IMAGE_EXTENSIONS, "") != expected_stem)
        ),
    }
    if image_files is not None:
        checks["file_not_found"] = pl.col("file").is_not_null() & ~pl.col("file").is_in(
            sorted(image_files)
        )

    flagged = eyes.select(*KEY, *[expr.alias(reason) for reason, expr in checks.items()])
    flagged = flagged.join(
        categories.select(*KEY, pl.col("cat_ungradable").alias("ungradable")), on=KEY, how="left"
    ).with_columns(pl.col("ungradable").fill_null(False))

    return (
        flagged.unpivot(index=list(KEY), variable_name="reason", value_name="failed")
        .filter(pl.col("failed"))
        .select(*KEY, pl.col("reason").cast(REASON_DTYPE))
        .sort(*KEY, "reason")
    )


def exclusion_flow(eyes: pl.DataFrame, exclusions: pl.DataFrame) -> pl.DataFrame:
    """P7 flow diagram: reasons applied in `REASONS` order, each eye counted under its first
    reason only."""
    first = exclusions.group_by(KEY).agg(pl.col("reason").min())
    remaining = eyes.select(KEY)
    rows = [
        {
            "step": 0,
            "reason": None,
            "excluded_eyes": 0,
            "remaining_eyes": remaining.height,
            "remaining_patients": remaining["patient_id"].n_unique(),
        }
    ]
    for step, reason in enumerate(REASONS, start=1):
        dropped = first.filter(pl.col("reason") == reason).select(KEY)
        remaining = remaining.join(dropped, on=KEY, how="anti")
        rows.append(
            {
                "step": step,
                "reason": reason,
                "excluded_eyes": dropped.height,
                "remaining_eyes": remaining.height,
                "remaining_patients": remaining["patient_id"].n_unique(),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "step": pl.Int64,
            "reason": REASON_DTYPE,
            "excluded_eyes": pl.Int64,
            "remaining_eyes": pl.Int64,
            "remaining_patients": pl.Int64,
        },
    )


def included(eyes: pl.DataFrame, exclusions: pl.DataFrame) -> pl.DataFrame:
    return eyes.join(exclusions.select(KEY).unique(), on=KEY, how="anti")
