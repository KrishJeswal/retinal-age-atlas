from pathlib import Path

import polars as pl

from rra.data.manifest import KEY
from rra.data.odir import LABELS, SchemaError

CATEGORIES = (
    "normal",
    "artefact",
    "quality",
    "dr",
    "hr",
    "glaucoma",
    "cataract",
    "lens",
    "amd",
    "myopia",
    "other",
)
BENIGN = ("normal", "artefact")
DR_SEVERITY = pl.Enum(["unspecified", "mild_npdr", "moderate_npdr", "severe_npdr", "pdr"])
GLAUCOMA_STATUS = pl.Enum(["suspected", "confirmed"])
SEVERITIES = {"dr": DR_SEVERITY, "glaucoma": GLAUCOMA_STATUS}
MAPPING_COLUMNS = ("term", "category", "severity", "note")

LABEL_CATEGORY = {
    "D": "dr",
    "G": "glaucoma",
    "C": "cataract",
    "A": "amd",
    "H": "hr",
    "M": "myopia",
    "O": "other",
}


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


def read_mapping(path: Path) -> pl.DataFrame:
    mapping = pl.read_csv(path, infer_schema=False)
    validate_mapping(mapping)
    return mapping


def validate_mapping(mapping: pl.DataFrame) -> None:
    """Collects every problem before raising, so one run lists all fixes needed in the CSV."""
    missing = sorted(set(MAPPING_COLUMNS) - set(mapping.columns))
    if missing:
        raise SchemaError(f"mapping is missing columns: {missing}")

    problems = []
    for column in ("term", "category"):
        if mapping[column].null_count():
            problems.append(f"empty {column} in {mapping[column].null_count()} rows")

    terms = mapping.drop_nulls("term")
    duplicated = terms.filter(pl.col("term").is_duplicated())["term"].unique().sort()
    if len(duplicated):
        problems.append(f"duplicate terms: {duplicated.to_list()}")

    not_normalised = terms.filter(
        normalised_terms(pl.col("term")).list.join(",") != pl.col("term")
    )["term"]
    if len(not_normalised):
        problems.append(f"terms not in normalised form: {not_normalised.to_list()}")

    unknown = mapping.filter(~pl.col("category").is_in(CATEGORIES))["category"].unique().sort()
    if len(unknown.drop_nulls()):
        problems.append(f"unknown categories: {unknown.drop_nulls().to_list()}")

    for category, dtype in SEVERITIES.items():
        bad = mapping.filter(
            (pl.col("category") == category)
            & ~pl.col("severity").is_in(dtype.categories.to_list()).fill_null(False)
        )["term"]
        if len(bad):
            problems.append(
                f"{category} terms need severity in {dtype.categories.to_list()}: {bad.to_list()}"
            )
    stray = mapping.filter(
        ~pl.col("category").is_in(list(SEVERITIES)) & pl.col("severity").is_not_null()
    )["term"]
    if len(stray):
        problems.append(f"severity given for a category without severities: {stray.to_list()}")

    if problems:
        raise SchemaError("invalid keyword mapping:\n" + "\n".join(problems))


def eye_categories(eyes: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Per eye: n_terms, cat_<category> flags, the highest DR severity and glaucoma status.
    Every term must be mapped (D2 covers the full vocabulary)."""
    validate_mapping(mapping)
    terms = eye_terms(eyes)

    unmapped = terms.join(mapping, on="term", how="anti")["term"].unique().sort()
    if len(unmapped):
        raise SchemaError(f"terms missing from the mapping: {unmapped.to_list()}")

    per_eye = (
        terms.join(mapping.select("term", "category", "severity"), on="term")
        .group_by(KEY)
        .agg(
            pl.len().cast(pl.Int64).alias("n_terms"),
            *[(pl.col("category") == c).any().alias(f"cat_{c}") for c in CATEGORIES],
            *[
                pl.col("severity")
                .filter(pl.col("category") == category)
                .cast(dtype)
                .max()
                .alias(f"{category}_severity")
                for category, dtype in SEVERITIES.items()
            ],
        )
    )
    return (
        eyes.select(KEY)
        .join(per_eye, on=KEY, how="left")
        .with_columns(
            pl.col("n_terms").fill_null(0),
            *[pl.col(f"cat_{c}").fill_null(False) for c in CATEGORIES],
        )
        .sort(KEY)
    )


def benign_eye() -> pl.Expr:
    """Eye has at least one term and every term is normal or artefact."""
    return (pl.col("n_terms") > 0) & ~pl.any_horizontal(
        pl.col(f"cat_{c}") for c in CATEGORIES if c not in BENIGN
    )


def keyword_labels(categories: pl.DataFrame) -> pl.DataFrame:
    """Patient labels implied by the keywords: N if both eyes are benign, otherwise the label
    whose category appears in either eye. Lens, quality and artefact imply no label."""
    return (
        categories.group_by("patient_id")
        .agg(
            benign_eye().all().alias("kw_N"),
            *[pl.col(f"cat_{c}").any().alias(f"kw_{label}") for label, c in LABEL_CATEGORY.items()],
        )
        .sort("patient_id")
    )


def agreement_table(patients: pl.DataFrame, implied: pl.DataFrame) -> pl.DataFrame:
    """Per label: patients where the readers' label and the keyword-implied label agree or not."""
    joined = patients.select("patient_id", *LABELS).join(implied, on="patient_id")
    rows = []
    for label in LABELS:
        given, kw = pl.col(label) == 1, pl.col(f"kw_{label}")
        counts = joined.select(
            (given & kw).sum().alias("both"),
            (given & ~kw).sum().alias("label_only"),
            (~given & kw).sum().alias("keyword_only"),
            (~given & ~kw).sum().alias("neither"),
        )
        rows.append(counts.with_columns(pl.lit(label).alias("label")))
    return pl.concat(rows).select("label", pl.exclude("label").cast(pl.Int64))
