from pathlib import Path

import polars as pl

LABELS = ("N", "D", "G", "C", "A", "H", "M", "O")
EYES = ("left", "right")
EYE_DTYPE = pl.Enum(EYES)

SHEET_COLUMNS = {
    "ID": "patient_id",
    "Patient Age": "age",
    "Patient Sex": "sex",
    "Left-Fundus": "left_file",
    "Right-Fundus": "right_file",
    "Left-Diagnostic Keywords": "left_keywords",
    "Right-Diagnostic Keywords": "right_keywords",
    **{label: label for label in LABELS},
}


class SchemaError(ValueError):
    pass


def parse_patients(raw: pl.DataFrame) -> pl.DataFrame:
    """Rename and type the annotation sheet. Values are not cleaned: data-quality problems
    (implausible ages, odd sex codes, misnamed files) are the integrity audit's job."""
    missing = sorted(set(SHEET_COLUMNS) - set(raw.columns))
    if missing:
        raise SchemaError(f"annotation sheet is missing columns: {missing}")

    patients = (
        raw.select(list(SHEET_COLUMNS))
        .rename(SHEET_COLUMNS)
        .with_columns(
            pl.col("patient_id", "age").cast(pl.Int64),
            pl.col("sex", "left_file", "right_file", "left_keywords", "right_keywords").cast(
                pl.String
            ),
            pl.col(LABELS).cast(pl.Int8),
        )
    )

    duplicated = patients.filter(pl.col("patient_id").is_duplicated())["patient_id"].unique()
    if len(duplicated):
        raise SchemaError(f"duplicate patient IDs: {sorted(duplicated)}")

    bad_labels = patients.filter(~pl.all_horizontal(pl.col(LABELS).is_in([0, 1]).fill_null(False)))[
        "patient_id"
    ]
    if len(bad_labels):
        raise SchemaError(f"labels not in {{0, 1}} for patients: {sorted(bad_labels)}")

    return patients


def read_annotations(path: Path) -> pl.DataFrame:
    return parse_patients(pl.read_excel(path))


def to_eyes(patients: pl.DataFrame) -> pl.DataFrame:
    """One row per eye. Patient-level fields (age, sex, labels) are repeated on both eyes."""
    return pl.concat(
        [
            patients.select(
                "patient_id",
                pl.lit(eye, dtype=EYE_DTYPE).alias("eye"),
                pl.col(f"{eye}_file").alias("file"),
                "age",
                "sex",
                pl.col(f"{eye}_keywords").alias("keywords"),
                *LABELS,
            )
            for eye in EYES
        ]
    ).sort("patient_id", "eye")
