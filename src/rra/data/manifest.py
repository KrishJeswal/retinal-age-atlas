import hashlib
from pathlib import Path

import polars as pl

from rra.data.odir import EYE_DTYPE, LABELS, SchemaError

KEY = ("patient_id", "eye")

BASE_SCHEMA: dict[str, pl.DataType] = {
    "patient_id": pl.Int64(),
    "eye": EYE_DTYPE,
    "file": pl.String(),
    "age": pl.Int64(),
    "sex": pl.String(),
    "keywords": pl.String(),
    **{label: pl.Int8() for label in LABELS},
}


def validate(manifest: pl.DataFrame) -> None:
    """Base columns must be present with their dtypes; derived columns may be added freely."""
    missing = sorted(set(BASE_SCHEMA) - set(manifest.columns))
    if missing:
        raise SchemaError(f"manifest is missing columns: {missing}")

    wrong = {
        name: (str(manifest.schema[name]), str(dtype))
        for name, dtype in BASE_SCHEMA.items()
        if manifest.schema[name] != dtype
    }
    if wrong:
        raise SchemaError(f"manifest columns with wrong dtype (got, expected): {wrong}")

    if manifest.select(pl.struct(KEY).is_duplicated().any()).item():
        raise SchemaError("manifest has duplicate (patient_id, eye) rows")


def content_hash(manifest: pl.DataFrame) -> str:
    """SHA-256 of the canonical CSV (rows sorted by key, columns by name). Independent of
    parquet encoding, but only stable within one Polars version's float formatting."""
    canonical = manifest.select(sorted(manifest.columns)).sort(KEY)
    return hashlib.sha256(canonical.write_csv().encode()).hexdigest()


def write(manifest: pl.DataFrame, path: Path) -> str:
    validate(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(path)
    return content_hash(manifest)


def read(path: Path) -> pl.DataFrame:
    manifest = pl.read_parquet(path)
    validate(manifest)
    return manifest
