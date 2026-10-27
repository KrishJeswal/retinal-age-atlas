import polars as pl
import pytest

from rra.data import manifest
from rra.data.odir import SchemaError, parse_patients, to_eyes


@pytest.fixture
def eyes(raw_sheet) -> pl.DataFrame:
    return to_eyes(parse_patients(raw_sheet))


def test_validate_accepts_odir_eyes_with_derived_columns(eyes):
    manifest.validate(eyes.with_columns(pl.lit(0.5).alias("quality")))


def test_validate_rejects_missing_column(eyes):
    with pytest.raises(SchemaError, match="keywords"):
        manifest.validate(eyes.drop("keywords"))


def test_validate_rejects_wrong_dtype(eyes):
    with pytest.raises(SchemaError, match="age"):
        manifest.validate(eyes.with_columns(pl.col("age").cast(pl.Float64)))


def test_validate_rejects_duplicate_key(eyes):
    with pytest.raises(SchemaError, match="duplicate"):
        manifest.validate(pl.concat([eyes, eyes.head(1)]))


def test_write_read_roundtrip(eyes, tmp_path):
    path = tmp_path / "derived" / "manifest.parquet"
    digest = manifest.write(eyes, path)
    loaded = manifest.read(path)
    assert loaded.equals(eyes)
    assert digest == manifest.content_hash(loaded)


def test_content_hash_ignores_row_and_column_order(eyes):
    shuffled = eyes.sample(fraction=1.0, shuffle=True, seed=0).select(reversed(eyes.columns))
    assert manifest.content_hash(shuffled) == manifest.content_hash(eyes)


def test_content_hash_changes_with_content(eyes):
    edited = eyes.with_columns(
        pl.when(pl.col("patient_id") == 1).then(58).otherwise(pl.col("age")).alias("age")
    )
    assert manifest.content_hash(edited) != manifest.content_hash(eyes)
