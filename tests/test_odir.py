import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.data.odir import LABELS, SchemaError, parse_patients, read_annotations, to_eyes
from synthetic import make_raw_sheet

keywords = st.text(alphabet="abcdefgh ，,", max_size=30)

patient_rows = st.lists(
    st.fixed_dictionaries(
        {
            "id": st.integers(0, 10_000),
            "age": st.integers(0, 110),
            "sex": st.sampled_from(["Male", "Female"]),
            "left_kw": keywords,
            "right_kw": keywords,
            "labels": st.fixed_dictionaries({label: st.integers(0, 1) for label in LABELS}),
        }
    ),
    min_size=1,
    max_size=20,
    unique_by=lambda r: r["id"],
)


def test_parse_renames_and_types(raw_sheet):
    patients = parse_patients(raw_sheet)
    assert patients.schema["patient_id"] == pl.Int64
    assert patients.schema["age"] == pl.Int64
    assert all(patients.schema[label] == pl.Int8 for label in LABELS)
    row = patients.filter(pl.col("patient_id") == 2).row(0, named=True)
    assert row["left_keywords"] == "laser spot，moderate non proliferative retinopathy"
    assert row["D"] == 1


def test_parse_ignores_extra_columns(raw_sheet):
    patients = parse_patients(raw_sheet.with_columns(pl.lit("x").alias("Unnamed: 15")))
    assert "Unnamed: 15" not in patients.columns


def test_parse_rejects_missing_column(raw_sheet):
    with pytest.raises(SchemaError, match="Left-Diagnostic Keywords"):
        parse_patients(raw_sheet.drop("Left-Diagnostic Keywords"))


def test_parse_rejects_duplicate_ids(raw_sheet):
    with pytest.raises(SchemaError, match="duplicate"):
        parse_patients(pl.concat([raw_sheet, raw_sheet.head(1)]))


@pytest.mark.parametrize("value", [2, None])
def test_parse_rejects_bad_label(raw_sheet, value):
    sheet = raw_sheet.with_columns(
        pl.when(pl.col("ID") == 1).then(value).otherwise(pl.col("G")).alias("G")
    )
    with pytest.raises(SchemaError, match="patients: \\[1\\]"):
        parse_patients(sheet)


@given(patient_rows)
def test_to_eyes_maps_each_side_to_its_own_eye(rows):
    eyes = to_eyes(parse_patients(make_raw_sheet(rows)))

    assert eyes.height == 2 * len(rows)
    for r in rows:
        pair = eyes.filter(pl.col("patient_id") == r["id"])
        assert pair["eye"].to_list() == ["left", "right"]
        assert pair["file"].to_list() == [f"{r['id']}_left.jpg", f"{r['id']}_right.jpg"]
        assert pair["keywords"].to_list() == [r["left_kw"], r["right_kw"]]
        assert pair["age"].to_list() == [r["age"]] * 2
        assert pair["sex"].to_list() == [r["sex"]] * 2
        for label in LABELS:
            assert pair[label].to_list() == [r["labels"][label]] * 2


def test_read_annotations_from_xlsx(raw_sheet, tmp_path):
    path = tmp_path / "annotations.xlsx"
    raw_sheet.write_excel(path)
    assert read_annotations(path).equals(parse_patients(raw_sheet))
