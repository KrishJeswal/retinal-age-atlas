import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.audit.keywords import (
    CATEGORIES,
    agreement_table,
    eye_categories,
    keyword_labels,
    read_mapping,
    validate_mapping,
)
from rra.data.odir import LABELS, SchemaError, parse_patients, to_eyes
from synthetic import make_raw_sheet


def eyes_with(keywords: list[str | None]) -> pl.DataFrame:
    rows = [
        {
            "id": i,
            "age": 60,
            "sex": "Male",
            "left_kw": kw,
            "right_kw": "normal fundus",
            "labels": dict.fromkeys(LABELS, 0),
        }
        for i, kw in enumerate(keywords)
    ]
    return to_eyes(parse_patients(make_raw_sheet(rows)))


def left_eye(categories: pl.DataFrame, patient_id: int = 0) -> dict:
    return categories.filter((pl.col("patient_id") == patient_id) & (pl.col("eye") == "left")).row(
        0, named=True
    )


def test_valid_mapping_passes(mapping):
    validate_mapping(mapping)


def test_read_mapping_reads_empty_cells_as_null(mapping, tmp_path):
    path = tmp_path / "keyword_mapping.csv"
    mapping.write_csv(path)
    assert read_mapping(path).equals(mapping)


def test_missing_column_raises(mapping):
    with pytest.raises(SchemaError, match="note"):
        validate_mapping(mapping.drop("note"))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (("cataract", "cataract", None, None), "duplicate terms"),
        (("Lens  Dust", "artefact", None, None), "not in normalised form"),
        (("drusen，macular", "amd", None, None), "not in normalised form"),
        (("drusen", "macular", None, None), "unknown categories"),
        (("severe dr", "dr", None, None), "dr terms need severity"),
        (("severe dr", "dr", "severe", None), "dr terms need severity"),
        (("glaucoma?", "glaucoma", None, None), "glaucoma terms need severity"),
        (("drusen", "amd", "mild_npdr", None), "severity given for a category"),
        ((None, "amd", None, None), "empty term"),
        (("drusen", None, None, None), "empty category"),
    ],
)
def test_invalid_rows_raise(mapping, row, message):
    bad = pl.concat([mapping, pl.DataFrame([row], schema=mapping.schema, orient="row")])
    with pytest.raises(SchemaError, match=message):
        validate_mapping(bad)


def test_all_problems_reported_together(mapping):
    bad = pl.concat(
        [
            mapping,
            pl.DataFrame(
                [("cataract", "cataract", None, None), ("drusen", "macular", None, None)],
                schema=mapping.schema,
                orient="row",
            ),
        ]
    )
    with pytest.raises(SchemaError, match=r"(?s)duplicate terms.*unknown categories"):
        validate_mapping(bad)


def test_eye_categories_on_fixture(raw_sheet, mapping):
    categories = eye_categories(to_eyes(parse_patients(raw_sheet)), mapping)
    assert categories.height == 6
    dr_eye = left_eye(categories, patient_id=2)
    assert dr_eye["n_terms"] == 2
    assert {c for c in CATEGORIES if dr_eye[f"cat_{c}"]} == {"dr", "other"}
    assert dr_eye["dr_severity"] == "moderate_npdr"
    assert dr_eye["glaucoma_severity"] is None


@pytest.mark.parametrize(
    ("keywords", "dr", "glaucoma"),
    [
        ("mild nonproliferative retinopathy，proliferative diabetic retinopathy", "pdr", None),
        ("diabetic retinopathy，mild nonproliferative retinopathy", "mild_npdr", None),
        ("diabetic retinopathy", "unspecified", None),
        ("glaucoma，suspected glaucoma", None, "confirmed"),
    ],
)
def test_severity_is_the_highest_on_the_eye(mapping, keywords, dr, glaucoma):
    eye = left_eye(eye_categories(eyes_with([keywords]), mapping))
    assert (eye["dr_severity"], eye["glaucoma_severity"]) == (dr, glaucoma)


@pytest.mark.parametrize("keywords", ["", None])
def test_eye_without_terms(mapping, keywords):
    eye = left_eye(eye_categories(eyes_with([keywords]), mapping))
    assert eye["n_terms"] == 0
    assert not any(eye[f"cat_{c}"] for c in CATEGORIES)


def test_unmapped_term_raises(mapping):
    with pytest.raises(SchemaError, match="\\['drusen'\\]"):
        eye_categories(eyes_with(["drusen，cataract"]), mapping)


def test_keyword_labels_and_agreement(raw_sheet, mapping):
    patients = parse_patients(raw_sheet)
    implied = keyword_labels(eye_categories(to_eyes(patients), mapping))
    assert implied.filter(pl.col("patient_id") == 1)["kw_N"].item()
    assert implied.filter(pl.col("patient_id") == 2)["kw_O"].item()

    table = agreement_table(patients, implied).rows_by_key("label", named=True, unique=True)
    assert table["N"] == {"both": 1, "label_only": 0, "keyword_only": 0, "neither": 2}
    assert table["C"] == {"both": 1, "label_only": 0, "keyword_only": 0, "neither": 2}
    assert table["D"] == {"both": 1, "label_only": 0, "keyword_only": 0, "neither": 2}
    assert table["O"] == {"both": 0, "label_only": 0, "keyword_only": 1, "neither": 2}


@pytest.mark.parametrize(
    ("left", "kw_n"),
    [("lens dust", True), ("low image quality", False), ("", False)],
)
def test_keyword_normal_allows_artefacts_only(mapping, left, kw_n):
    implied = keyword_labels(eye_categories(eyes_with([left]), mapping))
    assert implied["kw_N"].item() is kw_n


mapped_terms = [
    "normal fundus",
    "lens dust",
    "low image quality",
    "cataract",
    "laser spot",
    "diabetic retinopathy",
    "glaucoma",
    "hypertensive retinopathy",
]
eye_keywords = st.lists(st.sampled_from(mapped_terms), max_size=3).map("，".join)


@given(st.lists(st.tuples(eye_keywords, eye_keywords), min_size=1, max_size=15))
def test_labels_built_from_keywords_agree_perfectly(pairs):
    mapping_ = pl.DataFrame(
        [
            ("normal fundus", "normal", None),
            ("lens dust", "artefact", None),
            ("low image quality", "quality", None),
            ("cataract", "cataract", None),
            ("laser spot", "other", None),
            ("diabetic retinopathy", "dr", "unspecified"),
            ("glaucoma", "glaucoma", "confirmed"),
            ("hypertensive retinopathy", "hr", None),
        ],
        schema={"term": pl.String, "category": pl.String, "severity": pl.String},
        orient="row",
    ).with_columns(pl.lit(None, dtype=pl.String).alias("note"))
    rows = [
        {
            "id": i,
            "age": 60,
            "sex": "Male",
            "left_kw": left,
            "right_kw": right,
            "labels": dict.fromkeys(LABELS, 0),
        }
        for i, (left, right) in enumerate(pairs)
    ]
    patients = parse_patients(make_raw_sheet(rows))
    implied = keyword_labels(eye_categories(to_eyes(patients), mapping_))
    planted = patients.drop(LABELS).join(
        implied.select(
            "patient_id", *[pl.col(f"kw_{lab}").cast(pl.Int8).alias(lab) for lab in LABELS]
        ),
        on="patient_id",
    )

    table = agreement_table(planted, implied)
    assert (table["label_only"] == 0).all()
    assert (table["keyword_only"] == 0).all()
    assert ((table["both"] + table["neither"]) == len(pairs)).all()
