import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.audit.integrity import REASONS, exclusion_flow, included, metadata_exclusions
from rra.audit.keywords import eye_categories
from rra.data.manifest import KEY
from rra.data.odir import parse_patients, to_eyes
from synthetic import THREE_PATIENTS, make_raw_sheet


@pytest.fixture
def eyes(raw_sheet) -> pl.DataFrame:
    return to_eyes(parse_patients(raw_sheet))


def reasons_of(exclusions: pl.DataFrame, patient_id: int, eye: str) -> list[str]:
    return exclusions.filter((pl.col("patient_id") == patient_id) & (pl.col("eye") == eye))[
        "reason"
    ].to_list()


def edit(eyes: pl.DataFrame, patient_id: int, eye: str | None = None, **values) -> pl.DataFrame:
    target = pl.col("patient_id") == patient_id
    if eye is not None:
        target &= pl.col("eye") == eye
    return eyes.with_columns(
        pl.when(target).then(pl.lit(v)).otherwise(pl.col(k)).cast(eyes.schema[k]).alias(k)
        for k, v in values.items()
    )


def test_clean_fixture_has_no_exclusions(eyes, mapping):
    files = set(eyes["file"])
    assert metadata_exclusions(eyes, eye_categories(eyes, mapping), files).is_empty()


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ({"age": None}, "age_missing"),
        ({"age": 130}, "age_implausible"),
        ({"age": -1}, "age_implausible"),
        ({"age": 17}, "under_18"),
        ({"age": 0}, "under_18"),
        ({"sex": "U"}, "sex_uncoded"),
        ({"sex": None}, "sex_uncoded"),
    ],
)
def test_patient_level_reasons_apply_to_both_eyes(eyes, mapping, values, reason):
    edited = edit(eyes, 1, **values)
    exclusions = metadata_exclusions(edited, eye_categories(edited, mapping))
    assert reasons_of(exclusions, 1, "left") == [reason]
    assert reasons_of(exclusions, 1, "right") == [reason]
    assert exclusions.filter(pl.col("patient_id") != 1).is_empty()


@pytest.mark.parametrize("age", [18, 100])
def test_age_bounds_are_inclusive(eyes, mapping, age):
    edited = edit(eyes, 1, age=age)
    assert metadata_exclusions(edited, eye_categories(edited, mapping)).is_empty()


def test_sex_coding_ignores_case_and_whitespace(eyes, mapping):
    edited = edit(eyes, 1, sex=" FEMALE ")
    assert metadata_exclusions(edited, eye_categories(edited, mapping)).is_empty()


@pytest.mark.parametrize(
    ("file", "reason"),
    [
        (None, "file_missing"),
        ("", "file_missing"),
        ("1_right.jpg", "filename_mismatch"),
        ("2_left.jpg", "filename_mismatch"),
        ("1_left.tif", "filename_mismatch"),
    ],
)
def test_file_reasons_are_per_eye(eyes, mapping, file, reason):
    edited = edit(eyes, 1, "left", file=file)
    exclusions = metadata_exclusions(edited, eye_categories(edited, mapping))
    assert reason in reasons_of(exclusions, 1, "left")
    fellow = ["duplicate_file"] if file == "1_right.jpg" else []
    assert reasons_of(exclusions, 1, "right") == fellow


def test_filename_check_accepts_case_and_jpeg(eyes, mapping):
    edited = edit(eyes, 1, "left", file="1_LEFT.JPEG")
    assert metadata_exclusions(edited, eye_categories(edited, mapping)).is_empty()


def test_duplicate_file_flags_every_copy(eyes, mapping):
    edited = edit(eyes, 1, "left", file="0_left.jpg")
    exclusions = metadata_exclusions(edited, eye_categories(edited, mapping))
    assert "duplicate_file" in reasons_of(exclusions, 0, "left")
    assert "duplicate_file" in reasons_of(exclusions, 1, "left")


def test_file_not_found_only_when_files_given(eyes, mapping):
    categories = eye_categories(eyes, mapping)
    files = set(eyes["file"]) - {"2_right.jpg"}
    assert reasons_of(metadata_exclusions(eyes, categories, files), 2, "right") == [
        "file_not_found"
    ]
    assert metadata_exclusions(eyes, categories).is_empty()


def test_ungradable_keyword_excludes_the_eye(eyes, mapping):
    mapping = pl.concat(
        [
            mapping,
            pl.DataFrame(
                [("no fundus image", "ungradable", None, None)],
                schema=mapping.schema,
                orient="row",
            ),
        ]
    )
    edited = edit(eyes, 0, "right", keywords="no fundus image")
    exclusions = metadata_exclusions(edited, eye_categories(edited, mapping))
    assert exclusions.select(*KEY, pl.col("reason").cast(pl.String)).rows() == [
        (0, "right", "ungradable")
    ]


def test_flow_counts_each_eye_under_its_first_reason(eyes, mapping):
    edited = edit(edit(eyes, 1, age=12), 1, "left", file="1_right.jpg")
    exclusions = metadata_exclusions(edited, eye_categories(edited, mapping))
    flow = exclusion_flow(edited, exclusions)

    by_reason = dict(zip(flow["reason"].to_list(), flow["excluded_eyes"].to_list(), strict=True))
    assert by_reason["under_18"] == 2
    assert by_reason["filename_mismatch"] == 0
    assert flow["remaining_eyes"].to_list()[-1] == 4
    assert flow["remaining_patients"].to_list()[-1] == 2
    assert included(edited, exclusions).height == 4


@given(
    st.lists(
        st.tuples(st.integers(0, 2), st.sampled_from(["left", "right"]), st.sampled_from(REASONS)),
        max_size=12,
    )
)
def test_flow_is_monotone_and_ends_at_the_included_set(picks):
    eyes = to_eyes(parse_patients(make_raw_sheet(THREE_PATIENTS)))
    exclusions = pl.DataFrame(
        picks,
        schema={"patient_id": pl.Int64, "eye": eyes.schema["eye"], "reason": pl.String},
        orient="row",
    ).with_columns(pl.col("reason").cast(pl.Enum(REASONS)))
    flow = exclusion_flow(eyes, exclusions)

    remaining = flow["remaining_eyes"].to_list()
    assert remaining == sorted(remaining, reverse=True)
    assert flow["excluded_eyes"].sum() == exclusions.select(KEY).unique().height
    assert remaining[-1] == included(eyes, exclusions).height
