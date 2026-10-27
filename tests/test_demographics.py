import math

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.audit.cohorts import healthy_pools
from rra.audit.demographics import (
    age_bins,
    demographics_table,
    group_membership,
    matching_candidates,
    matching_feasibility,
    patient_table,
)
from rra.audit.keywords import eye_categories
from rra.audit.laterality import unilateral_patients
from rra.data.odir import LABELS, parse_patients, to_eyes
from synthetic import make_raw_sheet


def people(rows: list[tuple[int, int, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"patient_id": pl.Int64, "age": pl.Int64, "sex": pl.String}, orient="row"
    )


def test_patient_table_has_one_row_per_patient_and_normalised_sex(raw_sheet):
    eyes = to_eyes(parse_patients(raw_sheet.with_columns(pl.lit(" FEMALE").alias("Patient Sex"))))
    patients = patient_table(eyes)
    assert patients["patient_id"].to_list() == [0, 1, 2]
    assert set(patients["sex"]) == {"female"}
    assert patients.filter(pl.col("patient_id") == 2)["D"].item() == 1


def test_group_membership_combines_pools_and_labels(raw_sheet, mapping):
    eyes = to_eyes(parse_patients(raw_sheet))
    membership = group_membership(
        patient_table(eyes), healthy_pools(eyes, eye_categories(eyes, mapping))
    )
    groups = membership.group_by("group").agg(pl.col("patient_id").sort()).sort("group")
    assert dict(groups.rows()) == {
        "C": [0],
        "D": [2],
        "N": [1],
        "healthy_low_quality": [1],
        "healthy_main": [1],
        "healthy_strict": [1],
    }


def test_demographics_table_matches_hand_computation():
    patients = people([(0, 50, "female"), (1, 60, "male"), (2, 70, "female"), (3, 80, "female")])
    membership = pl.DataFrame(
        {"patient_id": [0, 1, 2, 3], "group": ["healthy_main"] * 2 + ["D"] * 2}
    )
    table = demographics_table(patients, membership).rows_by_key("group", named=True, unique=True)

    ref, d = table["healthy_main"], table["D"]
    assert (ref["n"], ref["age_mean"], ref["p_female"]) == (2, 55.0, 0.5)
    assert ref["age_smd"] == 0
    assert ref["female_smd"] == 0
    sd = math.sqrt(50)
    assert d["age_smd"] == pytest.approx((75 - 55) / math.sqrt((sd**2 + sd**2) / 2))
    assert d["female_smd"] == pytest.approx((1.0 - 0.5) / math.sqrt((0 + 0.25) / 2))


def test_demographics_without_reference_has_null_smd():
    patients = people([(0, 50, "female"), (1, 60, "male")])
    membership = pl.DataFrame({"patient_id": [0, 1], "group": ["D", "D"]})
    table = demographics_table(patients, membership)
    assert table["age_smd"].is_null().all()


def test_age_bins():
    patients = people([(0, 54, "male"), (1, 55, "male"), (2, 59, "male"), (3, 61, "male")])
    membership = pl.DataFrame({"patient_id": [0, 1, 2, 3], "group": ["N"] * 4})
    assert age_bins(patients, membership).select("bin_start", "n").rows() == [
        (50, 1),
        (55, 2),
        (60, 1),
    ]


def test_matching_candidates_caliper_is_inclusive_and_sex_exact():
    cases = people([(100, 60, "male"), (101, 30, "female")])
    controls = people([(0, 58, "male"), (1, 62, "male"), (2, 63, "male"), (3, 60, "female")])
    counts = matching_candidates(cases, controls, caliper=2, exact=("sex",))
    assert counts.rows() == [(100, 2), (101, 0)]


ages = st.integers(18, 90)
sexes = st.sampled_from(["male", "female"])


@given(
    st.lists(st.tuples(ages, sexes), min_size=1, max_size=10),
    st.lists(st.tuples(ages, sexes), max_size=30),
    st.integers(0, 5),
)
def test_matching_candidates_agree_with_brute_force(case_rows, control_rows, caliper):
    cases = people([(100 + i, a, s) for i, (a, s) in enumerate(case_rows)])
    controls = people([(i, a, s) for i, (a, s) in enumerate(control_rows)])
    counts = matching_candidates(cases, controls, caliper=caliper, exact=("sex",))["n_candidates"]
    expected = [
        sum(1 for ca, cs in control_rows if cs == s and abs(ca - a) <= caliper)
        for a, s in case_rows
    ]
    assert counts.to_list() == expected


def test_matching_feasibility_for_tier_c(mapping):
    rows = [
        {
            "id": i,
            "age": age,
            "sex": "Male",
            "left_kw": left,
            "right_kw": "normal fundus",
            "labels": {**dict.fromkeys(LABELS, 0), "N": int(left == "normal fundus")},
        }
        for i, (age, left) in enumerate(
            [(60, "diabetic retinopathy"), (40, "diabetic retinopathy")]
            + [(a, "normal fundus") for a in (59, 60, 61, 62, 75)]
        )
    ]
    eyes = to_eyes(parse_patients(make_raw_sheet(rows)))
    categories = eye_categories(eyes, mapping)
    patients = patient_table(eyes)
    pools = healthy_pools(eyes, categories)

    tier_c = unilateral_patients(categories, "dr", benign_fellow=True)
    cases = patients.join(tier_c, on="patient_id")
    controls = patients.join(pools.filter(pl.col("pool_main")), on="patient_id")
    summary = matching_feasibility(cases, controls).row(0, named=True)
    assert summary == {
        "n_cases": 2,
        "n_controls": 5,
        "candidates_min": 0,
        "candidates_median": 2.0,
        "cases_with_ratio": 1,
        "controls_needed": 6,
    }
