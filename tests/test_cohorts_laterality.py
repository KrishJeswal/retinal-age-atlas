import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.audit.cohorts import healthy_pools
from rra.audit.keywords import eye_categories
from rra.audit.laterality import LATERALITY, laterality_table, tier_counts
from rra.data.odir import LABELS, parse_patients, to_eyes
from synthetic import make_raw_sheet


def build(pairs: list[tuple[str, str]], n_label: list[int] | None = None) -> pl.DataFrame:
    rows = [
        {
            "id": i,
            "age": 60,
            "sex": "Male",
            "left_kw": left,
            "right_kw": right,
            "labels": {**dict.fromkeys(LABELS, 0), "N": (n_label or [1] * len(pairs))[i]},
        }
        for i, (left, right) in enumerate(pairs)
    ]
    return to_eyes(parse_patients(make_raw_sheet(rows)))


def pools_of(eyes: pl.DataFrame, mapping: pl.DataFrame) -> list[tuple[bool, bool, bool]]:
    pools = healthy_pools(eyes, eye_categories(eyes, mapping))
    return pools.select("pool_strict", "pool_main", "pool_low_quality").rows()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("normal fundus", "normal fundus", (True, True, True)),
        ("normal fundus，lens dust", "normal fundus", (False, True, True)),
        ("low image quality", "normal fundus", (False, False, True)),
        ("normal fundus，cataract", "normal fundus", (False, False, False)),
        ("", "normal fundus", (False, False, False)),
    ],
)
def test_pool_definitions(mapping, left, right, expected):
    assert pools_of(build([(left, right)]), mapping) == [expected]


def test_pools_require_the_n_label(mapping):
    eyes = build([("normal fundus", "normal fundus")], n_label=[0])
    assert pools_of(eyes, mapping) == [(False, False, False)]


def test_pools_require_both_eyes(mapping):
    eyes = build([("normal fundus", "normal fundus")])
    assert pools_of(eyes.filter(pl.col("eye") == "left"), mapping) == [(False, False, False)]


eye_keywords = st.lists(
    st.sampled_from(
        ["normal fundus", "lens dust", "low image quality", "cataract", "diabetic retinopathy"]
    ),
    max_size=3,
).map("，".join)


@given(st.lists(st.tuples(eye_keywords, eye_keywords), min_size=1, max_size=12))
def test_pools_are_nested(pairs):
    mapping = pl.DataFrame(
        [
            ("normal fundus", "normal", None, None),
            ("lens dust", "artefact", None, None),
            ("low image quality", "quality", None, None),
            ("cataract", "cataract", None, None),
            ("diabetic retinopathy", "dr", "unspecified", None),
        ],
        schema={c: pl.String for c in ("term", "category", "severity", "note")},
        orient="row",
    )
    for strict, main, low_quality in pools_of(build(pairs), mapping):
        assert not strict or main
        assert not main or low_quality


def lateral(eyes: pl.DataFrame, mapping: pl.DataFrame, category: str) -> dict[str, int]:
    table = laterality_table(eye_categories(eyes, mapping))
    return table.filter(pl.col("category") == category).drop("category").row(0, named=True)


def test_laterality_table(mapping):
    eyes = build(
        [
            ("cataract", "normal fundus"),
            ("normal fundus", "cataract"),
            ("cataract", "cataract"),
            ("normal fundus", "normal fundus"),
        ]
    )
    assert lateral(eyes, mapping, "cataract") == {
        "none": 1,
        "left_only": 1,
        "right_only": 1,
        "bilateral": 1,
        "single_eye": 0,
    }


def test_laterality_marks_patients_with_one_eye_left(mapping):
    eyes = build([("cataract", "normal fundus"), ("normal fundus", "normal fundus")])
    remaining = eyes.filter(~((pl.col("patient_id") == 0) & (pl.col("eye") == "right")))
    counts = lateral(remaining, mapping, "cataract")
    assert counts["single_eye"] == 1
    assert counts["left_only"] == 0


@given(st.lists(st.tuples(eye_keywords, eye_keywords), min_size=1, max_size=12))
def test_laterality_partitions_patients_and_mirrors_on_swap(pairs):
    mapping = pl.DataFrame(
        [
            ("normal fundus", "normal", None, None),
            ("lens dust", "artefact", None, None),
            ("low image quality", "quality", None, None),
            ("cataract", "cataract", None, None),
            ("diabetic retinopathy", "dr", "unspecified", None),
        ],
        schema={c: pl.String for c in ("term", "category", "severity", "note")},
        orient="row",
    )
    table = laterality_table(eye_categories(build(pairs), mapping))
    swapped = laterality_table(eye_categories(build([(r, lft) for lft, r in pairs]), mapping))

    sides = list(LATERALITY.categories)
    assert (table.select(pl.sum_horizontal(sides)).to_series() == len(pairs)).all()
    assert table["left_only"].equals(swapped["right_only"])
    assert table["bilateral"].equals(swapped["bilateral"])


def test_tier_counts(mapping):
    eyes = build(
        [
            ("mild nonproliferative retinopathy", "normal fundus"),
            ("moderate non proliferative retinopathy", "lens dust"),
            ("diabetic retinopathy", "cataract"),
            ("diabetic retinopathy", "diabetic retinopathy"),
            ("glaucoma", "normal fundus"),
            ("glaucoma", "suspected glaucoma"),
            ("glaucoma", "low image quality"),
        ]
    )
    counts = dict(tier_counts(eye_categories(eyes, mapping)).select("count", "n").rows())
    assert counts == {
        "tier_a_dr_eyes": 5,
        "tier_b_mild_npdr_eyes": 1,
        "tier_c_cases": 2,
        "h2_g_unilateral": 2,
        "h2_g_unilateral_benign_fellow": 1,
        "h2_d_unilateral": 3,
        "h2_d_unilateral_benign_fellow": 2,
    }


def test_paired_counts_need_both_eyes(mapping):
    eyes = build([("diabetic retinopathy", "normal fundus")])
    categories = eye_categories(eyes, mapping).filter(
        ~((pl.col("patient_id") == 0) & (pl.col("eye") == "right"))
    )
    counts = dict(tier_counts(categories).select("count", "n").rows())
    assert counts["tier_a_dr_eyes"] == 1
    assert counts["tier_c_cases"] == 0
    assert counts["h2_d_unilateral"] == 0
