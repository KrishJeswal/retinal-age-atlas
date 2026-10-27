import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from rra.audit.keywords import eye_terms, vocabulary
from rra.data.odir import LABELS, parse_patients, to_eyes
from synthetic import make_raw_sheet


def terms_of(keywords: str | None) -> list[str]:
    eyes = pl.DataFrame(
        {"patient_id": [0], "eye": ["left"], "keywords": [keywords]},
        schema={"patient_id": pl.Int64, "eye": pl.Enum(["left", "right"]), "keywords": pl.String},
    )
    return eye_terms(eyes)["term"].to_list()


@pytest.mark.parametrize(
    ("keywords", "expected"),
    [
        ("normal fundus", ["normal fundus"]),
        (
            "laser spot，moderate non proliferative retinopathy",
            ["laser spot", "moderate non proliferative retinopathy"],
        ),
        ("drusen,  Glaucoma ", ["drusen", "glaucoma"]),
        ("\uff2c\uff45\uff4e\uff53\u3000dust", ["lens dust"]),
        ("myopia\tretinopathy", ["myopia retinopathy"]),
        ("cataract，，cataract，", ["cataract"]),
        ("", []),
        (None, []),
    ],
)
def test_terms_are_split_and_normalised(keywords, expected):
    assert terms_of(keywords) == expected


def test_vocabulary_counts(raw_sheet):
    vocab = vocabulary(to_eyes(parse_patients(raw_sheet)))
    normal = vocab.filter(pl.col("term") == "normal fundus").row(0, named=True)
    assert normal == {
        "term": "normal fundus",
        "n_eyes": 3,
        "n_patients": 2,
        "n_left": 1,
        "n_right": 2,
        **{f"with_{label}": {"N": 2, "C": 1}.get(label, 0) for label in LABELS},
    }
    assert vocab["term"].to_list()[0] == "normal fundus"


keyword_strings = st.builds(
    str.join,
    st.sampled_from(["，", ","]),
    st.lists(
        st.sampled_from(
            [
                "Normal fundus",
                "cataract",
                "drusen ",
                "lens  dust",
                "",
                "\uff2d\uff39\uff2f\uff30\uff29\uff21",
            ]
        ),
        max_size=4,
    ),
)

sheets = st.lists(
    st.fixed_dictionaries(
        {
            "id": st.integers(0, 500),
            "age": st.just(60),
            "sex": st.just("Male"),
            "left_kw": keyword_strings,
            "right_kw": keyword_strings,
            "labels": st.fixed_dictionaries({label: st.integers(0, 1) for label in LABELS}),
        }
    ),
    min_size=1,
    max_size=15,
    unique_by=lambda r: r["id"],
).map(lambda rows: to_eyes(parse_patients(make_raw_sheet(rows))))


@given(sheets)
def test_vocabulary_counts_are_consistent(eyes):
    vocab = vocabulary(eyes)
    assert vocab["n_eyes"].sum() == eye_terms(eyes).height
    assert (vocab["n_left"] + vocab["n_right"] == vocab["n_eyes"]).all()
    assert (vocab["n_patients"] <= vocab["n_eyes"]).all()
    assert (vocab["n_eyes"] <= 2 * vocab["n_patients"]).all()
    for label in LABELS:
        assert (vocab[f"with_{label}"] <= vocab["n_eyes"]).all()


@given(sheets)
def test_vocabulary_ignores_row_order(eyes):
    shuffled = eyes.sample(fraction=1.0, shuffle=True, seed=1)
    assert vocabulary(shuffled).equals(vocabulary(eyes))


@given(sheets)
def test_splitting_is_idempotent(eyes):
    terms = eye_terms(eyes)
    rejoined = eyes.drop("keywords").join(
        terms.group_by(["patient_id", "eye"], maintain_order=True).agg(
            pl.col("term").str.join("，").alias("keywords")
        ),
        on=["patient_id", "eye"],
        how="left",
    )
    assert eye_terms(rejoined).sort(pl.all()).equals(terms.sort(pl.all()))
