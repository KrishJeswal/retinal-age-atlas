import polars as pl
import pytest

from rra.data.odir import LABELS
from synthetic import make_raw_sheet


@pytest.fixture
def raw_sheet() -> pl.DataFrame:
    def labels(**on: int) -> dict[str, int]:
        return {label: on.get(label, 0) for label in LABELS}

    return make_raw_sheet(
        [
            {
                "id": 0,
                "age": 69,
                "sex": "Female",
                "left_kw": "cataract",
                "right_kw": "normal fundus",
                "labels": labels(C=1),
            },
            {
                "id": 1,
                "age": 57,
                "sex": "Male",
                "left_kw": "normal fundus",
                "right_kw": "normal fundus",
                "labels": labels(N=1),
            },
            {
                "id": 2,
                "age": 42,
                "sex": "Male",
                "left_kw": "laser spot，moderate non proliferative retinopathy",
                "right_kw": "moderate non proliferative retinopathy",
                "labels": labels(D=1),
            },
        ]
    )


@pytest.fixture
def mapping() -> pl.DataFrame:
    return pl.DataFrame(
        [
            ("normal fundus", "normal", None, None),
            ("lens dust", "artefact", None, None),
            ("low image quality", "quality", None, None),
            ("cataract", "cataract", None, None),
            ("laser spot", "other", None, "treatment scar"),
            ("diabetic retinopathy", "dr", "unspecified", None),
            ("mild nonproliferative retinopathy", "dr", "mild_npdr", None),
            ("moderate non proliferative retinopathy", "dr", "moderate_npdr", None),
            ("proliferative diabetic retinopathy", "dr", "pdr", None),
            ("suspected glaucoma", "glaucoma", "suspected", None),
            ("glaucoma", "glaucoma", "confirmed", None),
            ("hypertensive retinopathy", "hr", None, None),
        ],
        schema={c: pl.String for c in ("term", "category", "severity", "note")},
        orient="row",
    )
