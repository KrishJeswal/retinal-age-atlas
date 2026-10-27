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
