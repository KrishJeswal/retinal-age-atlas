import polars as pl
import pytest

from synthetic import THREE_PATIENTS, make_raw_sheet


@pytest.fixture
def raw_sheet() -> pl.DataFrame:
    return make_raw_sheet(THREE_PATIENTS)


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
