import numpy as np
import polars as pl
import pytest
from PIL import Image

from rra.audit.images import (
    FINE_SIZE,
    fine_mirrored,
    fine_signature,
    residual_unit,
    shared_basis,
    standardise,
    thumbnail,
    verify_pairs,
)
from rra.data.fov import fov_geometry
from synthetic import make_fundus
from test_images import recompressed

DUST = ((0.3, 0.2), (-0.4, -0.1), (0.1, -0.5), (0.55, 0.3), (-0.2, 0.45))


def standardised(rgb: np.ndarray):
    fov = fov_geometry(rgb)
    return standardise(Image.fromarray(rgb), fov.cx, fov.cy, fov.r)


def fine(rgb: np.ndarray) -> np.ndarray:
    return fine_signature(*standardised(rgb))[0]


def featureless(seed: int) -> np.ndarray:
    """Hazy image from a camera with dust on its optics: no anatomy survives, the dust does."""
    return make_fundus(seed=seed, disc=None, blur=14, dust=DUST)


@pytest.fixture(scope="module")
def basis() -> np.ndarray:
    reference = [fine(featureless(100 + i)) for i in range(12)]
    reference += [fine(make_fundus(seed=200 + i, disc=(0.5 * (-1) ** i, 0.0))) for i in range(12)]
    return shared_basis(np.stack(reference), components=8)


def test_shared_basis_rows_are_orthonormal(basis):
    assert basis.shape == (8, FINE_SIZE**2)
    assert basis @ basis.T == pytest.approx(np.eye(8), abs=1e-6)


def test_featureless_dusty_pair_fools_stage_one_but_not_stage_two(basis):
    a, b = featureless(1), featureless(2)
    coarse = float(thumbnail(*standardised(a)) @ thumbnail(*standardised(b)))
    refined = float(residual_unit(fine(a), basis) @ residual_unit(fine(b), basis))
    assert coarse > 0.5
    assert refined < 0.3


def test_resized_recompressed_copy_survives_stage_two(basis):
    original = make_fundus(seed=7, dust=DUST)
    copy = recompressed(original, (640, 480), quality=80)
    unrelated = make_fundus(seed=8, dust=DUST)
    same = float(residual_unit(fine(original), basis) @ residual_unit(fine(copy), basis))
    other = float(residual_unit(fine(original), basis) @ residual_unit(fine(unrelated), basis))
    assert same > 0.8
    assert other < 0.3


def test_mirrored_copy_matches_after_mirroring(basis):
    original = make_fundus(seed=9)
    flipped = original[:, ::-1].copy()
    direct = float(residual_unit(fine(original), basis) @ residual_unit(fine(flipped), basis))
    mirrored = float(
        residual_unit(fine(original), basis) @ residual_unit(fine_mirrored(fine(flipped)), basis)
    )
    assert mirrored > 0.95
    assert direct < 0.3


def test_detail_separates_textured_from_featureless():
    assert (
        fine_signature(*standardised(featureless(3)))[1]
        < 0.5 * (fine_signature(*standardised(make_fundus(seed=3)))[1])
    )


def test_residual_of_a_signature_inside_the_basis_is_nan(basis):
    assert np.isnan(residual_unit(basis[0] * 3.0, basis)).all()


def test_verify_pairs_end_to_end(tmp_path):
    rows, images = [], {}
    for pid in range(10):
        for eye in ("left", "right"):
            seed = 2 * pid + (eye == "right")
            images[(pid, eye)] = featureless(seed) if pid < 5 else make_fundus(seed=seed)
    images[(9, "right")] = recompressed(images[(6, "left")], (640, 480), quality=85)
    for (pid, eye), rgb in images.items():
        Image.fromarray(rgb).save(tmp_path / f"{pid}_{eye}.jpg", quality=92)
        rows.append({"patient_id": pid, "eye": eye, "file": f"{pid}_{eye}.jpg"})
    eyes = pl.DataFrame(
        rows, schema={"patient_id": pl.Int64, "eye": pl.Enum(["left", "right"]), "file": pl.String}
    )
    candidates = pl.DataFrame(
        [(6, "left", 9, "right", 0.99, "direct"), (0, "left", 1, "left", 0.95, "direct")],
        schema={
            "patient_a": pl.Int64,
            "eye_a": pl.Enum(["left", "right"]),
            "patient_b": pl.Int64,
            "eye_b": pl.Enum(["left", "right"]),
            "correlation": pl.Float64,
            "kind": pl.String,
        },
        orient="row",
    )
    verified = verify_pairs(candidates, eyes, tmp_path, workers=2, reference_eyes=20, components=4)
    assert verified.columns == [*candidates.columns, "fine_correlation", "detail_a", "detail_b"]
    copy, dusty = verified.rows(named=True)
    assert (copy["patient_a"], copy["patient_b"]) == (6, 9)
    assert copy["fine_correlation"] > 0.8
    assert dusty["fine_correlation"] < 0.3
