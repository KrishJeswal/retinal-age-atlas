import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from rra.audit.images import (
    FEATURE_SCHEMA,
    THUMBNAIL_SIZE,
    image_features,
    near_duplicates,
    nearest_correlation,
    standardise,
    sweep,
    thumbnail,
)
from rra.audit.laterality import disc_offset
from rra.audit.quality import quality_metrics, srgb_to_lab
from rra.data.fov import circle_mask, fov_geometry
from synthetic import make_fundus


def save_jpeg(rgb: np.ndarray, path, quality: int = 95):
    Image.fromarray(rgb).save(path, quality=quality)
    return path


def disk_mask(size: int = 256) -> np.ndarray:
    centre = (size - 1) / 2
    return circle_mask(size, size, centre, centre, 0.95 * centre)


def square(rgb_size: int = 256, **kwargs) -> np.ndarray:
    return make_fundus(width=rgb_size, height=rgb_size, r=rgb_size / 2, **kwargs) / 255


@pytest.mark.parametrize(
    ("rgb", "lab"),
    [
        ((1.0, 1.0, 1.0), (100.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (53.24, 80.09, 67.20)),
        ((0.0, 0.0, 1.0), (32.30, 79.19, -107.86)),
    ],
)
def test_srgb_to_lab_reference_values(rgb, lab):
    assert srgb_to_lab(np.array(rgb)) == pytest.approx(lab, abs=0.05)


def test_sharpness_falls_with_blur():
    mask = disk_mask()
    values = [quality_metrics(square(blur=b), mask)["sharpness"] for b in (0, 1, 2, 4)]
    assert values == sorted(values, reverse=True)


def test_clipping_fraction_matches_planted_saturation():
    rgb, mask = square(disc=None, colour=(0.5, 0.25, 0.1)), disk_mask()
    assert quality_metrics(rgb, mask)["clip_high"] == 0
    rgb[:64] = 1.0
    planted = mask[:64].sum() / mask.sum()
    assert quality_metrics(rgb, mask)["clip_high"] == pytest.approx(planted, abs=1e-9)


def test_illumination_cv_rises_with_a_gradient():
    rgb, mask = square(), disk_mask()
    ramp = np.linspace(0.3, 1.0, rgb.shape[1])[None, :, None]
    assert quality_metrics(rgb * ramp, mask)["illum_cv"] > quality_metrics(rgb, mask)["illum_cv"]


def test_yellower_image_has_higher_b_star():
    mask = disk_mask()
    plain = square(colour=(0.8, 0.35, 0.25))
    yellow = square(colour=(0.8, 0.35, 0.05))
    assert quality_metrics(yellow, mask)["lab_b"] > quality_metrics(plain, mask)["lab_b"]


@pytest.mark.parametrize(("dx", "dy"), [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.4), (-0.3, -0.3)])
def test_disc_offset_finds_the_planted_disc(dx, dy):
    offset = disc_offset(square(disc=(dx, dy)), disk_mask())
    assert (offset["disc_dx"], offset["disc_dy"]) == pytest.approx((dx, dy), abs=0.05)
    assert offset["disc_peak"] > 1.2


def standardised_thumbnail(rgb: np.ndarray) -> np.ndarray:
    fov = fov_geometry(rgb)
    return thumbnail(*standardise(Image.fromarray(rgb), fov.cx, fov.cy, fov.r))


def recompressed(rgb: np.ndarray, size: tuple[int, int], quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS).save(
        buffer, format="JPEG", quality=quality
    )
    return np.asarray(Image.open(buffer).convert("RGB"))


def test_thumbnail_is_unit_norm_and_zero_mean():
    vector = standardised_thumbnail(make_fundus(seed=1))
    assert vector.shape == (THUMBNAIL_SIZE**2,)
    assert float(vector @ vector) == pytest.approx(1, abs=1e-5)
    assert abs(float(vector.mean())) < 1e-6


def test_thumbnail_separates_copies_from_other_eyes():
    original = make_fundus(seed=1)
    base = standardised_thumbnail(original)
    copy = standardised_thumbnail(recompressed(original, (600, 450), quality=70))
    assert float(base @ copy) > 0.95
    for seed in range(2, 12):
        assert abs(float(base @ standardised_thumbnail(make_fundus(seed=seed)))) < 0.2


def duplicate_features() -> pl.DataFrame:
    original = make_fundus(seed=1)
    rows = [
        (0, "left", original),
        (0, "right", make_fundus(seed=2)),
        (1, "left", recompressed(original, (640, 480), quality=80)),
        (2, "right", original[:, ::-1].copy()),
        (3, "left", make_fundus(seed=3)),
    ]
    return pl.DataFrame(
        {
            "patient_id": [r[0] for r in rows],
            "eye": [r[1] for r in rows],
            "thumbnail": [standardised_thumbnail(r[2]) for r in rows],
        },
        schema={
            "patient_id": pl.Int64,
            "eye": pl.Enum(["left", "right"]),
            "thumbnail": pl.Array(pl.Float32, THUMBNAIL_SIZE**2),
        },
    )


def test_near_duplicates_finds_direct_and_mirrored_copies():
    features = duplicate_features()
    pairs = near_duplicates(features, threshold=0.9, chunk=2)
    found = {(a, ea, b, eb, kind) for a, ea, b, eb, _, kind in pairs.rows()}
    assert found == {
        (0, "left", 1, "left", "direct"),
        (0, "left", 2, "right", "mirrored"),
        (1, "left", 2, "right", "mirrored"),
    }


def test_nearest_correlation_separates_copies_from_unrelated_eyes():
    nearest = nearest_correlation(duplicate_features(), chunk=2)["nearest_correlation"].to_list()
    assert min(nearest[i] for i in (0, 2, 3)) > 0.9
    assert max(nearest[i] for i in (1, 4)) < 0.2


def test_image_features_on_a_jpeg(tmp_path):
    path = save_jpeg(make_fundus(width=1600, height=1200, r=560), tmp_path / "0_left.jpg")
    row = image_features(path)
    assert row["decode_ok"] is True
    assert (row["native_width"], row["native_height"], row["format"]) == (1600, 1200, "JPEG")
    assert row["fov_r"] == pytest.approx(560, rel=0.01)
    assert row["fov_shape"] == "full"
    assert row["disc_dx"] == pytest.approx(0.5, abs=0.05)
    assert set(row) == set(FEATURE_SCHEMA)


def test_image_features_records_failures(tmp_path):
    corrupt = tmp_path / "1_left.jpg"
    corrupt.write_bytes(b"not a jpeg")
    for path in (corrupt, tmp_path / "missing.jpg"):
        row = image_features(path)
        assert row["decode_ok"] is False
        assert row["error"]


def test_sweep_keeps_one_row_per_eye(tmp_path):
    save_jpeg(make_fundus(seed=3), tmp_path / "0_left.jpg")
    save_jpeg(make_fundus(seed=4, disc=(-0.5, 0)), tmp_path / "0_right.jpg")
    eyes = pl.DataFrame(
        {
            "patient_id": [0, 0, 1],
            "eye": ["left", "right", "left"],
            "file": ["0_left.jpg", "0_right.jpg", None],
        },
        schema={"patient_id": pl.Int64, "eye": pl.Enum(["left", "right"]), "file": pl.String},
    )
    features = sweep(eyes, tmp_path, workers=2)
    assert features.height == 3
    assert features["decode_ok"].to_list() == [True, True, False]
    assert features["disc_dx"][0] > 0 > features["disc_dx"][1]
    assert features["thumbnail"][2] is None
    assert features["thumbnail"][0] is not None
