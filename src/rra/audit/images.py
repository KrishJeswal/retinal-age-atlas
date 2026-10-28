import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from scipy import ndimage

from rra.audit.laterality import disc_offset
from rra.audit.quality import quality_metrics
from rra.data.fov import SHAPE_CLASSES, circle_mask, fov_geometry, fov_mask
from rra.data.manifest import KEY

WORK_LONG_SIDE = 1024
STANDARD_DIAMETER = 512
INNER_MARGIN = 0.05
THUMBNAIL_SIZE = 64
HIGHPASS_SIGMA = 0.02

FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "decode_ok": pl.Boolean(),
    "error": pl.String(),
    "native_width": pl.Int64(),
    "native_height": pl.Int64(),
    "format": pl.String(),
    "mode": pl.String(),
    "file_bytes": pl.Int64(),
    "sha256": pl.String(),
    "fov_cx": pl.Float64(),
    "fov_cy": pl.Float64(),
    "fov_r": pl.Float64(),
    "fov_rms": pl.Float64(),
    "fov_area_fraction": pl.Float64(),
    "fov_truncation": pl.Float64(),
    "fov_shape": pl.Enum(SHAPE_CLASSES),
    "thumbnail": pl.Array(pl.Float32, THUMBNAIL_SIZE**2),
    "sharpness": pl.Float64(),
    "contrast": pl.Float64(),
    "brightness": pl.Float64(),
    "illum_cv": pl.Float64(),
    "clip_high": pl.Float64(),
    "clip_low": pl.Float64(),
    "lab_l": pl.Float64(),
    "lab_a": pl.Float64(),
    "lab_b": pl.Float64(),
    "disc_dx": pl.Float64(),
    "disc_dy": pl.Float64(),
    "disc_peak": pl.Float64(),
}


def load_work_image(path: Path, long_side: int = WORK_LONG_SIDE) -> tuple[Image.Image, dict]:
    """Native size and format from the header, then a reduced decode. JPEG `draft` decodes at
    1/2, 1/4 or 1/8 scale in the DCT, which is far faster than decoding full resolution."""
    image = Image.open(path)
    native = {
        "native_width": image.width,
        "native_height": image.height,
        "format": image.format,
        "mode": image.mode,
    }
    scale = long_side / max(image.size)
    if scale < 1:
        image.draft("RGB", (round(image.width * scale), round(image.height * scale)))
    image = image.convert("RGB")
    image.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    return image, native


def standardise(
    image: Image.Image, cx: float, cy: float, r: float
) -> tuple[np.ndarray, np.ndarray]:
    """Square crop around the fitted circle (PIL pads out-of-frame areas with black), resized to
    a fixed FOV diameter, plus the mask: inside the shrunken circle and inside the real FOV."""
    box = (round(cx - r), round(cy - r), round(cx + r), round(cy + r))
    crop = image.crop(box).resize((STANDARD_DIAMETER, STANDARD_DIAMETER), Image.Resampling.LANCZOS)
    rgb = np.asarray(crop, dtype=np.float64) / 255
    centre = (STANDARD_DIAMETER - 1) / 2
    inner = circle_mask(
        STANDARD_DIAMETER, STANDARD_DIAMETER, centre, centre, (1 - INNER_MARGIN) * centre
    )
    return rgb, inner & fov_mask(np.asarray(crop))


def thumbnail(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Duplicate-detection signature: green channel minus a blur at ~10 px of a 512 px FOV,
    which keeps vessel-scale detail and removes illumination, vignetting, the FOV disk and the
    disc blob (all shared across eyes; a wider cutoff let the disc alone give unrelated eyes
    r ~ 0.45 on synthetic data). Block-averaged to 64x64 on the FOV-aligned image, centred,
    unit norm, so a dot product is a correlation."""
    green = rgb[..., 1].copy()
    green[~mask] = green[mask].mean()
    highpass = green - ndimage.gaussian_filter(green, HIGHPASS_SIGMA * green.shape[0])
    highpass[~mask] = 0
    block = green.shape[0] // THUMBNAIL_SIZE
    small = highpass.reshape(THUMBNAIL_SIZE, block, THUMBNAIL_SIZE, block).mean(axis=(1, 3))
    vector = small.ravel() - small.mean()
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def mirrored(thumbnails: np.ndarray) -> np.ndarray:
    n = THUMBNAIL_SIZE
    return thumbnails.reshape(-1, n, n)[:, :, ::-1].reshape(-1, n * n)


def correlation_blocks(vectors: np.ndarray, chunk: int = 1024):
    """(start, kind, correlations of rows start:start+chunk with every eye), as-is and with the
    other eye mirrored. Blocked so 7k x 7k never sits in memory twice."""
    flipped = mirrored(vectors)
    for start in range(0, len(vectors), chunk):
        block = vectors[start : start + chunk]
        for kind, other in (("direct", vectors), ("mirrored", flipped)):
            yield start, kind, block @ other.T


def usable_thumbnails(features: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    usable = features.filter(pl.col("thumbnail").is_not_null())
    return usable.select(KEY), usable["thumbnail"].to_numpy().astype(np.float64)


def nearest_correlation(features: pl.DataFrame, chunk: int = 1024) -> pl.DataFrame:
    """Per eye, its highest correlation with any other eye (direct or mirrored). The histogram of
    this is where the duplicate threshold is read off: unrelated eyes form a bulk near 0,
    copies a separate mode near 1."""
    keys, vectors = usable_thumbnails(features)
    best = np.full(len(vectors), -np.inf)
    for start, _, correlation in correlation_blocks(vectors, chunk):
        rows = np.arange(len(correlation))
        correlation[rows, rows + start] = -np.inf
        best[start : start + len(correlation)] = np.maximum(
            best[start : start + len(correlation)], correlation.max(axis=1)
        )
    return keys.with_columns(nearest_correlation=pl.Series(best))


def near_duplicates(features: pl.DataFrame, threshold: float, chunk: int = 1024) -> pl.DataFrame:
    """Every pair of eyes whose thumbnails correlate above `threshold`, as-is or with one
    mirrored (a flipped copy filed as the fellow eye). One row per pair, best correlation."""
    keys, vectors = usable_thumbnails(features)
    pairs = []
    for start, kind, correlation in correlation_blocks(vectors, chunk):
        rows, cols = np.nonzero(correlation > threshold)
        keep = rows + start < cols
        pairs += [
            (int(i + start), int(j), float(correlation[i, j]), kind)
            for i, j in zip(rows[keep], cols[keep], strict=True)
        ]
    keys = keys.with_row_index("index")
    found = pl.DataFrame(
        pairs,
        schema={"a": pl.UInt32, "b": pl.UInt32, "correlation": pl.Float64, "kind": pl.String},
        orient="row",
    )
    return (
        found.sort("correlation", descending=True)
        .unique(["a", "b"], keep="first")
        .join(keys.rename({"index": "a", "patient_id": "patient_a", "eye": "eye_a"}), on="a")
        .join(keys.rename({"index": "b", "patient_id": "patient_b", "eye": "eye_b"}), on="b")
        .select("patient_a", "eye_a", "patient_b", "eye_b", "correlation", "kind")
        .sort("correlation", descending=True)
    )


def image_features(path: Path) -> dict:
    """Every per-image audit feature from one decode. Failures are recorded, never raised, so
    one corrupt file cannot stop the sweep."""
    row: dict = dict.fromkeys(FEATURE_SCHEMA)
    try:
        data = path.read_bytes()
        row["file_bytes"] = len(data)
        row["sha256"] = hashlib.sha256(data).hexdigest()
        image, native = load_work_image(path)
        row.update(native)
    except Exception as error:
        row.update(decode_ok=False, error=f"{type(error).__name__}: {error}")
        return row
    row["decode_ok"] = True

    work = np.asarray(image)
    fov = fov_geometry(work)
    scale = native["native_width"] / image.width
    row.update(
        fov_cx=fov.cx * scale,
        fov_cy=fov.cy * scale,
        fov_r=fov.r * scale,
        fov_rms=fov.rms * scale,
        fov_area_fraction=fov.area_fraction,
        fov_truncation=fov.truncation,
        fov_shape=fov.shape_class,
    )
    if fov.shape_class == "failed":
        return row

    rgb, mask = standardise(image, fov.cx, fov.cy, fov.r)
    row["thumbnail"] = thumbnail(rgb, mask)
    row.update(quality_metrics(rgb, mask))
    row.update(disc_offset(rgb, mask))
    return row


def sweep(eyes: pl.DataFrame, image_dir: Path, workers: int | None = None) -> pl.DataFrame:
    """One row per eye in `eyes` (KEY + FEATURE_SCHEMA). Missing files appear as decode
    failures; the integrity audit reports them separately."""
    paths = [image_dir / (file or "") for file in eyes["file"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(image_features, paths, chunksize=16))
    return eyes.select(KEY).hstack(pl.DataFrame(rows, schema=FEATURE_SCHEMA))
