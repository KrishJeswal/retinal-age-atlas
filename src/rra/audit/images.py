import hashlib
import multiprocessing
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
FINE_SIZE = 256
FINE_SIGMA = 0.006
FINE_RADIUS = 0.85
REFERENCE_EYES = 400
SHARED_COMPONENTS = 8

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
    """Square window around the fitted circle at a fixed FOV diameter (out-of-frame areas are
    black), plus the mask: inside the shrunken circle and inside the real FOV. The image is
    first resized as a whole (Lanczos, antialiased), then the window is cut at sub-pixel
    position by a bicubic EXTENT transform at ~1:1 scale; rounding the crop box instead
    shifts flipped or recropped copies by up to half a pixel, which the fine duplicate
    signature is sensitive to."""
    scale = STANDARD_DIAMETER / (2 * r)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    sx, sy = size[0] / image.width, size[1] / image.height
    # fov_geometry works in pixel-index coordinates; EXTENT in continuous ones (pixel i spans
    # [i, i + 1)), so the centre moves by +0.5, or a flipped copy lands a pixel off.
    x0, y0 = cx + 0.5, cy + 0.5
    extent = (sx * (x0 - r), sy * (y0 - r), sx * (x0 + r), sy * (y0 + r))
    crop = resized.transform(
        (STANDARD_DIAMETER, STANDARD_DIAMETER),
        Image.Transform.EXTENT,
        extent,
        resample=Image.Resampling.BICUBIC,
    )
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
            yield start, kind, np.clip(block @ other.T, -1.0, 1.0)


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


def fine_signature(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Stage-2 duplicate signature: green minus a blur at ~3 px of a 512 px FOV (individual
    vessel branching survives), inner 85% of the FOV only (no rim vignetting), 2x2-averaged to
    256x256. Returned unnormalised, with `detail` = SD of the high-pass inside the region; a
    featureless (hazy, blurred) image has little detail and cannot be matched on anatomy."""
    green = rgb[..., 1].copy()
    green[~mask] = green[mask].mean()
    highpass = green - ndimage.gaussian_filter(green, FINE_SIGMA * green.shape[0])
    centre = (green.shape[0] - 1) / 2
    region = mask & circle_mask(*green.shape, centre, centre, FINE_RADIUS * centre)
    highpass[~region] = 0
    block = green.shape[0] // FINE_SIZE
    small = highpass.reshape(FINE_SIZE, block, FINE_SIZE, block).mean(axis=(1, 3))
    return small.ravel().astype(np.float32), float(highpass[region].std())


def fine_features(path: Path) -> dict:
    try:
        image, _ = load_work_image(path)
        fov = fov_geometry(np.asarray(image))
        if fov.shape_class == "failed":
            return {"fine": None, "detail": None}
        signature, detail = fine_signature(*standardise(image, fov.cx, fov.cy, fov.r))
    except Exception:
        return {"fine": None, "detail": None}
    return {"fine": signature, "detail": detail}


def shared_basis(reference: np.ndarray, components: int = SHARED_COMPONENTS) -> np.ndarray:
    """Orthonormal rows spanning what fine signatures share across the population (rim, disc
    and fovea positions, camera dust): the top right singular vectors of the uncentred reference
    matrix, so the mean pattern is included. Rows are scaled to unit norm first: a sharp image
    carries ~15x the high-pass energy of a hazy one, and without scaling the top components
    are just the sharpest individual images, not what images share."""
    rows = reference.astype(np.float64)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    rows = rows[norms[:, 0] > 0] / norms[norms[:, 0] > 0]
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    return vt[:components]


def residual_unit(signature: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Unit-norm, centred part of `signature` outside the shared subspace; NaN when (almost)
    nothing is left, so floating-point remainders are never normalised into a pattern."""
    signature = signature.astype(np.float64)
    residual = signature - (signature @ basis.T) @ basis
    residual = residual - residual.mean()
    norm = np.linalg.norm(residual)
    if norm <= 1e-6 * max(np.linalg.norm(signature), 1e-12):
        return np.full_like(residual, np.nan)
    return residual / norm


def fine_mirrored(signature: np.ndarray) -> np.ndarray:
    return signature.reshape(FINE_SIZE, FINE_SIZE)[:, ::-1].ravel()


def verify_pairs(
    candidates: pl.DataFrame,
    eyes: pl.DataFrame,
    image_dir: Path,
    workers: int | None = None,
    reference_eyes: int = REFERENCE_EYES,
    components: int = SHARED_COMPONENTS,
    seed: int = 0,
) -> pl.DataFrame:
    """Stage 2: re-decode only the candidate images plus a seeded random reference sample of
    eyes outside every candidate pair (a duplicate pair inside the reference would become a
    "shared" component and be projected out of itself), remove the population-shared subspace
    from each fine signature and correlate the residuals
    (the second image mirrored for `mirrored` pairs). Adds `fine_correlation`, `detail_a`,
    `detail_b`. Label-free."""
    files = dict(zip(eyes.select(KEY).rows(), eyes["file"], strict=True))
    wanted = set(candidates.select("patient_a", "eye_a").rows())
    wanted |= set(candidates.select("patient_b", "eye_b").rows())
    pool = sorted(k for k, f in files.items() if f and k not in wanted)
    picked = np.random.default_rng(seed).choice(
        len(pool), size=min(reference_eyes, len(pool)), replace=False
    )
    reference = [pool[i] for i in sorted(picked)]
    keys = sorted(wanted | set(reference))
    paths = [image_dir / (files.get(k) or "") for k in keys]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        computed = dict(zip(keys, executor.map(fine_features, paths, chunksize=8), strict=True))

    usable = [computed[k]["fine"] for k in reference if computed[k]["fine"] is not None]
    basis = shared_basis(np.stack(usable), components)
    rows = []
    for pair in candidates.iter_rows(named=True):
        a = computed[(pair["patient_a"], pair["eye_a"])]
        b = computed[(pair["patient_b"], pair["eye_b"])]
        correlation = None
        if a["fine"] is not None and b["fine"] is not None:
            other = fine_mirrored(b["fine"]) if pair["kind"] == "mirrored" else b["fine"]
            value = float(residual_unit(a["fine"], basis) @ residual_unit(other, basis))
            correlation = None if np.isnan(value) else float(np.clip(value, -1.0, 1.0))
        rows.append(
            {
                **pair,
                "fine_correlation": correlation,
                "detail_a": a["detail"],
                "detail_b": b["detail"],
            }
        )
    schema = {
        **candidates.schema,
        "fine_correlation": pl.Float64,
        "detail_a": pl.Float64,
        "detail_b": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema).sort(
        "fine_correlation", descending=True, nulls_last=True
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
    failures; the integrity audit reports them separately. Workers are spawned, not forked:
    forking after Polars/NumPy have started threads can deadlock the children on Linux."""
    paths = [image_dir / (file or "") for file in eyes["file"]]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        rows = list(pool.map(image_features, paths, chunksize=16))
    return eyes.select(KEY).hstack(pl.DataFrame(rows, schema=FEATURE_SCHEMA))
