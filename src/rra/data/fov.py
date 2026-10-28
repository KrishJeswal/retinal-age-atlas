from dataclasses import dataclass

import numpy as np
from scipy import ndimage

BACKGROUND_THRESHOLD = 15
CUT_TOLERANCE = 0.005
MAX_RELATIVE_RMS = 0.02
SHAPE_CLASSES = ("full", "cut_vertical", "cut_horizontal", "cut_both", "irregular", "failed")


@dataclass(frozen=True)
class FOV:
    cx: float
    cy: float
    r: float
    rms: float
    area_fraction: float
    truncation: float
    shape_class: str


def fov_mask(rgb: np.ndarray, threshold: float = BACKGROUND_THRESHOLD) -> np.ndarray:
    """Largest bright connected region, holes filled. The background of a fundus photo is near
    black in every channel; max over channels keeps dark-red fundi above threshold."""
    bright = ndimage.binary_opening(rgb.max(axis=2) > threshold, iterations=2)
    labels, n = ndimage.label(bright)
    if n == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(labels == largest)


def boundary_points(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge pixels of the mask, excluding those on the image frame: where the FOV runs off the
    frame the edge is the sensor crop, not the circle."""
    edge = mask & ~ndimage.binary_erosion(mask)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = False
    ys, xs = np.nonzero(edge)
    return xs.astype(float), ys.astype(float)


def fit_circle(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """Kasa algebraic fit: x^2 + y^2 = a x + b y + c is linear in (a, b, c); centre (a/2, b/2),
    r^2 = c + cx^2 + cy^2. Returns (cx, cy, r, rms radial residual)."""
    design = np.column_stack([xs, ys, np.ones_like(xs)])
    (a, b, c), *_ = np.linalg.lstsq(design, xs**2 + ys**2, rcond=None)
    cx, cy = a / 2, b / 2
    r = float(np.sqrt(c + cx**2 + cy**2))
    rms = float(np.sqrt(np.mean((np.hypot(xs - cx, ys - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms


def truncated_fraction(cx: float, cy: float, r: float, width: int, height: int) -> float:
    theta = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    x, y = cx + r * np.cos(theta), cy + r * np.sin(theta)
    outside = (x < -0.5) | (x > width - 0.5) | (y < -0.5) | (y > height - 0.5)
    return float(outside.mean())


def shape_class(cx: float, cy: float, r: float, rms: float, width: int, height: int) -> str:
    if rms > MAX_RELATIVE_RMS * r:
        return "irregular"
    margin = CUT_TOLERANCE * r
    vertical = (cy - r < -0.5 - margin) or (cy + r > height - 0.5 + margin)
    horizontal = (cx - r < -0.5 - margin) or (cx + r > width - 0.5 + margin)
    if vertical and horizontal:
        return "cut_both"
    if vertical:
        return "cut_vertical"
    if horizontal:
        return "cut_horizontal"
    return "full"


def fov_geometry(rgb: np.ndarray) -> FOV:
    """Geometry in the pixel units of `rgb`; callers rescale to native resolution."""
    height, width = rgb.shape[:2]
    mask = fov_mask(rgb)
    xs, ys = boundary_points(mask)
    if len(xs) < 20:
        return FOV(np.nan, np.nan, np.nan, np.nan, float(mask.mean()), np.nan, "failed")
    cx, cy, r, rms = fit_circle(xs, ys)
    return FOV(
        cx=cx,
        cy=cy,
        r=r,
        rms=rms,
        area_fraction=float(mask.mean()),
        truncation=truncated_fraction(cx, cy, r, width, height),
        shape_class=shape_class(cx, cy, r, rms, width, height),
    )


def circle_mask(height: int, width: int, cx: float, cy: float, r: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
