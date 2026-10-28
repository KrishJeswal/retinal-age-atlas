import numpy as np
import polars as pl
from scipy import ndimage

from rra.data.odir import LABELS


def make_raw_sheet(rows: list[dict]) -> pl.DataFrame:
    """Raw annotation sheet with the official ODIR headers, one dict per patient."""
    return pl.DataFrame(
        {
            "ID": [r["id"] for r in rows],
            "Patient Age": [r["age"] for r in rows],
            "Patient Sex": [r["sex"] for r in rows],
            "Left-Fundus": [f"{r['id']}_left.jpg" for r in rows],
            "Right-Fundus": [f"{r['id']}_right.jpg" for r in rows],
            "Left-Diagnostic Keywords": [r["left_kw"] for r in rows],
            "Right-Diagnostic Keywords": [r["right_kw"] for r in rows],
            **{label: [r["labels"][label] for r in rows] for label in LABELS},
        }
    )


def _labels(**on: int) -> dict[str, int]:
    return {label: on.get(label, 0) for label in LABELS}


THREE_PATIENTS = [
    {
        "id": 0,
        "age": 69,
        "sex": "Female",
        "left_kw": "cataract",
        "right_kw": "normal fundus",
        "labels": _labels(C=1),
    },
    {
        "id": 1,
        "age": 57,
        "sex": "Male",
        "left_kw": "normal fundus",
        "right_kw": "normal fundus",
        "labels": _labels(N=1),
    },
    {
        "id": 2,
        "age": 42,
        "sex": "Male",
        "left_kw": "laser spot，moderate non proliferative retinopathy",
        "right_kw": "moderate non proliferative retinopathy",
        "labels": _labels(D=1),
    },
]


def make_fundus(
    width: int = 800,
    height: int = 600,
    r: float = 280,
    cx: float | None = None,
    cy: float | None = None,
    disc: tuple[float, float] | None = (0.5, 0.0),
    blur: float = 0.0,
    colour: tuple[float, float, float] = (0.8, 0.35, 0.15),
    seed: int = 0,
    dust: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    """Orange disk on black with fine and coarse (vessel-scale) texture and a bright disc blob
    at `disc` (offset from the FOV centre in radii; None for no disc). `dust` adds small dark
    spots at fixed offsets after any blur, like dust on the camera optics, which repeats across
    every image a camera takes. Returns uint8 HxWx3."""
    rng = np.random.default_rng(seed)
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    yy, xx = np.mgrid[:height, :width]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2

    texture = ndimage.gaussian_filter(rng.normal(size=(height, width)), 1.5) * 0.6
    texture += ndimage.gaussian_filter(rng.normal(size=(height, width)), 6) * 1.5
    blob = np.zeros((height, width))
    if disc is not None:
        disc_x, disc_y = cx + disc[0] * r, cy + disc[1] * r
        blob = np.exp(-((xx - disc_x) ** 2 + (yy - disc_y) ** 2) / (2 * (0.08 * r) ** 2))

    rgb = np.zeros((height, width, 3))
    for channel, base in enumerate(colour):
        value = base * (1 + texture) + blob * (1 - base) * 0.9
        rgb[..., channel] = np.where(inside, value, 0.0)
    if blur:
        rgb = ndimage.gaussian_filter(rgb, (blur, blur, 0))
    for dx, dy in dust:
        spot = np.exp(-((xx - cx - dx * r) ** 2 + (yy - cy - dy * r) ** 2) / (2 * (0.012 * r) ** 2))
        rgb *= 1 - 0.35 * spot[..., None]
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
