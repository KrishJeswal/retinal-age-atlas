import numpy as np
from scipy import ndimage

SATURATED = 250 / 255
DARK = 5 / 255
ILLUMINATION_SIGMA = 0.1

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_D65_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1] to CIELAB (D65). b* is the blue-yellow axis used as the lens-yellowing
    proxy (threat 7)."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _SRGB_TO_XYZ.T / _D65_WHITE
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lightness = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def quality_metrics(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """D3 metrics on a standardised image (fixed FOV diameter, so sharpness is comparable
    across cameras). `rgb` in [0, 1]; statistics over `mask` only. Sharpness and contrast use
    the green channel, which carries most retinal detail."""
    green = rgb[..., 1]
    inside = green[mask]
    diameter = mask.shape[0]
    illumination = ndimage.gaussian_filter(green, ILLUMINATION_SIGMA * diameter)[mask]
    lab = srgb_to_lab(rgb[mask])
    return {
        "sharpness": float(ndimage.laplace(green)[ndimage.binary_erosion(mask)].var()),
        "contrast": float(inside.std()),
        "brightness": float(rgb[mask].mean()),
        "illum_cv": float(illumination.std() / illumination.mean()),
        "clip_high": float((rgb[mask].max(axis=1) >= SATURATED).mean()),
        "clip_low": float((rgb[mask].max(axis=1) <= DARK).mean()),
        "lab_l": float(lab[:, 0].mean()),
        "lab_a": float(lab[:, 1].mean()),
        "lab_b": float(lab[:, 2].mean()),
    }
