import warnings

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import ndimage, stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationWarning

from rra.analysis.bootstrap import cluster_bootstrap, percentile_ci
from rra.analysis.discrimination import auc, auc_with_ci
from rra.data.manifest import KEY

SATURATED = 250 / 255
DARK = 5 / 255
ILLUMINATION_SIGMA = 0.1
EYEQ_FALLBACK_AUC = 0.75
SCORE_FOLDS = 5

QUALITY_DIRECTION = {
    "sharpness": -1,
    "contrast": -1,
    "illum_cv": 1,
    "clip_high": 1,
    "clip_low": 1,
}

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


def normal_scores(values: np.ndarray) -> np.ndarray:
    """Rank-based z: Phi^-1((rank - 0.5) / n), ties averaged, NaN kept. Monotone like a plain
    z-score but not dominated by the long tails of sharpness and the clip fractions."""
    out = np.full(len(values), np.nan)
    present = ~np.isnan(values)
    out[present] = stats.norm.ppf((stats.rankdata(values[present]) - 0.5) / present.sum())
    return out


def signed_scores(features: pl.DataFrame) -> pl.DataFrame:
    """KEY + one normal score per metric, signed so higher always means worse quality. Ranks are
    over every eye with metrics (no labels involved)."""
    return features.select(KEY).with_columns(
        pl.Series(
            metric,
            sign * normal_scores(features[metric].cast(pl.Float64).to_numpy()),
            nan_to_null=True,
        )
        for metric, sign in QUALITY_DIRECTION.items()
    )


def quality_score(features: pl.DataFrame) -> pl.DataFrame:
    """D3 primary score: unweighted mean of the signed normal scores. Nothing is fitted, so its
    AUC against the keywords is an honest validation. Null if any metric is missing."""
    metrics = list(QUALITY_DIRECTION)
    return signed_scores(features).select(
        *KEY,
        pl.when(pl.all_horizontal(pl.col(metrics).is_not_null()))
        .then(pl.mean_horizontal(metrics))
        .alias("quality_score"),
    )


def low_quality_keyword(categories: pl.DataFrame) -> pl.DataFrame:
    """Validation target: any quality or ungradable keyword on the eye."""
    return categories.select(
        *KEY, (pl.col("cat_quality") | pl.col("cat_ungradable")).alias("low_quality_kw")
    )


def crossfit_logistic_score(
    features: pl.DataFrame, target: pl.DataFrame, folds: int = SCORE_FOLDS, seed: int = 0
) -> pl.DataFrame:
    """Sensitivity score: logistic regression of the keyword target on the signed scores, each
    eye predicted by a model that never saw its patient (patient-level folds). Constant metrics
    are dropped; if any fold cannot be fitted (separation, singular, no convergence) every score
    is NaN, so the sensitivity is reported as missing rather than half-computed."""
    data = signed_scores(features).join(target, on=KEY).drop_nulls()
    patients = data["patient_id"].unique().sort().to_numpy()
    fold_of = dict(
        zip(patients, np.random.default_rng(seed).permutation(len(patients)) % folds, strict=True)
    )
    fold = np.array([fold_of[p] for p in data["patient_id"]])
    metrics = data.select(list(QUALITY_DIRECTION)).to_numpy()
    x = sm.add_constant(metrics[:, metrics.std(axis=0) > 0], has_constant="add")
    y = data["low_quality_kw"].to_numpy().astype(float)
    predicted = np.full(len(y), np.nan)
    try:
        with warnings.catch_warnings(), np.errstate(over="ignore"):
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("error", PerfectSeparationWarning)
            for k in range(folds):
                test = fold == k
                model = sm.Logit(y[~test], x[~test]).fit(disp=0)
                if not model.mle_retvals["converged"]:
                    raise np.linalg.LinAlgError("logistic fit did not converge")
                predicted[test] = model.predict(x[test])
    except (np.linalg.LinAlgError, PerfectSeparationWarning):
        predicted[:] = np.nan
    return data.select(*KEY, "low_quality_kw").with_columns(logistic_score=pl.Series(predicted))


def validate_quality(
    features: pl.DataFrame, categories: pl.DataFrame, n_boot: int = 2000, seed: int = 0
) -> dict[str, float | int | bool]:
    """D3 validation: AUC of the fixed score for the low-quality keywords with a patient
    bootstrap CI, plus the cross-fitted logistic AUC. `passes` is False -> EyeQ fallback."""
    target = low_quality_keyword(categories)
    data = quality_score(features).join(target, on=KEY).drop_nulls()
    result = auc_with_ci(
        data["quality_score"].to_numpy(),
        data["low_quality_kw"].to_numpy(),
        data["patient_id"].to_numpy(),
        n_boot,
        seed,
    )
    logistic = crossfit_logistic_score(features, target, seed=seed)
    result["logistic_auc"] = auc(
        logistic["logistic_score"].to_numpy(), logistic["low_quality_kw"].to_numpy()
    )
    result["passes"] = result["auc"] >= EYEQ_FALLBACK_AUC
    return result


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x**2).sum())


def yellowness_slopes(
    data: pl.DataFrame, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> pl.DataFrame:
    """D5 / threat 7: per camera cluster (and "all"), the OLS slope of b* (yellowness) on age
    in healthy eyes, in b* units per year, with a patient-bootstrap CI. Descriptive: shows
    whether lens yellowing is visible to the model and whether cameras differ in it."""
    data = data.drop_nulls(["age", "lab_b", "camera_cluster"])
    groups = [("all", data)] + [
        (name, part) for (name,), part in data.group_by("camera_cluster", maintain_order=True)
    ]
    rows = []
    for name, part in sorted(groups, key=lambda g: (g[0] != "all", g[0])):
        x, y = part["age"].to_numpy().astype(float), part["lab_b"].to_numpy()
        if part["patient_id"].n_unique() < 3 or np.ptp(x) == 0:
            continue
        replicates = cluster_bootstrap(
            lambda i, x=x, y=y: ols_slope(x[i], y[i]) if np.ptp(x[i]) > 0 else np.nan,
            part["patient_id"].to_numpy(),
            n_boot,
            seed,
        )
        low, high = percentile_ci(replicates, level)
        rows.append(
            {
                "camera_cluster": name,
                "n_eyes": part.height,
                "n_patients": part["patient_id"].n_unique(),
                "slope_per_year": ols_slope(x, y),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pl.DataFrame(rows)
