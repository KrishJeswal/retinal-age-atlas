import dataclasses
import json
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import polars as pl

from rra.analysis import power
from rra.audit import (
    camera,
    cohorts,
    demographics,
    feasibility,
    images,
    integrity,
    keywords,
    quality,
)
from rra.audit import laterality as lat
from rra.audit.config import AuditConfig
from rra.data.manifest import KEY
from rra.data.odir import read_annotations, to_eyes

FEATURES_FILE = "image_features.parquet"
MANIFEST_FILE = "manifest.parquet"
HISTOGRAM_BINS = np.linspace(-1, 1, 201)
VERSIONED = ("numpy", "omegaconf", "pillow", "polars", "scipy", "statsmodels")


def load_eyes(cfg: AuditConfig) -> pl.DataFrame:
    return to_eyes(read_annotations(Path(cfg.annotations)))


def nearest_histogram(nearest: pl.DataFrame) -> pl.DataFrame:
    """Counts of each eye's nearest-neighbour correlation in 0.01 bins: label-free, aggregate,
    and the input for choosing `duplicate_threshold`."""
    counts, edges = np.histogram(nearest["nearest_correlation"].to_numpy(), HISTOGRAM_BINS)
    return pl.DataFrame(
        {"bin_low": edges[:-1].round(2), "bin_high": edges[1:].round(2), "n_eyes": counts}
    )


def sweep_stage(eyes: pl.DataFrame, cfg: AuditConfig) -> dict[str, pl.DataFrame]:
    """Image sweep + duplicate-threshold inputs. Uses no labels or keywords."""
    features = images.sweep(eyes, Path(cfg.image_dir), cfg.workers)
    nearest = images.nearest_correlation(features)
    return {
        "features": features,
        "nearest": nearest,
        "candidates": images.near_duplicates(features, cfg.duplicate_review_floor),
        "histogram": nearest_histogram(nearest),
    }


def included_patients(patients: pl.DataFrame, included: pl.DataFrame) -> pl.DataFrame:
    return patients.join(included.select("patient_id").unique(), on="patient_id", how="semi")


def run_stage(
    eyes: pl.DataFrame,
    mapping: pl.DataFrame,
    cfg: AuditConfig,
    features: pl.DataFrame | None = None,
    image_files: set[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, pl.DataFrame]]:
    """Every audit table plus the per-eye manifest. Without `features` (images not yet
    available) the image-dependent tables (E, F, duplicates, camera strata) are skipped."""
    categories = keywords.eye_categories(eyes, mapping)
    patients_all = demographics.patient_table(eyes)
    tables = {
        "keyword_agreement": keywords.agreement_table(
            patients_all, keywords.keyword_labels(categories)
        )
    }

    exclusions = integrity.metadata_exclusions(eyes, categories, image_files)
    clusters = None
    if features is not None:
        if cfg.duplicate_threshold is None:
            raise ValueError("set duplicate_threshold from the sweep histogram before `run`")
        pairs = images.near_duplicates(features, cfg.duplicate_threshold)
        exclusions = (
            pl.concat([exclusions, integrity.image_exclusions(features, pairs)])
            .unique()
            .sort(*KEY, "reason")
        )
        clusters = camera.camera_clusters(features, cfg.min_cluster_eyes)
        tables["duplicate_pairs_summary"] = pairs.group_by("kind").len().sort("kind")
    tables["exclusions"] = exclusions
    tables["exclusion_flow"] = integrity.exclusion_flow(eyes, exclusions)

    kept = integrity.included(eyes, exclusions)
    kept_categories = categories.join(kept.select(KEY), on=KEY, how="semi")
    pools = cohorts.healthy_pools(kept, kept_categories)
    patients = demographics.patient_table(kept)
    tables["pool_counts"] = pools.select(pl.col("^pool_.*$").sum())
    tables["laterality"] = lat.laterality_table(kept_categories)
    tier_counts = lat.tier_counts(kept_categories)
    tables["tier_counts"] = tier_counts

    exact: tuple[str, ...] = ("sex",)
    if clusters is not None:
        patients = patients.join(
            feasibility.patient_clusters(clusters), on="patient_id", how="left"
        )
        exact = ("sex", "camera_cluster")
    membership = demographics.group_membership(patients, pools)
    tables["demographics"] = demographics.demographics_table(patients, membership)
    tables["age_bins"] = demographics.age_bins(patients, membership, cfg.age_bin_width)
    healthy = patients.join(pools.filter(pl.col("pool_main")).select("patient_id"), on="patient_id")
    cases = patients.join(lat.unilateral_patients(kept_categories, "dr", True), on="patient_id")
    tables["matching_feasibility"] = demographics.matching_feasibility(cases, healthy, exact=exact)

    groups = power.contrast_eye_groups(kept_categories, pools)
    count = dict(tier_counts.select("count", "n").rows())
    tables["power_contrasts"] = power.contrast_mde_table(groups)
    tables["power_h2"] = power.h2_mde_table(
        {"G": count["h2_g_unilateral"], "D": count["h2_d_unilateral"]}
    )
    tables["power_tiers"] = power.tier_mde_table(groups, count["tier_c_cases"])

    if clusters is None:
        healthy = healthy.with_columns(camera_cluster=pl.lit(None, dtype=pl.String))
    tables["fold_feasibility"], tables["sparse_strata"] = feasibility.fold_feasibility(
        healthy, cfg.folds, cfg.seed, cfg.stratum_age_width
    )

    manifest = eyes.join(categories, on=KEY, how="left")
    if features is not None and clusters is not None:
        kept_features = features.join(kept.select(KEY), on=KEY, how="semi")
        validation = quality.validate_quality(kept_features, kept_categories, cfg.n_boot, cfg.seed)
        tables["quality_validation"] = pl.DataFrame([validation])
        kept_clusters = clusters.join(kept.select(KEY), on=KEY, how="semi")
        tables["camera_prevalence"] = camera.cluster_prevalence(kept_clusters, kept_categories)
        tables["camera_association"] = camera.cluster_association(kept_clusters, kept_categories)
        yellow = (
            kept.select(*KEY, "age")
            .join(pools.filter(pl.col("pool_main")).select("patient_id"), on="patient_id")
            .join(features.select(*KEY, "lab_b"), on=KEY)
            .join(clusters.select(*KEY, "camera_cluster"), on=KEY)
        )
        tables["yellowness"] = quality.yellowness_slopes(yellow, cfg.n_boot, cfg.seed)
        manifest = (
            manifest.join(features.drop("thumbnail"), on=KEY, how="left")
            .join(clusters, on=KEY, how="left")
            .join(quality.quality_score(kept_features), on=KEY, how="left")
        )

    first_reason = exclusions.group_by(KEY).agg(pl.col("reason").min().alias("exclusion_reason"))
    manifest = manifest.join(first_reason, on=KEY, how="left").with_columns(
        included=pl.col("exclusion_reason").is_null()
    )
    return manifest.sort(KEY), tables


def git_state() -> dict[str, str | bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def write_tables(tables: dict[str, pl.DataFrame], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.write_csv(directory / f"{name}.csv")


def write_record(cfg: AuditConfig, stage: str, directory: Path, **extra) -> None:
    record = {
        "stage": stage,
        "config": dataclasses.asdict(cfg),
        **git_state(),
        "versions": {name: metadata.version(name) for name in VERSIONED},
        **extra,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"run_{stage}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
