from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class AuditConfig:
    annotations: str = "data/raw/odir/annotations.xlsx"
    image_dir: str = "data/raw/odir/images"
    mapping: str = "prereg/keyword_mapping.csv"
    derived_dir: str = "data/derived"
    results_dir: str = "results/audit"
    workers: int | None = None
    seed: int = 0
    duplicate_threshold: float | None = None
    duplicate_review_floor: float = 0.5
    min_cluster_eyes: int = 50
    n_boot: int = 2000
    age_bin_width: int = 5
    stratum_age_width: int = 10
    folds: int = 5


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> AuditConfig:
    """Defaults <- YAML file <- dotlist overrides, type-checked against AuditConfig."""
    merged = OmegaConf.structured(AuditConfig)
    if path is not None:
        merged = OmegaConf.merge(merged, OmegaConf.load(path))
    merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides or []))
    return OmegaConf.to_object(merged)
