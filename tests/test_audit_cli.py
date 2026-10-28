import json

import numpy as np
import polars as pl
import pytest
from PIL import Image

from rra.audit.__main__ import main
from rra.data import manifest as mf
from rra.data.odir import LABELS
from synthetic import make_fundus, make_raw_sheet

KEYWORDS = [
    ("normal fundus", "normal fundus", "N"),
    ("normal fundus，lens dust", "normal fundus", "N"),
    ("mild nonproliferative retinopathy", "normal fundus", "D"),
    ("moderate non proliferative retinopathy", "diabetic retinopathy", "D"),
    ("glaucoma", "suspected glaucoma", "G"),
    ("normal fundus", "glaucoma", "G"),
    ("hypertensive retinopathy", "normal fundus", "H"),
    ("low image quality", "normal fundus", "O"),
    ("cataract", "cataract", "C"),
]


@pytest.fixture
def odir(tmp_path, mapping):
    rng = np.random.default_rng(0)
    rows = []
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for patient in range(36):
        left, right, label = KEYWORDS[patient % len(KEYWORDS)]
        rows.append(
            {
                "id": patient,
                "age": int(rng.integers(30, 80)),
                "sex": ["Male", "Female"][patient % 2],
                "left_kw": left,
                "right_kw": right,
                "labels": {**dict.fromkeys(LABELS, 0), label: 1},
            }
        )
        for side, eye_seed in (("left", 2 * patient), ("right", 2 * patient + 1)):
            rgb = make_fundus(width=400, height=300, r=140, seed=eye_seed)
            Image.fromarray(rgb).save(image_dir / f"{patient}_{side}.jpg", quality=92)
    Image.open(image_dir / "3_left.jpg").save(image_dir / "20_right.jpg", quality=80)
    make_raw_sheet(rows).write_excel(tmp_path / "annotations.xlsx")
    mapping.write_csv(tmp_path / "mapping.csv")
    return tmp_path


def run(odir, stage, *extra):
    main(
        [
            stage,
            "--config",
            str(odir / "absent.yaml"),
            f"annotations={odir / 'annotations.xlsx'}",
            f"image_dir={odir / 'images'}",
            f"mapping={odir / 'mapping.csv'}",
            f"derived_dir={odir / 'derived'}",
            f"results_dir={odir / 'results'}",
            "workers=2",
            "n_boot=50",
            "min_cluster_eyes=10",
            *extra,
        ]
    )


def test_vocabulary_stage(odir):
    run(odir, "vocabulary")
    vocabulary = pl.read_csv(odir / "results" / "vocabulary.csv")
    counts = dict(vocabulary.select("term", "n_eyes").rows())
    assert counts["normal fundus"] == 32
    assert counts["lens dust"] == 4
    assert (
        json.loads((odir / "results" / "run_vocabulary.json").read_text())["stage"] == "vocabulary"
    )


def test_metadata_only_run_before_images(odir):
    run(odir, "run", f"derived_dir={odir / 'no_features'}")
    results = odir / "results"
    assert not (results / "camera_association.csv").exists()
    assert pl.read_csv(results / "exclusion_flow.csv")["excluded_eyes"].sum() == 0
    manifest = mf.read(odir / "no_features" / "manifest.parquet")
    assert manifest.height == 72
    assert manifest["included"].all()


def test_sweep_then_run(odir):
    run(odir, "sweep")
    histogram = pl.read_csv(odir / "results" / "nearest_correlation_histogram.csv")
    assert histogram["n_eyes"].sum() == 72
    assert histogram.filter(pl.col("bin_low") >= 0.9)["n_eyes"].sum() == 2

    with pytest.raises(ValueError, match="duplicate_threshold"):
        run(odir, "run")
    run(odir, "run", "duplicate_threshold=0.9")

    results = odir / "results"
    exclusions = pl.read_csv(results / "exclusions.csv")
    assert set(exclusions.select("patient_id", "eye", "reason").rows()) == {
        (3, "left", "duplicate_image"),
        (20, "right", "duplicate_image"),
    }
    for name in (
        "keyword_agreement",
        "pool_counts",
        "laterality",
        "tier_counts",
        "demographics",
        "matching_feasibility",
        "power_contrasts",
        "power_h2",
        "power_tiers",
        "fold_feasibility",
        "quality_validation",
        "camera_prevalence",
        "camera_association",
        "yellowness",
    ):
        assert (results / f"{name}.csv").exists(), name

    manifest = mf.read(odir / "derived" / "manifest.parquet")
    assert "thumbnail" not in manifest.columns
    assert manifest.filter(~pl.col("included")).height == 2
    assert manifest["camera_cluster"].unique().to_list() == ["400x300/full"]
    digest = (results / "manifest_hash.txt").read_text().strip()
    assert digest == mf.content_hash(manifest)
    record = json.loads((results / "run_run.json").read_text())
    assert record["manifest_hash"] == digest
    assert record["config"]["duplicate_threshold"] == 0.9
