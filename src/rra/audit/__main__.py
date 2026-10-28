import argparse
from pathlib import Path

import polars as pl

from rra.audit import keywords, pipeline
from rra.audit.config import load_config
from rra.data import manifest as mf


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m rra.audit")
    parser.add_argument("stage", choices=["vocabulary", "sweep", "run"])
    parser.add_argument("--config", type=Path, default=Path("configs/audit.yaml"))
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. workers=4")
    args = parser.parse_intermixed_args(argv)

    cfg = load_config(args.config if args.config.exists() else None, args.overrides)
    results, derived = Path(cfg.results_dir), Path(cfg.derived_dir)
    eyes = pipeline.load_eyes(cfg)

    if args.stage == "vocabulary":
        pipeline.write_tables({"vocabulary": keywords.vocabulary(eyes)}, results)
    elif args.stage == "sweep":
        out = pipeline.sweep_stage(eyes, cfg)
        derived.mkdir(parents=True, exist_ok=True)
        out["features"].write_parquet(derived / pipeline.FEATURES_FILE)
        out["candidates"].write_csv(derived / "duplicate_candidates.csv")
        pipeline.write_tables({"nearest_correlation_histogram": out["histogram"]}, results)
    else:
        features_path = derived / pipeline.FEATURES_FILE
        features = pl.read_parquet(features_path) if features_path.exists() else None
        image_dir = Path(cfg.image_dir)
        image_files = {p.name for p in image_dir.iterdir()} if image_dir.is_dir() else None
        manifest, tables = pipeline.run_stage(
            eyes, keywords.read_mapping(Path(cfg.mapping)), cfg, features, image_files
        )
        digest = mf.write(manifest, derived / pipeline.MANIFEST_FILE)
        pipeline.write_tables(tables, results)
        (results / "manifest_hash.txt").write_text(digest + "\n")
        pipeline.write_record(
            cfg, args.stage, results, manifest_hash=digest, images=features is not None
        )
        return
    pipeline.write_record(cfg, args.stage, results)


if __name__ == "__main__":
    main()
