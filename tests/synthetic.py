import polars as pl

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
