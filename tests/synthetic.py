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
