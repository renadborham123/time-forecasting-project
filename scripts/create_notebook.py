"""Validate and copy the maintained academic notebook to an optional path."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "churnscope_pipeline.ipynb"
SECTIONS = (
    "A — Problem definition", "B — Synthetic data generation",
    "C — Prediction cutoff and churn target", "D — Exploratory data analysis",
    "E — Feature engineering and leakage audit", "F — Customer segmentation",
    "G — Churn classification", "H — Time-series forecasting",
    "I — SHAP explainability", "J — Customer preferences",
    "K — Agentic retention decision layer", "L — Limitations",
    "M — Exported artifacts",
)


def create_notebook(destination: Path = SOURCE) -> Path:
    notebook = nbformat.read(SOURCE, as_version=4)
    nbformat.validate(notebook)
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook.cells)
    missing = [section for section in SECTIONS if section not in text]
    if missing:
        raise ValueError(f"Academic notebook is missing sections: {missing}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return destination


if __name__ == "__main__":
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SOURCE
    print(f"Validated and wrote {create_notebook(output)}")
