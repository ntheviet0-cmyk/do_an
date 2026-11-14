"""Generate the self-contained Kaggle notebook from the percent-format source."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tft_m5_zero_test_src.py"
EVALUATOR = HERE / "wrmsse5.py"
FEATURES = HERE / "hurdle_features.py"
OUTPUT = HERE / "tft_m5_zero_test.ipynb"


def markdown_source(lines):
    output = []
    for line in lines:
        if line.startswith("# "):
            output.append(line[2:])
        elif line.startswith("#"):
            output.append(line[1:])
        else:
            output.append(line)
    return "".join(output).lstrip("\n")


def source_cells(text):
    cells = []
    kind = "code"
    block = []
    for line in text.splitlines(keepends=True):
        if line.startswith("# %%"):
            if block:
                source = markdown_source(block) if kind == "markdown" else "".join(block)
                cells.append({"cell_type": kind, "metadata": {}, "source": source})
            kind = "markdown" if "[markdown]" in line else "code"
            block = []
        elif kind == "markdown" and line.strip() and not line.startswith("#"):
            if block:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": markdown_source(block),
                })
            kind = "code"
            block = [line]
        else:
            block.append(line)
    if block:
        source = markdown_source(block) if kind == "markdown" else "".join(block)
        cells.append({"cell_type": kind, "metadata": {}, "source": source})
    return cells


def main():
    evaluator_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": "%%writefile wrmsse5.py\n" + EVALUATOR.read_text(encoding="utf-8"),
    }
    feature_cell = {
        "cell_type": "code",
        "metadata": {},
        "source": "%%writefile hurdle_features.py\n" + FEATURES.read_text(encoding="utf-8"),
    }
    cells = [evaluator_cell, feature_cell, *source_cells(SOURCE.read_text(encoding="utf-8"))]
    if any("FEAT =" in cell["source"] for cell in cells if cell["cell_type"] == "markdown"):
        raise RuntimeError("Feature code was incorrectly emitted as markdown")
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": False, "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=True, indent=1), encoding="utf-8")
    print("wrote", OUTPUT.name)


if __name__ == "__main__":
    main()
