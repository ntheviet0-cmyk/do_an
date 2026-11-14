"""Build the self-contained tft_m5_gemini_update.ipynb notebook."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tft_m5_gemini_update_src.py"
EVALUATOR = HERE / "wrmsse5.py"
FEATURES = HERE / "hurdle_features.py"
OUTPUT = HERE / "tft_m5_gemini_update.ipynb"


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
                cells.append({
                    "cell_type": kind,
                    "metadata": {},
                    "source": source,
                    **({"execution_count": None, "outputs": []} if kind == "code" else {}),
                })
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
        cells.append({
            "cell_type": kind,
            "metadata": {},
            "source": source,
            **({"execution_count": None, "outputs": []} if kind == "code" else {}),
        })
    return cells


def main():
    preset_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": (
            "# Cấu hình full M5 (30.490 chuỗi), đặt SMOKE=1 nếu muốn test nhanh trên CPU.\n"
            "import os\n"
            "os.environ.setdefault(\"PROFILE\", \"full_5level\")\n"
            "os.environ.setdefault(\"SMOKE\", \"0\")\n"
            "os.environ.setdefault(\"FORECAST_SPLIT\", \"evaluation\")\n"
            "os.environ.setdefault(\"FILTER_PRELAUNCH\", \"1\")\n"
            "print(\"Preset loaded:\", os.environ[\"PROFILE\"], \"| smoke:\", os.environ[\"SMOKE\"])\n"
        ),
    }
    evaluator_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "%%writefile wrmsse5.py\n" + EVALUATOR.read_text(encoding="utf-8"),
    }
    feature_cell = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "%%writefile hurdle_features.py\n" + FEATURES.read_text(encoding="utf-8"),
    }
    parsed_cells = source_cells(SOURCE.read_text(encoding="utf-8"))
    if parsed_cells and parsed_cells[0]["cell_type"] == "markdown":
        cells = [parsed_cells[0], preset_cell, evaluator_cell, feature_cell, *parsed_cells[1:]]
    else:
        cells = [preset_cell, evaluator_cell, feature_cell, *parsed_cells]

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
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Successfully built:", OUTPUT.name)


if __name__ == "__main__":
    main()
