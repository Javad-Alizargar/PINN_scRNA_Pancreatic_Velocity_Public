#!/usr/bin/env python3
"""Check package schemas, de-identification, checksums, and reported endpoints."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    training = pd.read_csv(ROOT / "data" / "pinn_training_data.csv")
    tcga = pd.read_csv(ROOT / "data" / "tcga_validation.csv")
    assert training.shape == (986, 57)
    assert tcga.shape == (182, 10)
    public_columns = set(training.columns) | set(tcga.columns)
    assert {"sample", "patient", "Unnamed: 0"}.isdisjoint(public_columns)
    assert training["cell_id"].str.fullmatch(r"CELL_\d{4}").all()
    assert tcga["case_id"].str.fullmatch(r"PAAD_\d{4}").all()

    subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_tcga.py")], check=True)
    stats = json.loads((ROOT / "outputs" / "tcga_survival_statistics.json").read_text())
    assert abs(stats["continuous_cox_per_sd"]["p"] - 4.118980594138761e-05) < 1e-12
    assert abs(stats["upper_quartile"]["hr"] - 1.8598253481948765) < 1e-12
    decrement = (stats["upper_quartile"]["bottom75_median_os_days"]
                 - stats["upper_quartile"]["top25_median_os_days"])
    assert decrement == 201.0

    expected = {}
    for line in (ROOT / "checksums.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    for name, digest in expected.items():
        observed = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert observed == digest, name
    print("Package verification passed.")


if __name__ == "__main__":
    main()
