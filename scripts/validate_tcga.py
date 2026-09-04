#!/usr/bin/env python3
"""Reproduce the univariable TCGA-PAAD survival analyses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


ROOT = Path(__file__).resolve().parents[1]


def cox_statistics(frame, covariate):
    model = CoxPHFitter()
    model.fit(frame[["OS_time_days", "OS_event", covariate]],
              duration_col="OS_time_days", event_col="OS_event")
    ci = model.confidence_intervals_.loc[covariate]
    return {
        "hr": float(np.exp(model.params_[covariate])),
        "ci_low": float(np.exp(ci["95% lower-bound"])),
        "ci_high": float(np.exp(ci["95% upper-bound"])),
        "p": float(model.summary.loc[covariate, "p"]),
    }


def main():
    data = pd.read_csv(ROOT / "data" / "tcga_validation.csv")
    results = {"cohort_n": int(len(data)), "models": "univariable"}

    continuous = cox_statistics(data, "squamous_module_score")
    results["continuous_cox_per_sd"] = continuous

    q75 = data["squamous_module_score"].quantile(0.75)
    data["top25"] = (data["squamous_module_score"] >= q75).astype(int)
    top = data[data["top25"] == 1]
    bottom = data[data["top25"] == 0]
    quartile = cox_statistics(data, "top25")
    quartile["logrank_p"] = float(logrank_test(
        top["OS_time_days"], bottom["OS_time_days"],
        top["OS_event"], bottom["OS_event"],
    ).p_value)
    for label, frame in (("top25", top), ("bottom75", bottom)):
        km = KaplanMeierFitter().fit(frame["OS_time_days"], frame["OS_event"])
        quartile[f"{label}_n"] = int(len(frame))
        quartile[f"{label}_events"] = int(frame["OS_event"].sum())
        quartile[f"{label}_median_os_days"] = float(km.median_survival_time_)
    results["upper_quartile"] = quartile

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "tcga_survival_statistics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

