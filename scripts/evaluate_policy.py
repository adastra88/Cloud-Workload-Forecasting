"""One-shot evaluation of the capacity policy on the holdout (after R7).

Uses the frozen band from ``reports/artifacts/forecast_intervals.csv`` and the
frozen provisioning level from ``reports/artifacts/policy_choice.json``.  Reports
shortfall events, average over-provisioning and net cost, plus the cost-ratio
sensitivity table.  Nothing here feeds back into the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cwf import policy as P  # noqa: E402
from cwf.data import load_dataset  # noqa: E402
from cwf.splits import chronological_split  # noqa: E402
from cwf.targets import build_targets  # noqa: E402

ART = ROOT / "reports" / "artifacts"
TARGET = "aggregate_cpu_utilization"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--holdout-len", type=int, default=24)
    args = parser.parse_args(argv)

    choice = json.loads((ART / "policy_choice.json").read_text())
    band = pd.read_csv(ART / "forecast_intervals.csv")

    series = build_targets(load_dataset(Path(args.data)).readings)
    split = chronological_split(series, args.holdout_len, None)
    realised = split.holdout[TARGET].to_numpy(dtype=float)

    forecast = band["forecast_aggregate_cpu_utilization"].to_numpy(dtype=float)
    lo90 = band["lower_90"].to_numpy(dtype=float)
    hi90 = band["upper_90"].to_numpy(dtype=float)

    levels = list(P.DEFAULT_LEVELS)
    prov = {lv: P.level_to_band(forecast, lo90, hi90, lv) for lv in levels}
    prov["mean_forecast"] = forecast.copy()

    rows = []
    for name, p in prov.items():
        level = float(name) if name != "mean_forecast" else float("nan")
        res = P.evaluate_policy(p, realised, c_under=choice["c_under"], c_over=choice["c_over"], level=level)
        row = res.as_dict()
        row["policy"] = name
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(ART / "policy_holdout_evaluation.csv", index=False)

    sens = pd.DataFrame(P.sensitivity(prov, realised))
    sens.to_csv(ART / "policy_holdout_sensitivity.csv", index=False)

    chosen = table[table["policy"] == str(choice["chosen_level"])]
    if chosen.empty:
        chosen = table[table["level"].notna() & np.isclose(table["level"], float(choice["chosen_level"]))]
    print("chosen level:", choice["chosen_level"])
    print(chosen.to_string(index=False))
    print("\nsensitivity (net cost per step):")
    pivot = sens.pivot_table(index="cost_ratio", columns="policy", values="net_cost_per_step")
    print(pivot.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
