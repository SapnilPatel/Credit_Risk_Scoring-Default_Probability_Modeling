from __future__ import annotations

import argparse
import joblib
import pandas as pd

from src.utils import load_config

RISK_BANDS = [
    ("A", 0.00, 0.02),
    ("B", 0.02, 0.05),
    ("C", 0.05, 0.10),
    ("D", 0.10, 0.20),
    ("E", 0.20, 1.01),
]


def band(pd_value: float) -> str:
    for name, lo, hi in RISK_BANDS:
        if lo <= pd_value < hi:
            return name
    return "NA"


def main(config_path: str, input_csv: str, output_csv: str, model_name: str) -> None:
    cfg = load_config(config_path)

    model_path = cfg["outputs"]["logreg_model_path"] if model_name == "logreg" else cfg["outputs"]["xgb_model_path"]
    model = joblib.load(model_path)

    df = pd.read_csv(input_csv)
    # IMPORTANT: df must contain the same raw feature columns as training (minus y)
    prob = model.predict_proba(df)[:, 1]
    out = df.copy()
    out["pd"] = prob
    out["risk_band"] = [band(x) for x in prob]
    out.to_csv(output_csv, index=False)
    print(f"Saved scored file -> {output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--input", required=True, help="CSV of loans/applicants with training columns")
    ap.add_argument("--output", default="data/processed/scored.csv")
    ap.add_argument("--model", default="logreg", choices=["logreg", "xgb"])
    args = ap.parse_args()
    main(args.config, args.input, args.output, args.model)
