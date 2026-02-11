from __future__ import annotations

import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.utils import load_config, ensure_dir, save_json


def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # KS = max difference between CDFs of positives and negatives
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_pos = np.cumsum(y_true == 1) / n_pos
    cum_neg = np.cumsum(y_true == 0) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def calibration_curve_points(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    xs, ys = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        xs.append(float(y_prob[mask].mean()))
        ys.append(float(y_true[mask].mean()))
    return xs, ys


def eval_model(model, X, y, tag: str):
    prob = model.predict_proba(X)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "ks": float(ks_stat(y, prob)),
    }
    return metrics, prob


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    dataset_path = cfg["data"]["output_dataset"]
    time_col = cfg["split"]["time_col"]
    test_size = cfg["split"]["test_size"]
    metrics_path = cfg["outputs"]["metrics_path"]

    df = pd.read_parquet(dataset_path)
    df = df.dropna(subset=["y"]).copy()
    df["y"] = df["y"].astype(int)

    # time-based split (same logic as train)
    if time_col in df.columns and df[time_col].notna().any():
        df = df.sort_values(time_col)
        n_test = int(len(df) * test_size)
        test_df = df.iloc[-n_test:]
    else:
        test_df = df.sample(frac=test_size, random_state=42)

    X_test = test_df.drop(columns=["y"])
    y_test = test_df["y"].to_numpy()

    results = {}

    # LogReg
    logreg_model = joblib.load(cfg["outputs"]["logreg_model_path"])
    m1, p1 = eval_model(logreg_model, X_test, y_test, "logreg")
    results["logreg"] = m1

    # XGB (optional)
    if cfg["model"]["use_xgb"]:
        xgb_model = joblib.load(cfg["outputs"]["xgb_model_path"])
        m2, p2 = eval_model(xgb_model, X_test, y_test, "xgb")
        results["xgb"] = m2

    ensure_dir("reports")
    save_json(metrics_path, results)
    print(json.dumps(results, indent=2))

    # Calibration plot
    plt.figure()
    xs, ys = calibration_curve_points(y_test, p1, n_bins=10)
    plt.plot(xs, ys, marker="o", label="LogReg Calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    if cfg["model"]["use_xgb"]:
        xs2, ys2 = calibration_curve_points(y_test, p2, n_bins=10)
        plt.plot(xs2, ys2, marker="o", label="XGB Calibrated")
    plt.xlabel("Mean predicted PD")
    plt.ylabel("Observed default rate")
    plt.title("Calibration Curve")
    plt.legend()
    ensure_dir("reports")
    plt.savefig("reports/calibration.png", dpi=160, bbox_inches="tight")
    print("Saved: reports/calibration.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
