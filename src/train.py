from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from src.utils import load_config, ensure_dir, set_seed


def time_split(df: pd.DataFrame, time_col: str, test_size: float, seed: int):
    """If time column exists, split by time (train older, test newer)."""
    if time_col in df.columns and df[time_col].notna().any():
        df_sorted = df.sort_values(time_col)
        n_test = int(len(df_sorted) * test_size)
        test = df_sorted.iloc[-n_test:]
        train = df_sorted.iloc[:-n_test]
        return train, test
    # fallback random
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["y"])
    return train, test


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg["project"]["seed"]
    set_seed(seed)

    dataset_path = cfg["data"]["output_dataset"]
    time_col = cfg["split"]["time_col"]
    test_size = cfg["split"]["test_size"]

    df = pd.read_parquet(dataset_path)

    label = "y"
    feature_cols = [c for c in df.columns if c not in [label]]
    df = df[feature_cols + [label]].copy()

    # Identify columns
    cat_cols = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols + [label, time_col]]

    train_df, test_df = time_split(df, time_col=time_col, test_size=test_size, seed=seed)

    X_train = train_df.drop(columns=[label])
    y_train = train_df[label].astype(int)

    # Preprocessor
    pre = build_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

    # 1) Logistic Regression baseline + calibration
    logreg = LogisticRegression(
        C=float(cfg["model"]["logreg"]["C"]),
        max_iter=int(cfg["model"]["logreg"]["max_iter"]),
        class_weight="balanced",
        n_jobs=None,
    )

    logreg_pipe = Pipeline(steps=[("pre", pre), ("clf", logreg)])
    # Calibrate probabilities (Platt scaling)
    logreg_cal = CalibratedClassifierCV(logreg_pipe, method="sigmoid", cv=3)
    logreg_cal.fit(X_train, y_train)

    ensure_dir("models")
    joblib.dump(logreg_cal, cfg["outputs"]["logreg_model_path"])
    print(f"Saved: {cfg['outputs']['logreg_model_path']}")

    # 2) XGBoost challenger (optional) + calibration
    if bool(cfg["model"]["use_xgb"]):
        xcfg = cfg["model"]["xgb"]
        xgb = XGBClassifier(
            n_estimators=int(xcfg["n_estimators"]),
            max_depth=int(xcfg["max_depth"]),
            learning_rate=float(xcfg["learning_rate"]),
            subsample=float(xcfg["subsample"]),
            colsample_bytree=float(xcfg["colsample_bytree"]),
            reg_lambda=float(xcfg["reg_lambda"]),
            eval_metric="logloss",
            n_jobs=4,
            random_state=seed,
        )
        xgb_pipe = Pipeline(steps=[("pre", pre), ("clf", xgb)])
        xgb_cal = CalibratedClassifierCV(xgb_pipe, method="isotonic", cv=3)
        xgb_cal.fit(X_train, y_train)

        joblib.dump(xgb_cal, cfg["outputs"]["xgb_model_path"])
        print(f"Saved: {cfg['outputs']['xgb_model_path']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
