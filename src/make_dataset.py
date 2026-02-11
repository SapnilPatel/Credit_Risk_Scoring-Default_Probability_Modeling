from __future__ import annotations

import argparse
import duckdb
import pandas as pd

from src.utils import load_config, ensure_dir


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    raw_csv = cfg["data"]["raw_csv"]
    duckdb_path = cfg["data"]["duckdb_path"]
    table_name = cfg["data"]["table_name"]
    features_table = cfg["data"]["features_table"]
    output_dataset = cfg["data"]["output_dataset"]
    sql_path = "sql/build_features.sql"

    ensure_dir("data/processed")

    con = duckdb.connect(duckdb_path)

    # Load CSV into DuckDB table
    con.execute(f"DROP TABLE IF EXISTS {table_name};")
    con.execute(
        f"""
        CREATE TABLE {table_name} AS
        SELECT * FROM read_csv_auto('{raw_csv}', ALL_VARCHAR=TRUE);
        """
    )

    # Build features table using SQL
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    con.execute(sql)

    # Export to parquet for ML pipeline
    df = con.execute(f"SELECT * FROM {features_table};").fetchdf()

    # Convert numeric columns safely (DuckDB read as VARCHAR via ALL_VARCHAR)
    # Try to cast everything except categoricals/time/label
    cat_cols = ["grade", "sub_grade", "home_ownership", "verification_status", "purpose"]
    time_col = "issue_dt"
    label_col = "y"
    for col in df.columns:
        if col in cat_cols or col in [time_col, label_col]:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # issue_dt is already datetime if parsing worked; coerce if needed
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # minimal cleaning
    df = df.dropna(subset=[label_col])
    # Keep rows with enough signal; drop rows that have too many nulls
    df = df.dropna(thresh=max(3, int(0.7 * len(df.columns))))

    ensure_dir("data/processed")
    df.to_parquet(output_dataset, index=False)

    print(f"Saved dataset: {output_dataset} | rows={len(df):,} cols={len(df.columns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
