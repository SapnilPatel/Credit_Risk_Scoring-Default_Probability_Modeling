import joblib
import pandas as pd
import streamlit as st
import yaml

from src.score import band

st.set_page_config(page_title="Credit PD Scoring", layout="wide")

@st.cache_resource
def load_model_and_cfg():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = joblib.load(cfg["outputs"]["logreg_model_path"])
    return cfg, model

cfg, model = load_model_and_cfg()

st.title("Credit Risk PD Scoring Tool (LogReg Calibrated)")

st.write("Upload a CSV containing the same feature columns used in training (excluding `y`).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    probs = model.predict_proba(df)[:, 1]
    df_out = df.copy()
    df_out["pd"] = probs
    df_out["risk_band"] = [band(x) for x in probs]

    st.subheader("Scored Results")
    st.dataframe(df_out, use_container_width=True)

    st.download_button(
        "Download scored CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="scored.csv",
        mime="text/csv",
    )
else:
    st.info("Tip: start with a small sample (10–50 rows) to validate schema.")
