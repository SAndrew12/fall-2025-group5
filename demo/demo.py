import streamlit as st

st.title("This works!")
st.write("Your Streamlit server is running on EC2 and reachable on port 8888.")

import streamlit as st
import pandas as pd
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "src" / "experiment_models"
DEMO_DIR = PROJECT_ROOT / "demo"

def main():
    st.title("Demo Loader")

    # ---- Check saved models ----
    model_files = list(MODELS_DIR.glob("*.pt"))

    if not model_files:
        st.error("No .pt model files found in src/experiment_models/")
    else:
        st.success(f"Found {len(model_files)} saved models!")
        st.write("Model files:")
        for f in model_files:
            st.write("- ", f.name)

    # ---- Load sample CSV ----
    sample_csv = DEMO_DIR / "sample.csv"

    if not sample_csv.exists():
        st.error(f"sample.csv not found in {DEMO_DIR}")
    else:
        df = pd.read_csv(sample_csv)
        st.success("Loaded sample.csv successfully!")
        st.write(df.head())

    st.write("If you see model names and sample.csv, everything works.")

if __name__ == "__main__":
    main()

