import streamlit as st

st.title("This works!")
st.write("Your Streamlit server is running on EC2 and reachable on port 8888.")

import streamlit as st
import pandas as pd
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = PROJECT_ROOT / "src" / "saved_models"
DEMO_DIR = PROJECT_ROOT / "demo"

def main():
    st.title("Demo Loader")



    st.write(f"Looking for models in: `{SAVED_MODELS_DIR}`")

    @st.cache_resource
    def load_all_models():
        """
        Returns a dict of:
            { model_type: [list of filenames] }
        """
        model_dict = {}

        if not SAVED_MODELS_DIR.exists():
            return {}

        for subfolder in SAVED_MODELS_DIR.iterdir():
            if subfolder.is_dir():
                files = [f.name for f in subfolder.iterdir() if f.is_file()]
                model_dict[subfolder.name] = files

        print("=== Loaded Models ===")
        print(model_dict)

        return model_dict

    models = load_all_models()

    if not models:
        st.error("❌ No models found in saved_models/")
    else:
        total = sum(len(v) for v in models.values())
        st.success(f"Found {total} saved models!")

        st.subheader("Model files:")
        for model_type, files in models.items():
            st.markdown(f"### 📂 {model_type}")
            for f in files:
                st.write(f"- {f}")

    # ---- Load sample CSV ----
    sample_csv = DEMO_DIR / "unattributed_attacks_processed.csv"

    if not sample_csv.exists():
        st.error(f"sample.csv not found in {DEMO_DIR}")
    else:
        df = pd.read_csv(sample_csv)
        st.success("Loaded sample.csv successfully!")
        st.write(df.head())

    st.write("If you see model names and sample.csv, everything works.")

if __name__ == "__main__":
    main()

