import pandas as pd
import os

def load_data():
    filename = 'ACLED Data_2025-09-11.csv'
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "src", "data", filename)
    df = pd.read_csv(data_path)

    return df
# will add more as time goes on