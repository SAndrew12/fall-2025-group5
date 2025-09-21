import pandas as od

#Load data
df = load_data()

#Pre-proccess data + features
working_df , unattrib_df = feature_creating(df)
#Classical Models