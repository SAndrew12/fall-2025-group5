import pandas as pd
from data_loader import load_data
from feature_eng import feature_creating
#from classical_models import

#Load data
df = load_data()

#Pre-proccess data + features
working_df , unattrib_df = feature_creating(df)
print(working_df.shape)
print(unattrib_df.shape)
#looks good

#Classical models



