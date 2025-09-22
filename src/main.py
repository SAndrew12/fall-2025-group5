import pandas as pd
from data_loader import load_data
from feature_eng import feature_creating
from classical_models import t_t_s

#Load data
df = load_data()

#Pre-proccess data + features
working_df  = feature_creating(df)
print(working_df.shape)

#looks good


print(working_df['target'].value_counts())
print(working_df['target'].unique())
#Classical models

# X_train, X_test, y_train, y_test = t_t_s(working_df)
#
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
