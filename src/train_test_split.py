#---Imports----
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#---Imports----
random_seed = 42
np.random.seed(random_seed)



# ---- tain,test,split ----
def t_t_s(working_df):
    feature_cols = [
        'civilian_targeting','fatalities','violence_against_women',
        'sub_event_type_Abduction/forced disappearance',
        'sub_event_type_Air/drone strike',
        'sub_event_type_Armed clash',
        'sub_event_type_Attack',
        'sub_event_type_Government regains territory',
        'sub_event_type_Grenade',
        'sub_event_type_Non-state actor overtakes territory',
        'sub_event_type_Remote explosive/landmine/IED',
        'sub_event_type_Sexual violence',
        'sub_event_type_Shelling/artillery/missile attack',
        'sub_event_type_Suicide bomb'
    ]
    X = working_df[feature_cols]
    y = working_df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=random_seed,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
