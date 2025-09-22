import pandas as pd
from data_loader import load_data



#-----Cols to Drop----
# to_drop = [
#     'event_id_cnty',
#     'disorder_type',
#     'time_precision',
#     'source',
#     'source_scale',
#     'iso',
#     'country',
#     'timestamp',
#     'geo_precision',
#     'year',
#     'region',
#     'latitude',
#     'longitude',
#     'interaction']
# #-----Cols to Drop----
#
#
# #------Groups-----
# unattrib = [
#     'Unidentified Armed Group (Afghanistan)',
#     'Taliban and/or Islamic State Khorasan Province (ISKP)'
#
# ]
#
# taliban = [
#     'Taliban',
#     'Taliban - Red Unit',
#     'Mutiny of Taliban'
# ]
#
# iskp = [
#     'Islamic State Khorasan Province (ISKP)'
# ]
#
# other = [
#     'Al Qaeda',
#     'HQN: Haqqani Network',
#     'TTP: Tehreek-i-Taliban Pakistan',
#     'LeI: Lashkar-e-Islam'
# ]
# #------Groups-----
#
#
#
# #-----Tags for 'violence against women tags'----
# violence_against_women_tags = [
#     'women targeted: government officials',
#        'women targeted: girls',
#        'women targeted: girls; women targeted: relatives of targeted groups or persons',
#        'local administrators',
#        'women targeted: government officials; women targeted: relatives of targeted groups or persons',
#        'women targeted: candidates for office',
#        'women targeted: activists/human rights defenders/social leaders',
#        'women targeted: relatives of targeted groups or persons',
#        'local administrators; women targeted: politicians',
#        'women targeted: activists/human rights defenders/social leaders; women targeted: government officials'
# ]
#-----Tags for 'violence against women tags'----

#----Load Data---
# df = load_data()
# print(df.shape)
#----Load Data---

#---------Feature Eng--------


def feature_creating(df):
    # -----Cols to Drop----
    to_drop = [
        'event_id_cnty',
        'disorder_type',
        'time_precision',
        'source',
        'source_scale',
        'iso',
        'country',
        'timestamp',
        'geo_precision',
        'year',
        'region',
        'latitude',
        'longitude',
        'interaction']
    # -----Cols to Drop----

    # ------Groups-----
    unattrib = [
        'Unidentified Armed Group (Afghanistan)',
        'Taliban and/or Islamic State Khorasan Province (ISKP)'

    ]

    taliban = [
        'Taliban',
        'Taliban - Red Unit',
        'Mutiny of Taliban'
    ]

    iskp = [
        'Islamic State Khorasan Province (ISKP)'
    ]

    other = [
        'Al Qaeda',
        'HQN: Haqqani Network',
        'TTP: Tehreek-i-Taliban Pakistan',
        'LeI: Lashkar-e-Islam'
    ]
    # ------Groups-----

    # -----Tags for 'violence against women tags'----
    violence_against_women_tags = [
        'women targeted: government officials',
        'women targeted: girls',
        'women targeted: girls; women targeted: relatives of targeted groups or persons',
        'local administrators',
        'women targeted: government officials; women targeted: relatives of targeted groups or persons',
        'women targeted: candidates for office',
        'women targeted: activists/human rights defenders/social leaders',
        'women targeted: relatives of targeted groups or persons',
        'local administrators; women targeted: politicians',
        'women targeted: activists/human rights defenders/social leaders; women targeted: government officials'
    ]
    # -----Tags for 'violence against women tags'----

    df = df.drop(columns=to_drop)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df.sort_values(by='event_date', ascending=True)
    df = df.reset_index(drop=True)

    # create a dataframe of unattributed attacks
    # to be used later
    unattrib_df = df[df['actor1'].isin(unattrib)].copy()
    unattrib_df = unattrib_df.reset_index(drop=True)

    #mapping the target vars
    mapping = {name: 0 for name in taliban}
    mapping.update({name: 1 for name in iskp})

    working_df = df.copy()

    # map to the df
    working_df = working_df[working_df['actor1'].isin(taliban + iskp)].copy()
    working_df['target'] = working_df['actor1'].map(mapping).astype(int)

    # working_df['target'] = working_df['actor1'].map(mapping).astype('Int64')

    #create violence against women tags
    working_df['violence_against_women'] = (
       (working_df['sub_event_type'] == 'Sexual violence') |
       ((working_df['tags'].isin(violence_against_women_tags)) & (working_df['tags'] != 'local administrators'))
    ).astype(int)

    working_df = working_df.reset_index(drop=True)
    col = "civilian_targeting"

    working_df[col] = working_df[col].notna().astype(int)



    working_df = working_df.drop(index=[2300,28966,31115])


    encoded_cols = ['sub_event_type']
    working_df = pd.get_dummies(working_df, columns=encoded_cols, dtype=int)




    return working_df, unattrib_df

#---------Feature Eng--------

# working_df  = feature_creating(df)
#
# print(working_df.shape)
