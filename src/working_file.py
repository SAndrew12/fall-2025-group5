#%%
import pandas as pd
from pprint import pprint
import folium
#from html5lib.constants import unadjustForeignAttributes
from overrides.typing_utils import unknown


import re







random_seed = 42
import numpy as np
np.random.seed(random_seed)

import os
from pathlib import Path
#%%

import pandas as pd
df = pd.read_csv("C:/Users/steph/Desktop/GW/Semester 3/Capstone/repo/fall-2025-group5/src/data/ACLED Data_2025-09-11.csv")




#%%
#process data 1

# # Dates
#
# df['event_date'] = pd.to_datetime(df['event_date'])
# # Sort by date ascending (oldest → newest)
# df = df.sort_values(by='event_date', ascending=True)
# # Reset index if you want clean numbering
# df = df.reset_index(drop=True)

#%%
# process data 2 - split groups

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

# create hold out of just unattrib attacks

unattrib_df = df[df['actor1'].isin(unattrib)].copy()
unattrib_df = unattrib_df.reset_index(drop=True)

# create ISKP/Tali/Other dataframe

df["target"] = df["actor1"].apply(
    lambda x: 1 if x in iskp
              else 2 if x in  taliban
              else 3 if x in other
              else None
)

# Create holdout df
working_df = df[df['actor1'].isin(iskp + taliban + other)].copy()

working_df.shape
working_df['actor1'].value_counts()

#%%

# process data 3
 #create violence against women flag



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
working_df['violence_against_women'] = (
    (working_df['sub_event_type'] == 'Sexual violence') |
    ((working_df['tags'].isin(violence_against_women_tags)) & (working_df['tags'] != 'local administrators'))
).astype(int)

#%%
working_df = working_df.reset_index(drop=True)
#%%
working_df.columns

#%%

working_df['sub_event_type'].unique()

#%%

y = working_df[working_df['sub_event_type'] == 'Air/drone strike']

look = y[y['interaction'] == 'State forces-Rebel group']
look.to_csv('test.csv')
#%%
# import folium
# import pandas as pd
#
# # --- 1. Load data ---
# df = pd.read_csv(r"C:\Users\steph\PycharmProjects\data_vis\Capstone\data\ACLED Data_2025-09-11.csv")
#
# # --- 2. Subset actors of interest ---
# actors_of_interest = [
#     'Taliban',
# ,    'HQN: Haqqani Network'
#     'Unidentified Armed Group (Afghanistan)',
#     'Islamic State Khorasan Province (ISKP)'
# ]
# df_subset = df[df['actor1'].isin(actors_of_interest)].copy()
#
# # --- 3. Merge Taliban + HQN into "Taliban/AQ" ---
# df_subset['actor_group'] = df_subset['actor1'].replace({
#     'Taliban': 'Taliban/AQ',
#     'HQN: Haqqani Network': 'Taliban/AQ',
#     'Unidentified Armed Group (Afghanistan)': 'Unidentified Armed Group',
#     'Islamic State Khorasan Province (ISKP)': 'ISKP'
# })
#
# print(df_subset['actor_group'].value_counts())
#
# # --- 4. Initialize map ---
# m = folium.Map(location=[33.9391, 67.7100], zoom_start=6, tiles="cartodbpositron")
#
# # --- 5. Color map for groups ---
# color_map = {
#     'Taliban/AQ': 'red',
#     'Unidentified Armed Group': 'blue',
#     'ISKP': 'purple'
# }
#
# # --- 6. Create a FeatureGroup per actor_group ---
# groups = {}
# for g in sorted(df_subset['actor_group'].dropna().unique()):
#     fg = folium.FeatureGroup(name=g, show=True)   # set show=False to start hidden
#     fg.add_to(m)
#     groups[g] = fg
#
# # --- 7. Add ONLY CircleMarkers to their group layers ---
# for _, row in df_subset.iterrows():
#     g = row['actor_group']
#     lat, lon = float(row['latitude']), float(row['longitude'])
#     color = color_map.get(g, 'black')
#     folium.CircleMarker(
#         location=[lat, lon],
#         radius=4,
#         color=color,
#         weight=1,
#         fill=True,
#         fill_color=color,
#         fill_opacity=0.7,
#         popup=f"{g} ({row.get('event_date', 'date?')})"
#     ).add_to(groups[g])
#
# # --- 8. Optional: fit bounds to your data ---
# if not df_subset.empty:
#     sw = [df_subset['latitude'].min(), df_subset['longitude'].min()]
#     ne = [df_subset['latitude'].max(), df_subset['longitude'].max()]
#     m.fit_bounds([sw, ne])
#
# # --- 9. Layer control to toggle groups on/off ---
# folium.LayerControl(collapsed=False).add_to(m)
#
# # --- 10. Save map ---
# m.save("afghanistan_conflict_map2.html")
# print('saved')

#%%

