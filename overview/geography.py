'''

'''

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ptitprince as pt 
import seaborn as sns
import geopandas as gpd
import os 
import re
pd.set_option('display.max_colwidth', None)

# geopandas loads
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

# match with our data
data_raw = pd.read_csv('../data/raw/drh_20221019.csv')
data_geography = data_raw[['entry_name', 'entry_id', 'region_id']].drop_duplicates()

# import kml files 
regex_region = r"(\d+).kml"
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
files = os.listdir('../data/kml')
geography_information = []
for file in files: 
    region_id = int(re.search(regex_region, file).group(1))
    d = gpd.read_file(f'../data/kml/{file}', driver='KML')
    d['region_id'] = region_id
    geography_information.append(d)
geography_information = pd.concat(geography_information)
geography_information = geography_information[['Name', 'geometry', 'region_id']]
geography_centroid = geography_information.assign(geometry = lambda x: x['geometry'].centroid)

# clearly something wrong with the Unitarian Univeralist Organizations y-coordinate
base = world.plot(color = 'white', edgecolor = 'black')
geography_centroid.plot(ax=base, marker='o', color='red', markersize=5)
plt.show();

# remove the weird point (n = 626)
geography_filtered = geography_centroid[geography_centroid['Name'] != 'Unitarian Univeralist Organizations']
base = world.plot(color = 'white', edgecolor = 'black')
geography_filtered.plot(ax=base, marker='o', color='red', markersize=5)
plt.show();

# merge with our full data set (n = 755 so some overlap)
geography_merge = geography_filtered.merge(data_geography, on = 'region_id', how = 'inner')
base = world.plot(color = 'white', edgecolor = 'black')
geography_merge.plot(ax=base, marker='o', color='red', markersize=5)
plt.show();

# merge with the subset that we use in the Entropy paper (n = 370)
entry_reference = pd.read_csv('../data/preprocessing/entry_reference.csv')
geography_sub = geography_merge.merge(entry_reference, on = ['entry_id', 'entry_name'], how = 'inner')
base = world.plot(color = 'white', edgecolor = 'black')
geography_sub.plot(ax=base, marker='o', color='red', markersize=5)
plt.show();

# only 370 entries with regions but we have 407 total entries
# which ones do not have a region? 
list(set(set(data_geography['entry_name'])) - set(set(geography_sub['entry_name'])))

# why are we lacking region for 60 cases when 407-370 = 37?
# some entry_names 
geography_sub.groupby('entry_name').size().reset_index(name='counts').sort_values('counts', ascending=False)

# for instance the Free Methodist Church
geography_sub[geography_sub['entry_name'] == "Free Methodist Church"]