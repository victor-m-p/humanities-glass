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
pd.set_option('display.max_colwidth', None)

# geopandas loads
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))

# match with our data
data_raw = pd.read_csv('../data/raw/drh_20221019.csv')
data_geography = data_raw[['entry_name', 'entry_id', 'region_name', 'region_id', 'region_tags', 'region_desc']].drop_duplicates()

# which column are we supposed to match the "Name"
# column that the .kml files have (here geography_information) on?


#entry_reference = pd.read_csv('../data/analysis/entry_reference.csv')
#data_geography = data_raw[['entry_name', 'entry_id', 'region_name' 'region_id', 'region_tags', 'region_desc']]
data_raw.dtypes
# import data
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
files = os.listdir('../data/kml_files')
geography_information = []
for file in files: 
    d = gpd.read_file(f'../data/kml_files/{file}', driver='KML')
    geography_information.append(d)
geography_information = pd.concat(geography_information)
geography_information = geography_information[['Name', 'geometry']]
geography_centroid = geography_information.assign(geometry = lambda x: x['geometry'].centroid)

# plot 
base = world.plot(color = 'white', edgecolor = 'black')
geography_centroid.plot(ax=base, marker='o', color='red', markersize=5)
plt.show();