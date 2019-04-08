#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:01:39 2019

Temporary solution to create some synthetic data for Hamburg.
To be replaced when better data are available.

@author: doorleyr
"""

import pandas as pd
import json
import random
from shapely.geometry import shape
import numpy as np


BOSTON_SYNTHPOP_PATH='./Hamburg/raw/synth_pop_Boston.csv'
ZONES_RAW_PATH='./Hamburg/raw/zones.geojson'
ZONE_DATA_PATH='./Hamburg/raw/district_profiles.csv'

SYNTHPOP_PATH='./Hamburg/clean/synth_pop.csv'
CLEAN_ZONES_PATH='./Hamburg/clean/zones.geojson'

zones=json.load(open(ZONES_RAW_PATH))
synth_pop=pd.read_csv(BOSTON_SYNTHPOP_PATH)
zone_areas=[shape(f['geometry']).area for f in zones['features']]
inv_areas=[1/a for a in zone_areas]
zones_list=list(range(len(zones['features'])))

synth_pop['home_geo_index']=np.random.choice(zones_list, 1000)
synth_pop['work_geo_index']=np.random.choice(zones_list, 1000, inv_areas)
synth_pop['pop_per_sqmile_home']=synth_pop.apply(lambda row: 8000, axis=1)

for f in zones['features']:
    f['pop_per_sqmile']=8000
    
synth_pop.to_csv(SYNTHPOP_PATH, index=False)
json.dump(zones, open(CLEAN_ZONES_PATH, 'w'))