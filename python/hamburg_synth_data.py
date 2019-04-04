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

BOSTON_SYNTHPOP_PATH='./Hamburg/raw/synth_pop_Boston.csv'
ZONES_RAW_PATH='./Hamburg/raw/gis_osm_landuse_a_free_1.geojson'

SYNTHPOP_PATH='../ABM/includes//Hamburg/synth_pop.csv'
CLEAN_ZONES_PATH='../ABM/includes/Hamburg/zones.geojson'

zones=json.load(open(ZONES_RAW_PATH))
synth_pop=pd.read_csv(BOSTON_SYNTHPOP_PATH)

zones['features']=[f for f in zones['features'] if f['properties']['fclass'] in ['residential', 'commercial', 'industrial']]
zones['crs']= {"properties": {"name": "EPSG:4326"}, "type": "name"}

res_zone_nums=[i for i in range(len(zones['features'])) 
if zones['features'][i]['properties']['fclass']=='residential']

work_zone_nums=[i for i in range(len(zones['features'])) 
if zones['features'][i]['properties']['fclass']in ['commercial', 'industrial']]

synth_pop['home_geo_index']=random.sample( res_zone_nums,len(synth_pop))
synth_pop['work_geo_index']=random.sample( work_zone_nums,len(synth_pop))

for f in zones['features']:
    f['pop_per_sqmile']=15000
    
synth_pop.to_csv(SYNTHPOP_PATH)
json.dump(zones, open(CLEAN_ZONES_PATH, 'w'))