#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:43:56 2019

@author: doorleyr
"""
import pandas as pd
import json
import numpy as np

DETROIT_VACANT_PATH='scripts/cities/Detroit/clean/vacant.json'
DETROIT_FLOATING_PATH='scripts/cities/Detroit/clean/floating.json'
ALL_ZONES_PATH='scripts/cities/Hamburg/raw/model_area.geojson'
CLEAN_ALL_ZONES_PATH='scripts/cities/Hamburg/clean/model_area.geojson'
SIM_ZONES_PATH='scripts/cities/Hamburg/raw/sim_area.geojson'
CLEAN_SIM_ZONES_PATH='scripts/cities/Hamburg/clean/sim_area.geojson'
OD_PATH='scripts/cities/Hamburg/raw/Elbbrucken_flows.csv'
SIM_POP_PATH='scripts/cities/Hamburg/clean/sim_pop.json'
VACANT_PATH='scripts/cities/Hamburg/clean/vacant.json'
FLOATING_PATH='scripts/cities/Hamburg/clean/floating.json'

od=pd.read_csv(OD_PATH, sep='\t')
detroit_floating_persons=json.load(open(DETROIT_FLOATING_PATH))
detroit_vacant_houses=json.load(open(DETROIT_VACANT_PATH))

all_zones=json.load(open(ALL_ZONES_PATH))
sim_zones=json.load(open(SIM_ZONES_PATH))

for i in range(len(all_zones['features'])):
    all_zones['features'][i]['properties']['GEO_ID']=int(
            all_zones['features'][i]['properties']['BezirksNr'])
for i in range(len(sim_zones['features'])):
    sim_zones['features'][i]['properties']['GEO_ID']=int(
            sim_zones['features'][i]['properties']['BezirksNr'])
all_zone_nums=[f['properties']['GEO_ID'] for f in all_zones['features']]

sim_people=[]

# assign home lcations to the houses based on 
# distribution of home locations for people working in 234
# assign work locations based on 
# distributions of work locations for people living in 234
home_loc_dist=od.loc[((od['Nach']==234) & (od['Von'].isin(all_zone_nums))), 
                      ['Von', 'Nr. 1007']]
home_flows=[float(home_loc_dist.iloc[i]['Nr. 1007'].replace(
        ',', '.')) for i in range(len(home_loc_dist))]
home_weights=[f/sum(home_flows) for f in home_flows]

work_loc_dist=od.loc[((od['Von']==234) & (od['Nach'].isin(all_zone_nums))), 
                     ['Nach', 'Nr. 1007']]
work_flows=[float(work_loc_dist.iloc[i]['Nr. 1007'].replace(
        ',', '.')) for i in range(len(work_loc_dist))]
work_weights=[f/sum(work_flows) for f in work_flows]

vacant_houses = detroit_vacant_houses     
for house in vacant_houses:
    house['home_geoid']=int(np.random.choice(home_loc_dist['Von'].values,
                 1, p=home_weights)[0])
    
floating_persons=detroit_floating_persons
for persons in floating_persons:
    persons['work_geoid']=int(np.random.choice(work_loc_dist['Nach'].values,
                 1, p=work_weights)[0])
        
json.dump(sim_people, open(SIM_POP_PATH, 'w'))
json.dump(floating_persons, open(FLOATING_PATH, 'w'))
json.dump(vacant_houses, open(VACANT_PATH, 'w'))
json.dump(all_zones, open(CLEAN_ALL_ZONES_PATH, 'w'))
json.dump(sim_zones, open(CLEAN_SIM_ZONES_PATH, 'w'))