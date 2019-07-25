#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:43:56 2019

@author: doorleyr
"""
import random
import pandas as pd
import json
import numpy as np

DETROIT_SIM_POP_PATH='../Detroit/clean/sim_pop.json'
DETROIT_VACANT_PATH='../Detroit/clean/vacant.json'
DETROIT_FLOATING_PATH='../Detroit/clean/floating.json'
ALL_ZONES_PATH='./raw/model_area.geojson'
SIM_ZONES_PATH='./raw/sim_area.geojson'
CLEAN_ALL_ZONES_PATH='./clean/model_area.geojson'
CLEAN_SIM_ZONES_PATH='./clean/sim_area.geojson'
OD_PATH='./raw/Elbbrucken_flows.csv'
SIM_POP_PATH='./clean/sim_pop.json'
VACANT_PATH='./clean/vacant.json'
FLOATING_PATH='./clean/floating.json'


sample_factor=10
# load the O-D data
# rename the geoids in OD and in shape file for clarity
# for each O-D flow, sample the appropriate number of people from the Detroit pop

od=pd.read_csv(OD_PATH, sep='\t')
detroit_sim_pop=json.load(open(DETROIT_SIM_POP_PATH))
detroit_floating_persons=json.load(open(DETROIT_FLOATING_PATH))
detroit_vacant_houses=json.load(open(DETROIT_VACANT_PATH))

all_zones=json.load(open(ALL_ZONES_PATH))
sim_zones=json.load(open(SIM_ZONES_PATH))
for i in range(len(all_zones['features'])):
    all_zones['features'][i]['properties']['GEO_ID']=int(
            all_zones['features'][i]['properties']['BezirksNr'])
all_zone_nums=[f['properties']['GEO_ID'] for f in all_zones['features']]
for i in range(len(sim_zones['features'])):
    sim_zones['features'][i]['properties']['GEO_ID']=int(
            sim_zones['features'][i]['properties']['BezirksNr'])

sim_people=[]

for ind, row in od.iterrows():
    if ((row['Nach'] in all_zone_nums) and (row['Von'] in all_zone_nums)):
        people_to_sample=float(row['Nr. 1007'].replace(',', '.'))/sample_factor
        people_to_sample_int=int(people_to_sample)
        if people_to_sample%1>random.uniform(0, 1):
            people_to_sample_int+=1
        if people_to_sample_int>0:
            sample_people=random.sample(detroit_sim_pop, people_to_sample_int).copy()
            for p in sample_people:
                p['home_geoid']=row['Von']
                p['work_geoid']=row['Nach']
            sim_people.extend(sample_people.copy())

# assign home lcations to the houses based on 
# distribution of home locations for people working in 234
# assign work locations based on 
# distributions of work locations for people living in 234
home_loc_dist=od.loc[od['Nach']==234, ['Von', 'Nr. 1007']]
home_flows=[float(home_loc_dist.iloc[i]['Nr. 1007'].replace(
        ',', '.')) for i in range(len(home_loc_dist))]
home_weights=[f/sum(home_flows) for f in home_flows]

work_loc_dist=od.loc[od['Von']==234, ['Nach', 'Nr. 1007']]
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