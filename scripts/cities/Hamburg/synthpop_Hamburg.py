#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:43:56 2019

@author: doorleyr
"""
import random
import pandas as pd
import json

DETROIT_SIM_POP_PATH='./cities/Detroit/clean/sim_pop.json'
ALL_ZONES_PATH='./cities/Hamburg/raw/model_area.geojson'
CLEAN_ZONES_PATH='./cities/Hamburg/clean/model_area.geojson'
OD_PATH='./cities/Hamburg/raw/Elbbrucken_flows.csv'
SIM_POP_PATH='./cities/Hamburg/clean/sim_pop.json'


sample_factor=10
# load the O-D data
# rename the geoids in OD and in shape file for clarity
# for each O-D flow, sample the appropriate number of people from the Detroit pop

od=pd.read_csv(OD_PATH, sep='\t')
detroit_sim_pop=json.load(open(DETROIT_SIM_POP_PATH))

all_zones=json.load(open(ALL_ZONES_PATH))
for i in range(len(all_zones['features'])):
    all_zones['features'][i]['properties']['geoid']=int(all_zones['features'][i]['properties']['BezirksNr'])

sim_people=[]

for ind, row in od.iterrows():
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
        
json.dump(sim_people, open(SIM_POP_PATH, 'w'))
json.dump(all_zones, open(CLEAN_ZONES_PATH, 'w'))