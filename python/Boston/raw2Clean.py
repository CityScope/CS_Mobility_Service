#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:39:42 2019

creates baseline synthetic population as csv
creates potential population as csv
creates clean zone geojson with correct projection
prepares clean tour file for model calibration

@author: doorleyr
"""
import pandas as pd
import numpy as np
import json
import re
from collections import OrderedDict

# =============================================================================
# Functions
# =============================================================================

def get_NHTS_motif(u_id):
    personRecords=nhts_tour.loc[nhts_tour['uniquePersonId']==u_id]
    if len(personRecords)>0:
        sched=''.join(list(personRecords['TOURTYPE']))
        for r in reversed(regPatterns):
        #checking the most complex first
            if r.match(sched):
                return [sched, regPatterns[r]]
        return [sched, 'NA']
    return ['H', 'H']


# =============================================================================
# Constants
# =============================================================================
# paths to data files
NHTS_PATH='../NHTS/perpub.csv'
NHTS_TOUR_PATH='../NHTS/tour17.csv'
NHTS_TRIP_PATH='../NHTS/trippub.csv'
OD_PATH='raw/CPTT/tract2tractCommutingAllMass.csv'
INCOME_PATH='raw/census/ACS_16_5YR_B19001/ACS_16_5YR_B19001_with_ann.csv'
ZONE_LIST_PATH='raw/zones/zones_included.csv'
ZONE_SHAPE_PATH='raw/zones/tractsMass.geojson'
MODE_DICT_PATH='../NHTS/modeDict.json'
# paths for saving clean data
CLEAN_SHP_PATH='../../MoBalance_ABM/includes/Boston/zones.geojson'
SYNTH_POP_PATH='../../MoBalance_ABM/includes/Boston/synth_pop.csv'
#SYNTH_POP_PATH='clean/synth_pop.json'
POT_POP_PATH='clean/pot_pop.csv'
MODE_TABLE_PATH='clean/trip_modes.csv'
TOUR_TABLE_PATH='clean/tours.csv'
POT_POP_BY_JOB_PATH='../../MoBalance_ABM/includes/Boston/job_type_'
# preparing choice tables
ANNUAL_CAR_COST=5635
ANNUAL_BIKE_COST=200
PER_KM_CAR_COST=0.11
DAYS_PER_YEAR=221
COST_PT=2.25
# population size
N=200
#TODO Do below in an input file as it will be used elsewhere
# conditions for regional subset of the NHTS data
#New England (ME, NH, VT, CT, MA, RI) MSA or CMSA of 1 million or more with heavy rail
CDIVMSAR=11

nhts_to_simple_mode={
        # map NHTS modes to a simpler list of modes
        # 0: drive, 1: cycle, 2: walk, 3: PT
        -7:-99,-8:-99,-9:-99,
        1:2,
        2:1,
        3:0,4:0,5:0,6:0,
        7:-99,
        8:0,9:0,
        10:3,11:3,12:3,13:3,14:3,15:3,16:3,
        17:0,18:0,
        19:3,20:3,
        97:-99}

regPatterns=OrderedDict()
#define the motifs and create regular expressions to represent them
# H* means 0 or more H. W+ means 1 or more W
# TODO rethink this. simpler motifs based on non-home activites
regPatterns[re.compile('H+')]='H' 
regPatterns[re.compile('H*W+')]='HWH'
regPatterns[re.compile('H*O+')]='HOH'
regPatterns[re.compile('H*O+W+')]='HOWH'
regPatterns[re.compile('H*W+H*O+')]='HWOH'
regPatterns[re.compile('H*O+H*W+H*O+')]='HOWOH'
regPatterns[re.compile('H*W+H*O+H*W+')]= 'HWOWH'
       
#********************************************
#      Data Files
#********************************************

nhts_per=pd.read_csv(NHTS_PATH)
nhts_tour=pd.read_csv(NHTS_TOUR_PATH)
nhts_trip=pd.read_csv(NHTS_TRIP_PATH)
od=pd.read_csv(OD_PATH, skiprows=2)
income=pd.read_csv(INCOME_PATH, index_col='Id2', skiprows=1)
zones_shp=json.load(open(ZONE_SHAPE_PATH))
zones_list=list(pd.read_csv(ZONE_LIST_PATH).geoIds)
mode_dict=json.load(open(MODE_DICT_PATH)) #from cptt mode defs to nhts mode defs

# Clean zone data and take a subset of the zones
zones_shp['features']=[f for f in zones_shp['features'] if int(f['properties']['GEOID10']) in zones_list]
zones_shp['crs']['properties']['name']='EPSG:4326'
zones_order=[f['properties']['GEOID10'] for f in zones_shp['features']]
#********************************************
#      Clean up the O-D data
#********************************************
od['RESIDENCE']=od.apply(lambda row: str(row['RESIDENCE']).split(',')[0], axis=1)
od['WORKPLACE']=od.apply(lambda row: str(row['WORKPLACE']).split(',')[0], axis=1)
# remove the '000 separator ','
od['Workers 16 and Over']=od.apply(lambda row: float(str(row['Workers 16 and Over']).replace(',',"")), axis=1)
od=od.rename(columns={'Means of Transportation 18':'mode', 'Workers 16 and Over':'workers'})
# Change the names in CPTT to tract geoIds (Use the geojson which contains both)
names_to_geoIds={f['properties']['NAMELSAD10']:f['properties']['GEOID10'] for f in zones_shp['features']}
od['home_GEOID']=od.apply(lambda row: names_to_geoIds[row['RESIDENCE']] if row['RESIDENCE'] in names_to_geoIds else float('nan'), axis=1)
od['work_GEOID']=od.apply(lambda row: names_to_geoIds[row['WORKPLACE']] if row['WORKPLACE'] in names_to_geoIds else float('nan'), axis=1)
# only keep trips which start and end in the relevant zones
od=od.dropna(how='any', subset=['home_GEOID', 'work_GEOID']).reset_index(drop=True)
#get rid of the 'Total' rows
od=od.loc[od['mode'].isin(mode_dict)]
# sample the required number of people
sample_ind=od.sample(n=N, weights=od['workers'])[['home_GEOID', 'work_GEOID', 'mode']]

#********************************************
#      Clean up the NHTS data
#      create and/or rename variables which will be used 
#      in the populations and/or model fitting
#********************************************
# need to cbsa information in the tour fie for later.
# look up from person file using HOUSE_ID
nhts_tour=nhts_tour.merge(nhts_per[['HOUSEID', 'HH_CBSA']], on='HOUSEID', how='left')
nhts_tour['uniquePersonId']=nhts_tour.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
nhts_trip['uniquePersonId']=nhts_trip.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)
nhts_per['uniquePersonId']=nhts_per.apply(lambda row: str(row['HOUSEID'])+'_'+str(row['PERSONID']), axis=1)

# person file: used for synth pop and calibration data for main mode model
# only keep workers over age of 16
nhts_per=nhts_per.loc[nhts_per['R_AGE_IMP']>15]
#only keep workers
nhts_per=nhts_per.loc[nhts_per['OCCAT']>=0]
nhts_per=nhts_per.loc[nhts_per['OCCAT']<97]
#remove records with unknown variables that we need
nhts_per=nhts_per.loc[nhts_per['HHFAMINC']>=0]
nhts_per=nhts_per.loc[nhts_per['LIF_CYC']>=0]
#only keep people whose travel diary was on a weekday
#nhts_per_zone=nhts_per_zone.loc[nhts_per_zone['TRAVDAY'].isin([2,3,4,5,6])]

#rename and create more informative variables
nhts_per=nhts_per.rename(columns={'R_AGE_IMP': 'age',
                                  'HTPPOPDN': 'pop_per_sqmile_home', 
                                  'HHFAMINC':'hh_income'})
# categorical variables must be one-hot-encoded for the model fitting and prediction
newDummies=pd.get_dummies(nhts_per['OCCAT'], prefix='job_type')
nhts_per=pd.concat([nhts_per, newDummies],  axis=1)
nhts_per['male']=nhts_per.apply(lambda row: row['R_SEX_IMP']==1, axis=1)
nhts_per['bachelor_degree']=nhts_per.apply(lambda row: row['EDUC']>=4, axis=1)
# main mode column renamed as main_mode
nhts_per['main_mode']=nhts_per.apply(lambda row: nhts_to_simple_mode[row['WRKTRANS']], axis=1)
# select people in relevant area
nhts_per_zone=nhts_per.loc[nhts_per['CDIVMSAR']==CDIVMSAR]
nhts_per_zone['motif']=nhts_per_zone.apply(lambda row: get_NHTS_motif(row['uniquePersonId'])[1], axis=1)
#********************************************
#      Create the synthetic population
#********************************************
# iterate through modal O-D and select people from NHTS based on 
# (i) conditional on mode and (ii) bayes rule on income
synth_pop=[]
#TODO smple with probability according to income distribution
#TODO add motifs and modes to synth_pop
pop_cols=['main_mode', 'age', 'hh_income', 'pop_per_sqmile_home', 'male', 'bachelor_degree', 'motif']
for i, row in sample_ind.iterrows():
    if row['mode']=='Worked at home':
        candidates=nhts_per_zone.loc[nhts_per_zone['WRK_HOME']==1]
    else:
        candidates=nhts_per_zone.loc[nhts_per_zone['WRKTRANS'].isin(mode_dict[row['mode']])]
        if len(candidates)==0: # in case there's no surveyed individuals with this mode
            print('global sample')
            candidates=nhts_per.loc[nhts_per['WRKTRANS'].isin(mode_dict[row['mode']])]    
    selection=candidates.sample(1)
    new_person={pc: str(selection.iloc[0][pc]) for pc in pop_cols}
    new_person['male']=int(new_person['male']=='True')
    new_person['bachelor_degree']=int(new_person['bachelor_degree']=='True')
    new_person['home_GEOID']=row['home_GEOID']
    new_person['work_GEOID']=row['work_GEOID']
    new_person['home_geo_index']=zones_order.index(new_person['home_GEOID'])
    new_person['work_geo_index']=zones_order.index(new_person['work_GEOID'])
    synth_pop.append(new_person)
synth_pop_df=pd.DataFrame(synth_pop)

#********************************************
#      Create the potential population
#********************************************        
pot_pop=nhts_per_zone[['age', 'hh_income', 'male', 'bachelor_degree','job_type_1',
 'job_type_2','job_type_3','job_type_4', 'motif']]
# TODO the housing requirements should be based on PUMS, Bayes Net
pot_pop['res_type'] = np.random.randint(0, 8, pot_pop.shape[0])

# =============================================================================
# Prepare data for model fitting
# The person file data will be used for the main mode and motif models
# The trips file will be used for the trip mode model
# =============================================================================

# create empty df
mode_table=pd.DataFrame()
# rename the columns 'main_mode', 'age', 'hh_income', 'pop_per_sqmile_home', 'male', 'bachelor_degree'
nhts_trip=nhts_trip.rename(columns={'R_AGE_IMP': 'age',
                                  'HTPPOPDN': 'pop_per_sqmile_home', 
                                  'HHFAMINC':'hh_income'})
nhts_trip['male']=nhts_trip.apply(lambda row: row['R_SEX_IMP']==1, axis=1)
nhts_trip['bachelor_degree']=nhts_trip.apply(lambda row: row['EDUC']>=4, axis=1)
# main mode column renamed as main_mode. if not there add them from person file
nhts_trip['mode']=nhts_trip.apply(lambda row: nhts_to_simple_mode[row['TRPTRANS']], axis=1)
# remove rows with mode<0 or other errors
nhts_trip=nhts_trip.loc[nhts_trip['mode']>=0]
nhts_trip.loc[nhts_trip['TRPMILES']<0, 'TRPMILES']=0 # -9 for work-from-home
# set the network_dist_km column
nhts_trip['network_dist_km']=nhts_trip.apply(lambda row: row['TRPMILES']/1.62, axis=1)
speeds={c:{} for c in set(nhts_per['HH_CBSA'])}
nhts_tour['main_mode']=nhts_tour.apply(lambda row: nhts_to_simple_mode[row['MODE_D']], axis=1)

# hypothetical travel times for the modes which were NOT taken by each person
# are not available from the data. Need to estimate how long the trip would have taken 
# by the alternate modes
for c in speeds:
    this_cbsa=nhts_tour[nhts_tour['HH_CBSA']==c]
    for m in [0,1,2, 3]:
        all_speeds=this_cbsa.loc[((this_cbsa['main_mode']==m) & (this_cbsa['TIME_M']>0))].apply(lambda row: row['DIST_M']/row['TIME_M'], axis=1)
        if len(all_speeds)>0:
            speeds[c]['km_per_minute_'+str(m)]=1.62* all_speeds.mean()
        else:
            speeds[c]['km_per_minute_'+str(m)]=float('nan')
    speeds[c]['walk_km_'+str(m)]=1.62*this_cbsa.loc[this_cbsa['main_mode']==3,'PMT_WALK'].mean()
    speeds[c]['drive_km_'+str(m)]=1.62*this_cbsa.loc[this_cbsa['main_mode']==3,'PMT_POV'].mean()

mode_table['drive_time']=nhts_trip.apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)
mode_table['cycle_time']=nhts_trip.apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(1)], axis=1)
mode_table['walk_time']=nhts_trip.apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
mode_table['PT_time']=nhts_trip.apply(lambda row: row['network_dist_km']/speeds[row['HH_CBSA']]['km_per_minute_'+str(3)], axis=1)
mode_table['walk_time_PT']=nhts_trip.apply(lambda row: speeds[row['HH_CBSA']]['walk_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(2)], axis=1)
mode_table['drive_time_PT']=nhts_trip.apply(lambda row: speeds[row['HH_CBSA']]['drive_km_'+str(3)]/speeds[row['HH_CBSA']]['km_per_minute_'+str(0)], axis=1)
## monetary costs
mode_table['drive_cost']=ANNUAL_CAR_COST/(DAYS_PER_YEAR*2)
mode_table['cycle_cost']=ANNUAL_BIKE_COST/(DAYS_PER_YEAR*2)
mode_table['walk_cost']=0
mode_table['PT_cost']=COST_PT
# above are must-have variables. in live environment, these will be calculated in real time
# individual-specific variables are flexible but must all be in pot_pop and synth_pop_df
mode_table[['age', 'hh_income', 'male', 'bachelor_degree', 'mode', 'pop_per_sqmile_home']]=nhts_trip[['age', 'hh_income', 'male', 'bachelor_degree', 'mode', 'pop_per_sqmile_home']]

# add motifs to the table

# delete NaN
mode_table=mode_table.dropna()
mode_table_sample=mode_table.sample(n=15000)

#nhts_tour_zone=nhts_tour.loc[nhts_tour['uniquePersonId'].isin(nhts_per_zone['uniquePersonId'])]
## add variables from person/hh files


#********************************************
#      Save the outputs
#********************************************  
# clean zones
json.dump(zones_shp, open(CLEAN_SHP_PATH, 'w'))
# synth_pop
#json.dump(synth_pop, open(SYNTH_POP_PATH, 'w'))
synth_pop_df.to_csv(SYNTH_POP_PATH, index=False)
# pot_pop
#json.dump(pot_pop, open(SYNTH_POP_PATH, 'w'))
pot_pop.to_csv(POT_POP_PATH, index=False)
# main commuting mode table for fitting long term model
for i in range(1,5):
    pot_pop[pot_pop['job_type_'+str(i)]==1].sample(n=50,replace=True).to_csv(POT_POP_BY_JOB_PATH+str(i)+'.csv', index=False)
mode_table_sample.to_csv(MODE_TABLE_PATH, index=False)
# tour table for fitting short term models
#nhts_tour_zone.to_csv(TOUR_TABLE_PATH, index=False)
