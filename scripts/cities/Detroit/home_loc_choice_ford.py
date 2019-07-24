# -*- coding: utf-8 -*-


import pandas as pd
import pylogit as pl
import numpy as np
import json
import pickle
import random
from collections import OrderedDict

import sys
import os
from os import path,chdir
#use relative path
chdir(path.dirname(sys.argv[0]))




city='Detroit'
NUM_ALTS=8
sample_size=5000

# PUMAS_INCLUDED_PATH='./'+city+'/raw/PUMS/pumas_included.json'        #use all Michigan data and thus do not use pumas_included
FITTED_HOME_LOC_MODEL_PATH='./models/home_loc_logit.p'          
PUMA_POP_PATH='./'+city+'/raw/ACS/ACS_17_1YR_B01003/population.csv'     
PUMS_HH_PATH='./'+city+'/raw/PUMS/csv_hmi 5-year/psam_h26.csv'      
PUMS_POP_PATH = './'+city+'/raw/PUMS/csv_pmi 5-year/psam_p26.csv'      #PUMS population record for places of work
PUMA_SHAPE_PATH='./'+city+'/raw/PUMS/pumas.geojson'         
PUMA_SHAPE_DIST_PATH = './'+city+'/raw/PUMS/puma_distance.csv'         #pre-cooked distance matrix among PUMAs 
RENT_NORM_PATH='./models/rent_norm.json'
PUMA_POI_PATH = './'+city+'/raw/PUMS/poi.geojson'                      #PUMAs geojson with poi counts and densities included in ['properties'], derived by point_in_polygon.py

hh=pd.read_csv(PUMS_HH_PATH)
# pumas_included=json.load(open(PUMAS_INCLUDED_PATH))
pumas_shape=json.load(open(PUMA_SHAPE_PATH))
pop = pd.read_csv(PUMS_POP_PATH)

puma_poi_list = json.loads(open(PUMA_POI_PATH, 'r', encoding='utf-8').read())['features']
puma_poi_dict = {x['properties']['PUMA']:x['properties'] for x in puma_poi_list}
          
puma_pop=pd.read_csv(PUMA_POP_PATH)
puma_pop['PUMA']=puma_pop.apply(lambda row: int(str(row['GEO.id2'])[2:]), axis=1)            #"2:" the first 2 letters is state code 
puma_pop=puma_pop.set_index('PUMA')


# identify recent movers and vacant houses                                            
hh_vacant_for_rent=hh[hh['VACS']==1].copy()          #hh in MI and for rent
hh_rented=hh[hh['TEN']==3].copy()                            #hh in MI and rented                            
renters_recent_move=hh_rented[hh_rented['MV']==1].copy()     #(MV=1: 12 months or less) hh in MI and rented in lt 12 months


# load pumas distance
distMatrix = pd.read_csv(PUMA_SHAPE_DIST_PATH)
# generate indices like: from_00100_to_00200
distMatrix['from_to_pair'] = distMatrix.apply(lambda row: 'from_'+str(int(row['from'])).zfill(5)+'_to_'+str(int(row['to'])).zfill(5), axis=1)
distMatrix = distMatrix.set_index('from_to_pair')

# Load the PUMA shape data and get the area of each PUMA: the area is acquired by geocalculation in km2
puma_land_sqm={int(f['properties']['PUMA']): f['properties']['ALAND']*1000000
                for f in pumas_shape['features']}
 
median_income_by_puma=hh.groupby('PUMA')['HINCP'].median()

# puma statistcs according to PUMS HH data
all_PUMAs=list(set(hh['PUMA']))
all_PUMAs_string = [str(int(x)).zfill(5) for x in all_PUMAs]            #PUMA names in 5-char string
puma_obj=[{'PUMA':puma,
           'med_income':median_income_by_puma.loc[puma],
           'puma_pop_per_sqm':float(puma_pop.loc[puma]['B01003_001'])/puma_land_sqm[puma]
           } for puma in all_PUMAs]
puma_df=pd.DataFrame(puma_obj)
puma_df=puma_df.set_index('PUMA')
# (puma_df is a dataframe containning media income and population density of each puma)


# create features at property level
# normalise rent stratifying by bedroom number
renters_recent_move.loc[renters_recent_move['BDSP']>2, 'BDSP']=3            # change [the number of bedroom] >2 to 3
renters_recent_move.loc[renters_recent_move['BDSP']<1, 'BDSP']=1            # change [the number of bedroom] <1 to 1
hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']>2, 'BDSP']=3          
hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']<1, 'BDSP']=1
rent_mean={}
rent_std={}

for beds in range(1,4):
    rent_mean[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].mean()       #RNTP: monthly rent
    rent_std[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].std()
    
for df in [renters_recent_move, hh_vacant_for_rent]:
    df['norm_rent']=df.apply(
        lambda row: (row['RNTP']-rent_mean[row['BDSP']])/rent_std[row['BDSP']], axis=1)             # standarlized
    # Age of building
    df['built_since_jan2010']=df.apply(lambda row: row['YBL']>=14, axis=1)
    df['puma_pop_per_sqmeter']=df.apply(lambda row: puma_df.loc[row['PUMA']]['puma_pop_per_sqm'], axis=1)
    df['med_income']=df.apply(lambda row: puma_df.loc[row['PUMA']]['med_income'], axis=1)
  
renters_recent_move=renters_recent_move[['PUMA','HINCP', 'norm_rent', 'RNTP', 'built_since_jan2010', 'puma_pop_per_sqmeter', 'med_income','SERIALNO','VEH']]
hh_vacant_for_rent=hh_vacant_for_rent[['PUMA', 'HINCP', 'norm_rent', 'RNTP','built_since_jan2010', 'puma_pop_per_sqmeter', 'med_income']]

# vehicle dummy
renters_recent_move['VEH_dummy'] = renters_recent_move.apply(lambda row: row['VEH']>=1, axis=1)

rent_normalisation={"mean": rent_mean, "std": rent_std}  
     
# TODO: include feature for same PUMA/POWPUMA
random.seed(1)
long_data_obj=[]
ind=0
sample_count = 0

for ind_actual, row_actual in renters_recent_move.iterrows():
    cid=1
    householdID = row_actual['SERIALNO']
    places_of_work = list(pop.loc[pop['SERIALNO']==householdID, 'POWPUMA'])
    places_of_work = np.unique([str(int(x)).zfill(5) for x in places_of_work if not np.isnan(x)])
    # fixing problem: a few pow_pumas can not be found in all_PUMAs_string
    places_of_work = [x for x in places_of_work if x in all_PUMAs_string]

    # skip the household without POWPUMA records
    if len(places_of_work) == 0:      
        # print('Household {} is excluded for unknown working place or working outside the region of interest'.format(householdID))
        continue

    # a valid sample WITH POWPUMA
    sample_count += 1
    if sample_count > sample_size:
        print('Sample size reached.')
        break
    
    # this is the real choice
    choiceObs={'custom_id':ind,# identify the individual
               'choice_id':cid, # fake choice identifier- shouldn't matter if no ASC
               'choice':1,
               'rent':row_actual['RNTP'],
               'norm_rent':row_actual['norm_rent'],
               'puma':row_actual['PUMA'],
               'built_since_jan2010':int(row_actual['built_since_jan2010']),
               'puma_pop_per_sqmeter': row_actual['puma_pop_per_sqmeter'],
               'med_income': row_actual['med_income'],
               'hh_income':row_actual['HINCP']
               }
    # get the distance from home PUMA to place of work PUMA
    work_dists = [distMatrix.loc['from_'+ str(int(choiceObs['puma'])).zfill(5) +'_to_'+ str(int(pow)).zfill(5), 'dist'] 
        for pow in places_of_work]
    choiceObs['work_dist'] = np.array(work_dists).mean()
    choiceObs['work_dist_veh'] = choiceObs['work_dist'] * row_actual['VEH_dummy']       #interaction with vehicle ownership dummy
    
    cid+=1
    long_data_obj.append(choiceObs)
    for i in range(NUM_ALTS):
        selected=random.choice(range(len(hh_vacant_for_rent)))
        alt_obs={'custom_id':ind,# identify the individual
                 'choice_id':cid, # fake choice identifier- shouldn't matter if no ASC
                 'choice':0,
                 'rent':hh_vacant_for_rent.iloc[selected]['RNTP'],
                 'norm_rent':hh_vacant_for_rent.iloc[selected]['norm_rent'],
                 'puma':hh_vacant_for_rent.iloc[selected]['PUMA'],
                 'built_since_jan2010':int(hh_vacant_for_rent.iloc[selected]['built_since_jan2010']),
                 'puma_pop_per_sqmeter': hh_vacant_for_rent.iloc[selected]['puma_pop_per_sqmeter'],
                 'med_income': hh_vacant_for_rent.iloc[selected]['med_income'],
                 'hh_income':row_actual['HINCP']
                 }
        work_dists = [distMatrix.loc['from_'+ str(int(alt_obs['puma'])).zfill(5) +'_to_'+ str(int(pow)).zfill(5), 'dist'] 
            for pow in places_of_work]
        alt_obs['work_dist'] = np.array(work_dists).mean()
        alt_obs['work_dist_veh'] = alt_obs['work_dist'] * row_actual['VEH_dummy']
        cid+=1
        long_data_obj.append(alt_obs)
    ind+=1

# get zonal attributes
long_data=pd.DataFrame(long_data_obj)  

long_data['log_rent']=long_data.apply(lambda row: np.log(row['rent']), axis=1)

# TODO: calculate interactions
long_data['income_disparity']=long_data.apply(lambda row: np.abs(row['hh_income']-row['med_income']), axis=1)

###  get poi data: use poiFields to define what kinds of amenity accessabilities would be introduced as regressors.

# poiFields = ['amenity_count', 'amenity_den', 'medical_count', 'medical_den', 'restaurant_count', 
    # 'restaurant_den', 'school_count', 'school_den', 'transport_count', 'transport_den', 'financial_count', 
    # 'financial_den', 'entertainment_count', 'entertainment_den', 'other_count', 'other_den', 'leisure_count', 
    # 'leisure_den']       
poiFields = ['medical_den', 'restaurant_den', 'school_den', 'transport_den', 
    'financial_den', 'entertainment_den', 'other_den', 'leisure_den']     
for poiField in poiFields:
    long_data[poiField] = long_data.apply(lambda row: puma_poi_dict[str(int(row['puma'])).zfill(5)][poiField], axis=1)
    
# fit model
long_data.to_csv(r'./tmp/home/long_data.csv', index=False)

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["puma_pop_per_sqmeter"] = [list(set(long_data['choice_id']))]
basic_names["puma_pop_per_sqmeter"] = ['puma_pop_per_sqmeter']

basic_specification["income_disparity"] = [list(set(long_data['choice_id']))]
basic_names["income_disparity"] = ['income_disparity']

basic_specification["norm_rent"] = [list(set(long_data['choice_id']))]
basic_names["norm_rent"] = ['norm_rent']

basic_specification["built_since_jan2010"] = [list(set(long_data['choice_id']))]
basic_names["built_since_jan2010"] = ['built_since_jan2010']

basic_specification["work_dist"] = [list(set(long_data['choice_id']))]
basic_names["work_dist"] = ['work_dist']

for poiField in poiFields:
    basic_specification[poiField] = [list(set(long_data['choice_id']))]
    basic_names[poiField] = [poiField]

# interation of work distance and vehcile ownership: insignificant in the model
# basic_specification["work_dist_veh"] = [list(set(long_data['choice_id']))]
# basic_names["work_dist_veh"] = ['work_dist_veh']

home_loc_mnl = pl.create_choice_model(data=long_data,
                                        alt_id_col='choice_id',
                                        obs_id_col='custom_id',
                                        choice_col='choice',
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

print('Fitting Model')
numCoef=sum([len(basic_specification[s]) for s in basic_specification])
home_loc_mnl.fit_mle(np.zeros(numCoef))

# Look at the estimation results
print(home_loc_mnl.get_statsmodels_summary())

pickle.dump(home_loc_mnl, open(FITTED_HOME_LOC_MODEL_PATH, 'wb'))
json.dump(rent_normalisation, open(RENT_NORM_PATH, 'w'))




