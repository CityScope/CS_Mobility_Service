#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 12:50:16 2019

@author: doorleyr
"""

import pandas as pd
import pylogit as pl
import numpy as np
import json
import math
import pickle
import random
from collections import OrderedDict
from shapely.geometry import shape


def get_haversine_distance(point_1, point_2):
    """
    Calculate the distance between any 2 points on earth given as [lon, lat]
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [point_1[0], point_1[1], 
                                                point_2[0], point_2[1]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


city='Detroit'
state_codes={'Detroit': 'mi', 'Boston': 'ma'}
state_fips={'Detroit': '26', 'Boston': '25'}
NUM_ALTS=8
sample_size=5000

#PUMAS_INCLUDED_PATH='./'+city+'/raw/PUMS/pumas_included.json'
FITTED_HOME_LOC_MODEL_PATH='./cities/'+city+'/models/home_loc_logit.p'
PUMA_POP_PATH='./cities/'+city+'/raw/ACS/ACS_17_5YR_B01003/population.csv'
PUMS_HH_PATH='./cities/'+city+'/raw/PUMS/csv_h'+state_codes[city]+'/ss16h'+state_codes[city]+'.csv'
PUMS_POP_PATH='./cities/'+city+'/raw/PUMS/csv_p'+state_codes[city]+'/ss16p'+state_codes[city]+'.csv'
PUMA_SHAPE_PATH='./cities/'+city+'/raw/PUMS/pumas.geojson'
PUMA_TO_POW_PUMA_PATH='./puma_to_pow_puma.csv'
RENT_NORM_PATH='./cities/'+city+'/models/rent_norm.json'

hh=pd.read_csv(PUMS_HH_PATH)
pop = pd.read_csv(PUMS_POP_PATH)
hh['PUMA']=hh.apply(lambda row: str(int(row['PUMA'])).zfill(5), axis=1)
pop['PUMA']=pop.apply(lambda row: str(int(row['PUMA'])).zfill(5), axis=1)
pop['POWPUMA']=pop.apply(lambda row: str(int(row['POWPUMA'])).zfill(5) 
                        if not np.isnan(row['POWPUMA']) else 'NaN', axis=1)

#pumas_included=json.load(open(PUMAS_INCLUDED_PATH))
pumas_shape=json.load(open(PUMA_SHAPE_PATH))
pumas_order=[f['properties']['PUMACE10'] for f in pumas_shape['features']]
          
puma_pop=pd.read_csv(PUMA_POP_PATH)
puma_pop['PUMA']=puma_pop.apply(lambda row: str(row['GEO.id2'][2:]).zfill(5), axis=1)
puma_pop=puma_pop.set_index('PUMA')


# identify recent movers and vacant houses                                            
hh_vacant_for_rent=hh[hh['VACS']==1].copy()          #hh in MI and for rent
hh_rented=hh[hh['TEN']==3].copy()                            #hh in MI and rented                            
renters_recent_move=hh_rented[hh_rented['MV']==1].copy()     #(MV=1: 12 months or less) hh in MI and rented in lt 12 months

# get the area of each PUMA
puma_land_sqm={str(int(f['properties']['PUMACE10'])).zfill(5): f['properties']['ALAND10']
                for f in pumas_shape['features']}

# =============================================================================
# Distance Matrix
# =============================================================================
# get the distance between each puma and each pow-puma
# first get a lookup between pow-pumas and pumas
# because we only have the shapes of the PUMAS
pow_puma_df=pd.read_csv(PUMA_TO_POW_PUMA_PATH, skiprows=1, header=1)
pow_puma_df_state=pow_puma_df.loc[pow_puma_df[
        'State of Residence (ST)']==state_fips[city]]
pow_puma_df_state['POW_PUMA']=pow_puma_df_state.apply(
        lambda row: str(int(row['PWPUMA00 or MIGPUMA1'])).zfill(5), axis=1)
pow_puma_df_state['PUMA']=pow_puma_df_state.apply(
        lambda row: str(int(row['PUMA'])).zfill(5), axis=1)
all_pow_pumas=set(pow_puma_df_state['POW_PUMA'])
pow_puma_to_puma={}
for p in all_pow_pumas:
    pow_puma_to_puma[p]=list(pow_puma_df_state.loc[
            pow_puma_df_state['POW_PUMA']==p, 'PUMA'].values)

# find the centroid of each puma
puma_centroids={}
pow_puma_centroids={}
for puma in set(pow_puma_df_state['PUMA']):
    centr=shape(pumas_shape['features'][pumas_order.index(puma)]['geometry']).centroid
    puma_centroids[puma]=[centr.x, centr.y]
# and each pow-puma
all_pow_pumas=set(pow_puma_df_state['POW_PUMA'])
for pow_puma in all_pow_pumas:
    pumas=pow_puma_to_puma[pow_puma]
    puma_centr=[puma_centroids[puma] for puma in pumas]
    # TODO, shold be weighted by area- ok if similar size
    pow_puma_centroids[pow_puma]=[np.mean([pc[0] for pc in puma_centr]),
                                  np.mean([pc[1] for pc in puma_centr])]
dist_mat={}
for puma in puma_centroids:
    dist_mat[puma]={}
    for pow_puma in pow_puma_centroids:
        dist_mat[puma][pow_puma]=get_haversine_distance(
                puma_centroids[puma], pow_puma_centroids[pow_puma])
        

# build the PUMA aggregate data data frame
median_income_by_puma=hh.groupby('PUMA')['HINCP'].median()
#TODO: get more zonal attributes such as access to employment, amenities etc.

all_PUMAs=list(set(hh['PUMA']))
puma_obj=[{'PUMA':puma,
           'med_income':median_income_by_puma.loc[puma],
           'puma_pop_per_sqm':float(puma_pop.loc[puma]['HD01_VD01'])/puma_land_sqm[puma]
           } for puma in all_PUMAs]
puma_attr_df=pd.DataFrame(puma_obj)
puma_attr_df=puma_attr_df.set_index('PUMA')


# create features at property level
# normalise rent stratifying by bedroom number
renters_recent_move.loc[renters_recent_move['BDSP']>2, 'BDSP']=3            # change [the number of bedroom] >2 to 3
renters_recent_move.loc[renters_recent_move['BDSP']<1, 'BDSP']=1            # change [the number of bedroom] <1 to 1
hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']>2, 'BDSP']=3          
hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']<1, 'BDSP']=1
rent_mean={}
rent_std={}
for beds in range(1,4):
    rent_mean[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].mean()
    rent_std[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].std()
    
for df in [renters_recent_move, hh_vacant_for_rent]:
    df['norm_rent']=df.apply(
        lambda row: (row['RNTP']-rent_mean[row['BDSP']])/rent_std[row['BDSP']], axis=1)
    # Age of building
    df['built_since_jan2010']=df.apply(lambda row: row['YBL']>=14, axis=1)
    df['puma_pop_per_sqmeter']=df.apply(lambda row: puma_attr_df.loc[row['PUMA']]['puma_pop_per_sqm'], axis=1)
    df['med_income']=df.apply(lambda row: puma_attr_df.loc[row['PUMA']]['med_income'], axis=1)
        
renters_recent_move=renters_recent_move[['SERIALNO', 'PUMA','HINCP',  'norm_rent', 'RNTP', 'built_since_jan2010', 'puma_pop_per_sqmeter', 'med_income']]
hh_vacant_for_rent=hh_vacant_for_rent[['PUMA', 'HINCP', 'norm_rent', 'RNTP','built_since_jan2010', 'puma_pop_per_sqmeter', 'med_income']]
 
rent_normalisation={"mean": rent_mean, "std": rent_std}   
        
# TODO: include feature for same PUMA/POWPUMA
random.seed(1)
long_data_obj=[]
ind=0
for ind_actual, row_actual in renters_recent_move[:sample_size].iterrows():
    cid=1
    householdID = row_actual['SERIALNO']
    places_of_work = set(pop.loc[pop['SERIALNO']==householdID, 'POWPUMA'])
    # fixing problem: a few pow_pumas can not be found in all_PUMAs_string
    places_of_work = [x for x in places_of_work if x in all_pow_pumas]
    if len(places_of_work) > 0:
        choiceObs={'custom_id':ind,# identify the individual
                   'choice_id':cid, # fake choice identifier- shouldn't matter if no ASC
                   'choice':1,
                   'rent':row_actual['RNTP'],
                   'norm_rent':row_actual['norm_rent'],
                   'puma':row_actual['PUMA'],
                   'built_since_jan2010':int(row_actual['built_since_jan2010']),
                   'puma_pop_per_sqmeter': row_actual['puma_pop_per_sqmeter'],
                   'med_income': row_actual['med_income'],
                   'hh_income':row_actual['HINCP'],
                   'work_dist': np.mean([dist_mat[row_actual['PUMA']][pow_puma] 
                                           for pow_puma in places_of_work])
                   }
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
                     'hh_income':row_actual['HINCP'],
                     'work_dist': np.mean([dist_mat[hh_vacant_for_rent.iloc[selected]['PUMA']][pow_puma] 
                                           for pow_puma in places_of_work])
                     }
            cid+=1
            long_data_obj.append(alt_obs)
        ind+=1

# get zonal attributes
long_data=pd.DataFrame(long_data_obj)  
long_data['log_rent']=long_data.apply(lambda row: np.log(row['rent']), axis=1)

# TODO: calculate interactions
long_data['income_disparity']=long_data.apply(lambda row: np.abs(row['hh_income']-row['med_income']), axis=1)

# fit model

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

home_loc_mnl = pl.create_choice_model(data=long_data,
                                        alt_id_col='choice_id',
                                        obs_id_col='custom_id',
                                        choice_col='choice',
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

# Specify the initial values and method for the optimization.
print('Fitting Model')
numCoef=sum([len(basic_specification[s]) for s in basic_specification])
home_loc_mnl.fit_mle(np.zeros(numCoef))

# Look at the estimation results
print(home_loc_mnl.get_statsmodels_summary())

pickle.dump(home_loc_mnl, open(FITTED_HOME_LOC_MODEL_PATH, 'wb'))
json.dump(rent_normalisation, open(RENT_NORM_PATH, 'w'))




