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
import pickle
import random
from collections import OrderedDict

city='Boston'

NUM_ALTS=5
sample_size=5000

PUMS_HH_PATH='./'+city+'/raw/PUMS/csv_hma/ss16hma.csv'
PUMA_POP_PATH='./'+city+'/raw/ACS/ACS_17_5YR_B01003/population.csv'
PUMAS_INCLUDED_PATH='./'+city+'/raw/PUMS/pumas_included.json'
PUMA_SHAPE_PATH='./'+city+'/raw/PUMS/pumas.geojson'
FITTED_HOME_LOC_MODEL_PATH='./models/home_loc_logit.p'

# get area of every PUMA
pumas_shape=json.load(open(PUMA_SHAPE_PATH))
print(pumas_shape['features'][0]['properties'])
puma_land_sqm={int(f['properties']['PUMACE10']): f['properties']['ALAND10']
                for f in pumas_shape['features']}

# load the PUMS data
hh=pd.read_csv(PUMS_HH_PATH)
pumas_included=json.load(open(PUMAS_INCLUDED_PATH))
# load the aggregate PUMA data
puma_pop=pd.read_csv(PUMA_POP_PATH)
puma_pop['PUMA']=puma_pop.apply(lambda row: int(row['GEO.id2'][2:]), axis=1)
puma_pop=puma_pop.set_index('PUMA')

#create subsets
hh_vacant=hh[hh['NP']==0]
hh_boston=hh[hh['PUMA'].isin(pumas_included)]
hh_vacant_for_rent=hh_boston[hh_boston['VACS']==1]
hh_rented=hh[hh['TEN']==3]
#asking_rent=hh_for_rent['RNTP']
renters_recent_move=hh_rented[hh_rented['MV']==1]

# create features at property level
# normalise rent in each cateogory og bedroom number
renters_recent_move.loc[renters_recent_move['BDSP']>2, 'BDSP']=3
renters_recent_move.loc[renters_recent_move['BDSP']<1, 'BDSP']=1
hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']>2, 'BDSP']=3
hh_vacant_for_rent.loc[hh_vacant_for_rent['BDSP']<1, 'BDSP']=1
rent_mean={}
rent_std={}
for beds in range(1,4):
    rent_mean[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].mean()
    rent_std[beds]=renters_recent_move.loc[renters_recent_move['BDSP']==beds, 'RNTP'].std()
renters_recent_move['norm_rent']=renters_recent_move.apply(
    lambda row: (row['RNTP']-rent_mean[row['BDSP']])/rent_std[row['BDSP']], axis=1)
hh_vacant_for_rent['norm_rent']=hh_vacant_for_rent.apply(
    lambda row: (row['RNTP']-rent_mean[row['BDSP']])/rent_std[row['BDSP']], axis=1)

# Age of building
for df in [renters_recent_move, hh_vacant_for_rent]:
    df['less_than_5yrs_old']=df.apply(lambda row: row['YBL']>14, axis=1)


# build the PUMA aggregate data data frame
median_income_by_puma=hh.groupby('PUMA')['HINCP'].median()
#TODO: get more zonal attributes such as access to employment, amenities etc.

all_PUMAs=list(set(hh['PUMA']))
puma_obj=[{'PUMA':puma,
           'med_income':median_income_by_puma.loc[puma],
           'puma_pop_per_sqm':float(puma_pop.loc[puma]['HD01_VD01'])/puma_land_sqm[puma]
           } for puma in all_PUMAs]
puma_df=pd.DataFrame(puma_obj)
puma_df=puma_df.set_index('PUMA')
# for each PUMS person, add to the long df their actual HH and N vacant HHs

random.seed(1)
long_data_obj=[]
ind=0
for ind_actual, row_actual in renters_recent_move[:sample_size].iterrows():
    cid=1
    choiceObs={'custom_id':ind,# identify the individual
               'choice_id':cid, # fake choice identifier- shouldn't matter if no ASC
               'choice':1,
               'rent':row_actual['RNTP'],
               'norm_rent':row_actual['norm_rent'],
               'puma':row_actual['PUMA'],
               'less_than_5yrs_old':int(row_actual['less_than_5yrs_old']),
               'hh_income':row_actual['HINCP']
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
                 'less_than_5yrs_old':int(hh_vacant_for_rent.iloc[selected]['less_than_5yrs_old']),
                 'hh_income':row_actual['HINCP']
                 }
        cid+=1
        long_data_obj.append(alt_obs)
    ind+=1

# get zonal attributes
long_data=pd.DataFrame(long_data_obj)  
long_data['puma_pop_per_sqm']=long_data.apply(lambda row: puma_df.loc[row['puma']]['puma_pop_per_sqm'], axis=1)
long_data['income_disparity']=long_data.apply(lambda row: np.abs(row['hh_income']-puma_df.loc[row['puma']]['med_income']), axis=1)
long_data['log_rent']=long_data.apply(lambda row: np.log(row['rent']), axis=1)

# TODO: calculate interactions

# fit model

basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["puma_pop_per_sqm"] = [list(set(long_data['choice_id']))]
basic_names["puma_pop_per_sqm"] = ['puma_pop_per_sqm']

basic_specification["income_disparity"] = [list(set(long_data['choice_id']))]
basic_names["income_disparity"] = ['income_disparity']

basic_specification["norm_rent"] = [list(set(long_data['choice_id']))]
basic_names["norm_rent"] = ['norm_rent']

basic_specification["less_than_5yrs_old"] = [list(set(long_data['choice_id']))]
basic_names["less_than_5yrs_old"] = ['less_than_5yrs_old']

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

pickle.dump(home_loc_mnl, FITTED_HOME_LOC_MODEL_PATH)





