#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:24:53 2019
    input data must contain:
    'drive_time', 'cycle_time', 'walk_time', 'PT_time', 'walk_time_PT',
    'drive_cost','cycle_cost', 'walk_cost','PT_cost', 'main_mode'
    Other columns are treated as individual-specific variables
@author: doorleyr
"""
from collections import OrderedDict  
import pandas as pd
import pylogit as pl
import numpy as np
#********************************************
#      Constants
#********************************************
MODE_TABLE_PATH='../../data/Boston/clean/main_modes.csv'
TOUR_TABLE_PATH='../../data/Boston/clean/tours.csv'
#********************************************
#      Data
#********************************************
mode_table=pd.read_csv(MODE_TABLE_PATH)
# put the data in long format suitable for pylogit

mode_table=mode_table.sort_values(by='main_mode').reset_index(drop=True)
###################### Create Long Dataframe for max likelihood estimation ############################

# Specify the variables that vary across individuals
ind_variables = [c for c in mode_table.columns if c not in 
                 ['walk_time', 'walk_time_PT','drive_time', 'drive_time_PT','PT_time',
                  'drive_cost', 'walk_cost','cycle_cost','PT_cost', "main_mode"]]

# Specify the variables that vary across individuals and some or all alternatives
alt_varying_variables = {
                        'walk_time': dict([(2, 'walk_time'),
                                             (3, 'walk_time_PT')]),
#                        'drive_time': dict([(0, 'drive_time'),
#                                             (3, 'drive_time_PT')]),
                        'vehicle_time': dict([(0, 'drive_time'),
                                             (3, 'PT_time')]),
                        'cost': dict([(0, 'drive_cost'),
                                       (1, 'cycle_cost'),
                                       (2, 'walk_cost'),
                                       (3, 'PT_cost')])}

# Specify the availability variables
mode_table['car_avail']=1
mode_table['cycle_avail']=1
mode_table['walk_avail']=1
mode_table['PT_avail']=1

availability_variables = {0:'car_avail',
                          1:'cycle_avail',
                          2:'walk_avail',
                          3:'PT_avail'}

# The 'custom_alt_id' is the name of a column to be created in the long-format data
# It will identify the alternative associated with each row.
custom_alt_id = "mode_id"
# Create a custom id column. Note the +1 ensures the id's start at one.
obs_id_column = "custom_id"
mode_table[obs_id_column] = np.arange(mode_table.shape[0],
                                            dtype=int) + 1

choice_column = "main_mode"
long_mode_table=pl.convert_wide_to_long(mode_table, 
                                   ind_variables, 
                                   alt_varying_variables, 
                                   availability_variables, 
                                   obs_id_column, 
                                   choice_column,
                                   new_alt_id_name=custom_alt_id)

#********************************************
#      Model Specification
#********************************************
basic_specification = OrderedDict()
basic_names = OrderedDict()

basic_specification["intercept"] = [1, 2, 3]
basic_names["intercept"] = ['cycle_intercept',
                            'walk_intercept',
                            'transit_intercept']
#only need N-1 vars for intercepts and individual variables

# specify  the alternative-specific variables
basic_specification["walk_time"] = [[2, 3]]
basic_names["walk_time"] = ['all_walking_time']

#basic_specification["drive_time"] = [[0, 3]]
#basic_names["drive_time"] = ['driving time']
basic_specification["vehicle_time"] = [[0, 3]]
basic_names["vehicle_time"] = ['all_vehicle_time']

basic_specification["cycle_time"] = [1]
basic_names["cycle_time"] = ['all_cycling_time']
#
#basic_specification["cost"] = [[0,1,2,3]]
#basic_names["cost"] = ['cost']

#basic_specification["PT_time"] = [3]
#basic_names["PT_time"] = ['transit time']

# all other variables treated as individual-specific variables
for v in ind_variables:
    if v not in basic_names:
        basic_specification[v] = [1,2,3]
        basic_names[v] = ['cycle_'+v, 'walk_'+v, 'transit_'+v]

#********************************************
#      Fit model
#********************************************

main_mode_mnl = pl.create_choice_model(data=long_mode_table,
                                        alt_id_col=custom_alt_id,
                                        obs_id_col=obs_id_column,
                                        choice_col=choice_column,
                                        specification=basic_specification,
                                        model_type="MNL",
                                        names=basic_names)

# Specify the initial values and method for the optimization.
numCoef=sum([len(basic_specification[s]) for s in basic_specification])
main_mode_mnl.fit_mle(np.zeros(numCoef))

# Look at the estimation results
summary=main_mode_mnl.get_statsmodels_summary()
print(summary)

#********************************************
#      Save Results
#********************************************
coefficient_names=[summary.tables[1].data[i][0] for i in range(1,len(summary.tables[1].data))]
coefficients=[summary.tables[1].data[i][1] for i in range(1,len(summary.tables[1].data))]
coef_dict={'drive': {}, 'cycle':{}, 'walk':{}, 'transit':{}}
for c in range(len(coefficient_names)):
    for m in coef_dict:
        if (((m+'_') in coefficient_names[c]) or ('all_' in coefficient_names[c])):
            coef_dict[m][coefficient_names[c].split('_',1)[1]]=float(coefficients[c])
        
            
# Test
long_mode_table['P']=main_mode_mnl.predict(long_mode_table)
def compute_probs(row, coefs):
    exp_score_cycle=np.exp(coefs['cycle']['intercept']+
                coefs['cycle']['age']*row['age']+
                coefs['cycle']['bachelor_degree']*row['bachelor_degree']+
                coefs['cycle']['cycling_time']*row['cycle_time']+
                coefs['cycle']['hh_income']*row['hh_income']+
                coefs['cycle']['male']*row['male']+
                coefs['cycle']['pop_per_sqmile_home']*row['pop_per_sqmile_home']+
                coefs['cycle']['vehicle_time']*0+
                coefs['cycle']['walking_time']*0)
    exp_score_walk=np.exp(coefs['walk']['intercept']+
                coefs['walk']['age']*row['age']+
                coefs['walk']['bachelor_degree']*row['bachelor_degree']+
                coefs['walk']['cycling_time']*0+
                coefs['walk']['hh_income']*row['hh_income']+
                coefs['walk']['male']*row['male']+
                coefs['walk']['pop_per_sqmile_home']*row['pop_per_sqmile_home']+
                coefs['walk']['vehicle_time']*0+
                coefs['walk']['walking_time']*row['walk_time'])
    exp_score_transit=np.exp(coefs['transit']['intercept']+
                coefs['transit']['age']*row['age']+
                coefs['transit']['bachelor_degree']*row['bachelor_degree']+
                coefs['transit']['cycling_time']*0+
                coefs['transit']['hh_income']*row['hh_income']+
                coefs['transit']['male']*row['male']+
                coefs['transit']['pop_per_sqmile_home']*row['pop_per_sqmile_home']
                +coefs['transit']['vehicle_time']*(row['PT_time'])
                +coefs['transit']['walking_time']*row['walk_time_PT']
                )
    exp_score_drive=np.exp(coefs['drive']['vehicle_time']*row['drive_time']+
                           coefs['drive']['walking_time']*0+
                           coefs['drive']['cycling_time']*0)
    
    sum_exp=exp_score_cycle+exp_score_walk+exp_score_transit+exp_score_drive
    p_cycle=exp_score_cycle/sum_exp
    p_walk=exp_score_walk/sum_exp
    p_transit=exp_score_transit/sum_exp
    p_drive=exp_score_drive/sum_exp
    return [p_drive, p_cycle, p_walk, p_transit]
            
x=mode_table.apply(lambda row:compute_probs(row, coef_dict ), axis=1)             
true_P_PT= [long_mode_table.iloc[3+i*4]['P'] for i in range(15000)]           
                
