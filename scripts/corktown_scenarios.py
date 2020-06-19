#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:25:03 2020

@author: doorleyr
"""

from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_choice_nhts import NhtsModeLogit, NhtsModeRF
from two_stage_logit_hlc import TwoStageLogitHLC
from cs_handler import CS_Handler
import json

import pandas as pd

# =============================================================================
# Create 2 new mode specs:
# dockless bikes and shuttle buses
# =============================================================================

new_mode_specs=json.load(open('cities/Detroit/clean/new_mode_specs.json'))

pt_dissimilarity = 0.7
walk_dissimilarity = 0.3
prop_pt_similarity=1-(pt_dissimilarity/(pt_dissimilarity+walk_dissimilarity))


nests_spec=[{'name': 'pt_like', 
             'alts':['PT','bikeshare', 'shuttle'], 
             'lambda':pt_dissimilarity},
             {'name': 'walk_like',
             'alts': ['walk', 'bikeshare'],
             'lambda':walk_dissimilarity}
            ]

mode_choice_model=NhtsModeLogit(table_name='corktown', city_folder='Detroit')

# =============================================================================
# Adjust Base ASCs
# =============================================================================
initial_ASC_PT=mode_choice_model.logit_model['params']['ASC for PT']
initial_ASC_cycle=mode_choice_model.logit_model['params']['ASC for cycle']
initial_ASC_walk=mode_choice_model.logit_model['params']['ASC for walk']
mode_choice_model.logit_model['params']['ASC for PT']=-0.9
mode_choice_model.logit_model['params']['ASC for cycle']=-0.9
mode_choice_model.logit_model['params']['ASC for walk']=2.9
print('Modified ASC of PT from {} to {}'.format(
        initial_ASC_PT, mode_choice_model.logit_model['params']['ASC for PT']))
print('Modified ASC of cycling from {} to {}'.format(
        initial_ASC_cycle, mode_choice_model.logit_model['params']['ASC for cycle']))
print('Modified ASC of walkinf from {} to {}'.format(
        initial_ASC_walk, mode_choice_model.logit_model['params']['ASC for walk']))

params_for_share_bike = {}
existing_params = mode_choice_model.logit_model['params']
for g_attr in mode_choice_model.logit_generic_attrs:
    params_for_share_bike['{} for bikeshare'.format(g_attr)] = \
        existing_params['{} for PT'.format(g_attr)] * prop_pt_similarity + \
        existing_params['{} for walk'.format(g_attr)] * (1-prop_pt_similarity)
params_for_share_bike['ASC for bikeshare'] = existing_params['ASC for PT'] * prop_pt_similarity + \
        existing_params['ASC for walk'] * (1-prop_pt_similarity)

            
# alpha: degree of membership (0, 1). Sum to 1 for each alt. Leave it as default (equal for each nest)
# lambda: degree of nest similarity (0, 1). Should be consistent with betas.
# i.e. if bikeshare and PT in nest together with small dissiilarity (eg. lambda =0.3)
# and bikeshare and walk in nest together with large dissiilarity: (eg. lambda =0.7)
# then betas for bikeshare should be closer to those of PT than walk


this_model=MobilityModel('corktown', 'Detroit')

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

this_model.assign_mode_choice_model(mode_choice_model)


this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))

all_results=[]

handler=CS_Handler(this_model)
print('Baseline')
#handler.post_trips_data()
outputs=handler.get_outputs()
outputs['Scenario']='BAU'
print(outputs)
all_results.append(outputs)

geogrid_data_campus=json.load(open('../../Scenarios/12_Jun_20/ford_campus.json'))
geogrid_data_housing=json.load(open('../../Scenarios/12_Jun_20/ford_housing.json'))
geogrid_data_inno_com=json.load(open('../../Scenarios/12_Jun_20/ford_inno_com.json'))

# 
print('Campus Only')
handler.model.update_simulation(geogrid_data_campus)
#handler.post_trips_data()
outputs=handler.get_outputs()
outputs['Scenario']='Campus Only'
print(outputs)
all_results.append(outputs)


print('Campus and Housing')
handler.model.update_simulation(geogrid_data_housing)
#handler.post_trips_data()
outputs=handler.get_outputs()
outputs['Scenario']='Campus and Housing'
print(outputs)
all_results.append(outputs)

# =============================================================================
# Mobility interventions
# =============================================================================
this_model.set_prop_electric_cars(0.5, co2_emissions_kg_met_ic= 0.000272,
                                  co2_emissions_kg_met_ev=0.00011)
this_model.set_new_modes(new_mode_specs, nests_spec=nests_spec)
mode_choice_model.set_logit_model_params(params_for_share_bike)


print('Campus and Mobility')
handler.model.update_simulation(geogrid_data_campus)
#handler.post_trips_data()
outputs=handler.get_outputs()
outputs['Scenario']='Campus and Mobility'
print(outputs)
all_results.append(outputs)

print('Campus and Housing and Mobility')
handler.model.update_simulation(geogrid_data_inno_com)
#handler.post_trips_data()
#handler.post_trips_data_w_attrs()
outputs=handler.get_outputs()
outputs['Scenario']='Innovation Community'
print(outputs)
all_results.append(outputs)

all_results_df=pd.DataFrame(all_results)

all_results_df=all_results_df.set_index('Scenario')
all_results_df=all_results_df.reindex(["BAU", "Campus Only", "Campus and Mobility",
                                       "Campus and Housing", "Innovation Community"])

all_results_df.to_csv('Mobility_Scenarios.csv')


