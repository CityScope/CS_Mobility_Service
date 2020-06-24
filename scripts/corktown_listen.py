#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:03:46 2020

@author: doorleyr
"""

from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_choice_nhts import NhtsModeLogit, NhtsModeRF
from two_stage_logit_hlc import TwoStageLogitHLC
from cs_handler import CS_Handler
import json


# =============================================================================
# Define New Modes
# =============================================================================
new_mode_specs=json.load(open('cities/Detroit/clean/new_mode_specs.json'))

ASC_micromobility= 2.8
ASC_shuttle= 2.75
beta_similarity_PT= 0.45
lambda_PT= 0.7
lambda_walk= 0.4

nests_spec=[{'name': 'PT_like', 'alts':['micromobility', 'PT', 'shuttle'], 'lambda':lambda_PT},
            {'name': 'walk_like', 'alts':['micromobility','walk'], 'lambda':lambda_walk}
               ]
mode_choice_model=NhtsModeLogit(table_name='corktown', city_folder='Detroit')

# update ASCs of base model
initial_ASC_PT=mode_choice_model.logit_model['params']['ASC for PT']
initial_ASC_cycle=mode_choice_model.logit_model['params']['ASC for cycle']
initial_ASC_walk=mode_choice_model.logit_model['params']['ASC for walk']

new_ASCs = {'ASC for cycle': -0.9, 'ASC for PT': -0.9, 'ASC for walk': 2.9}
initial_ASCs= {param:mode_choice_model.logit_model['params'][param] for param in new_ASCs}
mode_choice_model.set_logit_model_params(new_ASCs)

for param in new_ASCs:
    print('Modified {} from {} to {}'.format(param, initial_ASCs[param], mode_choice_model.logit_model['params'][param]))

# calculate paams for new modes
new_beta_params = {}
crt_logit_params = mode_choice_model.logit_model['params']
for g_attr in mode_choice_model.logit_generic_attrs:
    new_beta_params['{} for micromobility'.format(g_attr)] = \
        crt_logit_params['{} for PT'.format(g_attr)] * beta_similarity_PT + \
        crt_logit_params['{} for walk'.format(g_attr)] * (1-beta_similarity_PT)
new_beta_params['ASC for micromobility'] =  ASC_micromobility
new_beta_params['ASC for shuttle'] = ASC_shuttle


# =============================================================================
# Create model
# =============================================================================
this_model=MobilityModel('corktown', 'Detroit', seed=0)

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

this_model.assign_mode_choice_model(mode_choice_model)

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))

this_model.set_prop_electric_cars(0.5, co2_emissions_kg_met_ic= 0.000272,
                                  co2_emissions_kg_met_ev=0.00011)
this_model.set_new_modes(new_mode_specs, nests_spec=nests_spec)

handler=CS_Handler(this_model, new_logit_params=new_beta_params)

handler.listen_city_IO()