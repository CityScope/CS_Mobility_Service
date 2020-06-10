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

# =============================================================================
# Create 2 new mode specs:
# dockless bikes and shuttle buses
# =============================================================================

bikeshare_spec={'name': 'bikeshare', 
                 'attrs':{'active_time_minutes': 'c*1',
                          'cost':1.5},  
                 'copy': 'cycle',
                 "copy_route": "cycling","activity": "cycling","speed_m_s": 4.167,
                 "co2_emissions_kg_met": 0,"fixed_costs": {},
                 'internal_net': 'drive'}
shuttle_spec={'name': 'shuttle', 
                 'attrs':{'vehicle_time_minutes': 'd*1',
                          'cost':1.5},  
                 'copy': 'PT',
                 "copy_route": "driving","activity": "pt","speed_m_s": 8.33,
                 "co2_emissions_kg_met": 0.000066,"fixed_costs": {},
                 'internal_net': 'drive'}

new_mode_specs=[bikeshare_spec, shuttle_spec]

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
            
# alpha: degree of membership (0, 1). Sum to 1 for each alt. Leave it as default (equal for each nest)
# lambda: degree of nest similarity (0, 1). Should be consistent with betas.
# i.e. if bikeshare and PT in nest together with small dissiilarity (eg. lambda =0.3)
# and bikeshare and walk in nest together with large dissiilarity: (eg. lambda =0.7)
# then betas for bikeshare should be closer to those of PT than walk


# =============================================================================
# Create model and add it to a handler
# =============================================================================
this_model=MobilityModel('corktown', 'Detroit')

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

mode_choice_model=NhtsModeLogit(table_name='corktown', city_folder='Detroit')

mode_choice_model.logit_model['params']['ASC for PT']=-1.5

# =============================================================================
# Create new parameters for shared bike- dont assign yet
# =============================================================================

params_for_share_bike = {}
existing_params = mode_choice_model.logit_model['params']
for g_attr in mode_choice_model.logit_generic_attrs:
    params_for_share_bike['{} for bikeshare'.format(g_attr)] = \
        existing_params['{} for PT'.format(g_attr)] * prop_pt_similarity + \
        existing_params['{} for walk'.format(g_attr)] * (1-prop_pt_similarity)
params_for_share_bike['ASC for bikeshare'] = existing_params['ASC for PT'] * prop_pt_similarity + \
        existing_params['ASC for walk'] * (1-prop_pt_similarity)


# =============================================================================
# Start scenarios
# =============================================================================


this_model.assign_mode_choice_model(mode_choice_model)
#this_model.assign_mode_choice_model(NhtsModeRF(table_name='corktown', city_folder='Detroit'))

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))


handler=CS_Handler(this_model)
print('Baseline')
#handler.post_trips_data()
print(handler.get_outputs())

geogrid_data_campus=json.load(open('../../Scenarios/21_May_20/ford_campus.json'))
geogrid_data_housing=json.load(open('../../Scenarios/21_May_20/ford_housing.json'))
geogrid_data_inno_com=json.load(open('../../Scenarios/21_May_20/ford_inno_com.json'))

# 
print('Campus Only')
handler.model.update_simulation(geogrid_data_campus)
#handler.post_trips_data()
print(handler.get_outputs())


print('Campus and Housing')
handler.model.update_simulation(geogrid_data_housing)
#handler.post_trips_data()
print(handler.get_outputs())

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
print(handler.get_outputs())

print('Campus and Housing and Mobility')
handler.model.update_simulation(geogrid_data_inno_com)
#handler.post_trips_data()
print(handler.get_outputs())




