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

dockless_spec={'name': 'dockless', 
                 'attrs':{'time_minutes': 'c*1'},  # dockless_time_minutes = cycle_time_minutes * 0.7
                 'copy': 'PT',
                 "copy_route": "cycling","activity": "cycling","speed_m_s": 4.16,
                 "co2_emissions_kg_met": 0,"fixed_costs": {}}
shuttle_spec={'name': 'shuttle', 
                 'attrs':{'time_minutes': 'd*1'},  # dockless_time_minutes = cycle_time_minutes * 0.7
                 'copy': 'PT',
                 "copy_route": "driving","activity": "pt","speed_m_s": 11.1,
                 "co2_emissions_kg_met": 0.000066,"fixed_costs": {}}

new_mode_specs=[dockless_spec, shuttle_spec]

nests_spec=[{'name': 'pt_like', 
             'alts':[
                     'PT',
                     'dockless',
                     'shuttle'], 
             'sigma':0.9}
            ]
    
# =============================================================================
# Create model and add it to a handler
# =============================================================================
this_model=MobilityModel('corktown', 'Detroit')

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

mode_choice_model=NhtsModeLogit(table_name='corktown', city_folder='Detroit')
#mode_choice_model.logit_model['params']['ASC for PT']=-1
#mode_choice_model.logit_model['params']['ASC for cycle']=0.5

this_model.assign_mode_choice_model(mode_choice_model)
#this_model.assign_mode_choice_model(NhtsModeRF(table_name='corktown', city_folder='Detroit'))

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))


handler=CS_Handler(this_model)
print('Baseline')
handler.post_trips_data()
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
this_model.set_prop_electric_cars(0.5)
this_model.set_new_modes(new_mode_specs, nests_spec=nests_spec)
print('Campus and Mobility')
handler.model.update_simulation(geogrid_data_campus)
#handler.post_trips_data()
print(handler.get_outputs())

print('Campus and Housing and Mobility')
handler.model.update_simulation(geogrid_data_inno_com)
handler.post_trips_data()
print(handler.get_outputs())




