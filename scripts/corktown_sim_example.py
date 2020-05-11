#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:25:03 2020

@author: doorleyr
"""

from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_logit_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC
from cs_handler import CS_Handler

# =============================================================================
# Create 2 new mode specs:
# dockless bikes and shuttle buses
# =============================================================================

dockless_spec={'name': 'dockless', 
                 'attrs':{'time_minutes': 'c*1'},  # dockless_time_minutes = cycle_time_minutes * 0.7
                 'copy': 'cycle',
                 "copy_route": "cycling","activity": "cycling","speed_m_s": 4.167,
                 "co2_emissions_kg_met": 0,"fixed_costs": {}}
shuttle_spec={'name': 'shuttle', 
                 'attrs':{'time_minutes': 'd*1'},  # dockless_time_minutes = cycle_time_minutes * 0.7
                 'copy': 'PT',
                 "copy_route": "driving","activity": "pt","speed_m_s": 8.33,
                 "co2_emissions_kg_met": 0.000066,"fixed_costs": {}}

new_mode_specs=[dockless_spec, shuttle_spec]

nests_spec=[{'name': 'pt_like', 
             'alts':['PT','dockless', 'shuttle'], 
             'sigma':0.9}
            ]
    
# =============================================================================
# Create the model and add it to a handler
# =============================================================================
this_model=MobilityModel('corktown', 'Detroit')

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

this_model.assign_mode_choice_model(NhtsModeLogit(table_name='corktown', city_folder='Detroit'))

this_model.set_new_modes(new_mode_specs, nests_spec=nests_spec)

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))


handler=CS_Handler(this_model)

# =============================================================================
# perform an update with random input data
# =============================================================================
geogrid_data=handler.random_geogrid_data()
handler.model.update_simulation(geogrid_data)
print(handler.get_outputs())

# =============================================================================
# Perform multiple random updates, saving te inputs and outputs
# =============================================================================
X, Y = handler.generate_training_data(iterations=3)

all_persons=this_model.pop.base_sim+this_model.pop.new
print(this_model.get_mode_split(all_persons))


