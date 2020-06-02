#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:25:03 2020

@author: doorleyr
"""

from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_choice_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC
from cs_handler import CS_Handler

# =============================================================================
# Create 2 new mode specs:
# dockless bikes and shuttle buses
# =============================================================================

bikeshare_spec={'name': 'bikeshare', 
                 'attrs':{'time_minutes': 'c*1'},  # dockless_time_minutes = cycle_time_minutes * 0.7
                 'copy': 'cycle',
                 "copy_route": "cycling","activity": "cycling","speed_m_s": 4.167,
                 "co2_emissions_kg_met": 0,"fixed_costs": {}}
shuttle_spec={'name': 'shuttle', 
                 'attrs':{'time_minutes': 'd*1'},  # dockless_time_minutes = cycle_time_minutes * 0.7
                 'copy': 'PT',
                 "copy_route": "driving","activity": "pt","speed_m_s": 8.33,
                 "co2_emissions_kg_met": 0.000066,"fixed_costs": {}}

new_mode_specs=[bikeshare_spec, shuttle_spec]

nests_spec=[{'name': 'pt_like', 
             'alts':['PT','bikeshare', 'shuttle'], 
             'lambda':0.3},
             {'name': 'walk_like',
             'alts': ['walk', 'bikeshare'],
             'lambda':0.5}
            ]
    
# =============================================================================
# Create the model and add it to a handler
# =============================================================================
this_model=MobilityModel('corktown', 'Detroit')

this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))

this_model.assign_mode_choice_model(NhtsModeLogit(table_name='corktown', city_folder='Detroit'))

this_model.assign_home_location_choice_model(
        TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                         geogrid=this_model.geogrid, 
                         base_vacant_houses=this_model.pop.base_vacant))


handler=CS_Handler(this_model)
print(handler.get_outputs())

# =============================================================================
# perform an update with random input data
# =============================================================================
geogrid_data=handler.random_geogrid_data()
handler.model.update_simulation(geogrid_data)
print(handler.get_outputs())


# =============================================================================
# Perform multiple random updates, saving the inputs and outputs
# =============================================================================
X, Y = handler.generate_training_data(iterations=500)



# =============================================================================
# Mobility interventions
# =============================================================================
this_model.set_prop_electric_cars(0.5)
this_model.set_new_modes(new_mode_specs, nests_spec=nests_spec)

pt_similarity = 0.7
params_for_share_bike = {}
existing_params = this_model.mode_choice_model.logit_model['params']
for g_attr in this_model.mode_choice_model.logit_generic_attrs:
    params_for_share_bike['{} for bikeshare'.format(g_attr)] = \
        existing_params['{} for PT'.format(g_attr)] * pt_similarity + \
        existing_params['{} for walk'.format(g_attr)] * (1-pt_similarity)
params_for_share_bike['ASC for bikeshare'] = existing_params['ASC for PT'] * pt_similarity + \
        existing_params['ASC for walk'] * (1-pt_similarity)
this_model.mode_choice_model.set_logit_model_params(params_for_share_bike)

geogrid_data=handler.random_geogrid_data()
handler.model.update_simulation(geogrid_data)
print(handler.get_outputs())
handler.post_trips_data()

X_future_m, Y_future_m = handler.generate_training_data(iterations=500)




