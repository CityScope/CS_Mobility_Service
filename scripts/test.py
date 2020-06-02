#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:52:41 2020

@author: doorleyr
"""

persons=this_model.pop.base_sim
self=this_model

from copy import deepcopy

temp_mode_choice_model=deepcopy(self.mode_choice_model)
all_trips=[]
for person_id, p in enumerate(persons):
    all_trips.extend(p.trips_to_list(person_id=person_id))
temp_mode_choice_model.generate_feature_df(all_trips)
# =============================================================================
#         TODO: fix the drive time and PT times in building the Transport Network
# =============================================================================
temp_mode_choice_model.feature_df['drive_time_minutes']=temp_mode_choice_model.feature_df['drive_time_minutes']*3/4
temp_mode_choice_model.base_feature_df=deepcopy(temp_mode_choice_model.feature_df)     
# =============================================================================
if len(temp_mode_choice_model.new_alt_specs)>0:
    for new_spec in temp_mode_choice_model.new_alt_specs:
        temp_mode_choice_model.set_new_alt(new_spec)
temp_mode_choice_model.predict_modes()


test=temp_mode_choice_model.feature_df
test['P']=temp_mode_choice_model.predicted_modes
test_v=temp_mode_choice_model.predicted_v

test_long_data_df=temp_mode_choice_model.long_data_df

import matplotlib.pyplot as plt

plt.hist(test['network_dist_km'], bins=range(500))







data_in,=test_long_data_df
modelDict=this_model.mode_choice_model.logit_model
modelDict['params']['ASC for PT']
customIDColumnName, 
method='random'
seed=Nonealts={0:'drive', 1:'cycle', 2:'walk', 3:'PT'}