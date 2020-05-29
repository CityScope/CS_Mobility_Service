#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:20:55 2020

@author: doorleyr
"""
import pandas as pd
import random
import json
import numpy as np
from transport_network import get_haversine_distance


class Activity():
    def __init__(self, activity_id, start_time, activity_name, location):
        self.activity_id=activity_id
        self.location=location
        self.name=activity_name
        self.start_time=start_time
    
class ActivityScheduler():
    def __init__(self, model):
        ACTIVITY_NAME_PATH='./cities/'+model.city_folder+'/mappings/activities.json'
        self.activity_names=json.load(open(ACTIVITY_NAME_PATH))
        MOTIF_SAMPLE_PATH='./cities/'+model.city_folder+'/clean/motif_samples.csv'
        sample_motifs=pd.read_csv(MOTIF_SAMPLE_PATH)
        self.motif_sample_obj=sample_motifs.to_dict(orient='records')
        
    def sample_activity_schedules(self, person, model):
        last_loc=person.home_loc
        activities=[]
        # TODO predict rather than random sample
        motif=random.choice(self.motif_sample_obj)
        hourly_activity_ids=[motif['P{}'.format(str(period).zfill(3))] for period in range(24)]
#        print(hourly_activity_ids)
        for t, a_id in enumerate(hourly_activity_ids):
            if ((t == 0) or (a_id != hourly_activity_ids[t-1])):
                activity_start_time=t*3600+ np.random.randint(3600)
                activity_name=self.activity_names[a_id]
#                print(activity_name)
                if activity_name=='Home':
                    activity_location=person.home_loc
                elif activity_name=='Work':
                    activity_location=person.work_loc
                else:
                    potential_locations=model.geogrid.find_locations_for_activity(activity_name)
                    if len(potential_locations)==0:
#                        print('No locations for {} in geogrid'.format(activity_name))
                        potential_locations=[z for z in model.zones if not z.in_sim_area]
                    dist=[get_haversine_distance(last_loc.centroid, loc.centroid
                                                 )for loc in potential_locations]
                    prob, chosen_idx =self.huff_model(dist, beta=2, predict_y=True, topN=5)
                    activity_location=potential_locations[chosen_idx[0]]
                activities.append(Activity(activity_id=a_id, 
                                           start_time=activity_start_time,
                                           activity_name=activity_name,
                                           location=activity_location))
                last_loc=activity_location
        person.assign_activities(activities)

    def huff_model(self, dist, attract=None, alt_names=None, alpha=1, beta=2, predict_y=False, topN=None):
        """ 
        takes a distance matrix and a optional attraction matrix, calculates choice probabilities 
        and predicts choice outcomes by sampleing according to probabilities
        prob = (attract**alpha / dist**beta) / sum_over_all_alternatives(attract**alpha / dist**beta)
        
        Arguments:
        --------------------------------------------
        dist: distance matrix, ncs(number of choice situations) * nalt(number of alternatives), or 1-d array
        attract: optional attraction matrix, ncs * nalt, or 1-d array
        alt_names: optional matrix of alternative names, ncs * nalt, or 1-d array
        alpha, beta: coefficents of attraction and distance
        predict_y: whether or not to predict choice outcomes via sampling
        topN: when predicting choice outcomes, only alternatives with top N probabilities will be considered
        """
        dist = np.array(dist)
        dist = np.maximum(dist, np.ones_like(dist)*0.01)    # avoid dist=0
        if attract is None:
            attract = np.ones_like(dist)
        else:
            attract = np.array(attract)
        if dist.ndim == 1:
            dist = dist.reshape(1, -1)
            attract = attract.reshape(1, -1)
            if alt_names is not None:
                alt_names = alt_names.reshape(1, -1)
        ncs, nalt = dist.shape
        u = (attract ** alpha) / (dist ** beta)
        prob = u / u.sum(axis=1, keepdims=True)
        if predict_y:
            y = []
            if topN:
                use_prob = -np.sort(-prob, axis=1)[:, :topN]
                use_prob = use_prob / use_prob.sum(axis=1, keepdims=True)
                use_idx = np.argsort(-prob, axis=1)[:, :topN]
                if alt_names is None:
                    use_names = use_idx
                else:
                    use_names = np.asarray([alt_names[i, use_idx[i,:]] for i in range(ncs)])
            else:
                use_prob = prob
                if alt_names is None:
                    use_names = np.asarray([list(range(nalt)) for i in range(ncs)])
                else:
                    use_names = alt_names
            for i in range(ncs):
                this_y = np.random.choice(use_names[i, :], p=use_prob[i, :])
                y.append(this_y) 
        else:
            y = None
        return prob, y        
    
# acts= Activity_Scheduler(this_world)               
