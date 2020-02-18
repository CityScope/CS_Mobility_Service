#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:50:11 2018

@author: doorleyr
"""
import pickle
import json
import random
import urllib
#import pyproj
import math
import pandas as pd
import numpy as np
#import networkx as nx
from scipy import spatial
import requests
from time import sleep
import time
#from shapely.geometry import Point, shape
import matplotlib.path as mplPath
import sys
import time
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

if len(sys.argv)>1:
    city=sys.argv[1]
else:
    city='Detroit'

# =============================================================================
# Functions
# =============================================================================

def get_haversine_distance(point_1, point_2):
    """
    Calculate the distance between any 2 points on earth given as [lon, lat]
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [point_1[0], point_1[1], 
                                                point_2[0], point_2[1]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def approx_shape_centroid(geometry):
    if geometry['type']=='Polygon':
        centroid=list(np.mean(geometry['coordinates'][0], axis=0))
        return centroid
    elif geometry['type']=='MultiPolygon':
        centroid=list(np.mean(geometry['coordinates'][0][0], axis=0))
        return centroid
    else:
        print('Unknown geometry type')
    
def get_simulation_locations(persons):
    """
    For each agent, based on their actual home and work zones (zone geoids or geogrid cells)
    a "simulation" home and a workplace are assigned which may be metagrid cells or portals
    if the location is a geogrid cell, the assigned location is {type: geogrid, id grid_cell_id}
    if location is a portal, only the 'type' of 'portal' is assigned 
    and the portal id will be identified later based on the route attributes
    """
    # TODO: new persons have geo of grid_
    # then no lookup required- remove tui_cell_to_meta_grid
    # remove distinction between static and tui land uses- just keep 1 list
    for p in persons:  
        for place in ['home', 'work']:
            geoid=p[place+'_geoid']
            if 'grid' in str(geoid):
                p[place+'_sim']={'type': 'geogrid', 
                                 'ind': int(geoid[4:])}
            elif p[place+'_geoid'] in sim_area_zone_list:
                if place == 'work':
                    relevant_land_use_codes=employment_lus
                else:
                    relevant_land_use_codes=housing_lus
                possible_locations=[static_land_uses[rlu] for rlu in relevant_land_use_codes if rlu in static_land_uses]
                possible_locations=[item for sublist in possible_locations for item in sublist]
                p[place+'_sim']={'type': 'geogrid', 
                                 'ind': random.choice(possible_locations)}
            else:
                p[place+'_sim']={'type': 'portal'}


def get_LLs(persons, places):
    # TODO: remove function only used in HLC and OD end-point (supposed to be removed from GAMA model anyway)
    """ takes a list of person objects and 
    finds home and work coordinates for them
    modifies in place
    """
    # TODO: if grid in 
    # grid_points, not tui points
    for p in persons:  
        for place in places:
            geoid=p[place+'_geoid']
            if 'grid' in str(geoid):
                ll=geogrid['features'][int(geoid[4:])]['properties']['centroid']
            else:
                ll=[all_geoid_centroids[geoid][0]+np.random.normal(0, 0.002, 1)[0], 
                    all_geoid_centroids[geoid][1]+np.random.normal(0, 0.002, 1)[0]]
            p[place+'_ll']=ll
            
#def approx_route_costs(start_coord, end_coord):
#    approx_speeds_met_s={'driving':20/3.6,
#        'cycling':10/3.6,
#        'walking':3/3.6,
#        'pt': 15/3.6 
#        }
#    straight_line_commute=get_haversine_distance(start_coord, end_coord)
#    routes={}
#    for mode in range(4):
#        routes[mode]={'route': {'driving':0, 'walking':0, 'waiting':0,
#                      'cycling':0, 'pt':0}, 'external_time':0}
#    routes[0]['route']['driving']=(straight_line_commute/approx_speeds_met_s['driving'])/60
#    routes[1]['route']['cycling']=(straight_line_commute/approx_speeds_met_s['cycling'])/60
#    routes[2]['route']['walking']=(straight_line_commute/approx_speeds_met_s['walking'])/60
#    routes[3]['route']['pt']=(straight_line_commute/approx_speeds_met_s['pt'])/60
#    routes[3]['route']['walking']=(200/approx_speeds_met_s['walking'])/60
#    return routes
    
def get_person_type(persons):
    for ip, p in enumerate(persons):
        if ((p['home_sim']['type']=='geogrid') and  (p['work_sim']['type']=='geogrid')):
            p['type']=0 # lives and works on site
        elif p['work_sim']['type']=='geogrid':
            p['type']=1 # commute_in
        elif p['home_sim']['type']=='geogrid':
            p['type']=2 # commute_out
        else:
            print('None')
               
#def get_route_costs(persons):
#    """ takes a list of person objects 
#    and finds the travel time costs of travel by each mode
#    modifies in place
#    """
#    for ip, p in enumerate(persons):
#        p['routes']={}
#        if ((p['home_sim']['type']=='geogrid') and  (p['work_sim']['type']=='geogrid')):
#            p['type']=0 # lives and works on site
##            home_coord=geogrid['features'][p['home_sim']['ind']]['properties']['centroid']
##            work_coord=geogrid['features'][p['work_sim']['ind']]['properties']['centroid']
#            home_node_list=geogrid['features'][p['home_sim']['ind']]['properties']['closest_nodes']
#            work_node_list=geogrid['features'][p['work_sim']['ind']]['properties']['closest_nodes']
#
##            p['routes']=approx_route_costs(home_coord, work_coord)
#            p['routes']=internal_route_costs(home_node_list, work_node_list, 
#                         sim_net_floyd_result, nodes_to_link_attributes)
#        elif p['work_sim']['type']=='geogrid':
#            p['type']=1 # commute_in
##            work_coord=geogrid['features'][p['work_sim']['ind']]['properties']['centroid']
#            work_node_list=geogrid['features'][p['work_sim']['ind']]['properties']['closest_nodes']
#            for m in range(4):
#                best_portal=0 # TODO: fix this. Needs to be here in case all route costs are infinite
#                p['routes'][m]={}
#                best_portal_route_time=float('inf')
#                for portal in range(len(portals['features'])):
##                    portal_coord=portals['features'][portal]['properties']['centroid']
#                    portal_node_list=portals['features'][portal]['properties']['closest_nodes']
##                    internal_portal_route=approx_route_costs(portal_coord, work_coord)[m]
#                    internal_portal_route=internal_route_costs(portal_node_list, work_node_list, 
#                         sim_net_floyd_result, nodes_to_link_attributes)[m]
#                    external_portal_route=ext_route_costs[mode_graphs[m]][str(p['home_geoid'])][str(portal)]
#                    external_time=sum([external_portal_route[c] for c in external_portal_route])
#                    full_portal_route={c: internal_portal_route['route'][c] + external_portal_route[c] for
#                                       c in ['driving', 'walking', 'waiting','cycling', 'pt']}
#                    total_portal_route_time=sum([full_portal_route[c] for c in full_portal_route])
#                    if total_portal_route_time<best_portal_route_time:
#                        best_portal=portal
#                        best_route=full_portal_route
#                        best_external_time=external_time
#                        best_portal_route_time=total_portal_route_time                    
#                p['routes'][m]['portal']=best_portal
#                p['routes'][m]['external_time']= int(best_external_time*60)
#                p['routes'][m]['route']=best_route
#        elif p['home_sim']['type']=='geogrid':
#            p['type']=2 # commute_out
##            home_coord=geogrid['features'][p['home_sim']['ind']]['properties']['centroid']
#            home_node_list=geogrid['features'][p['home_sim']['ind']]['properties']['closest_nodes']
#            for m in range(4):
#                best_portal=0 # TODO: fix this. Needs to be here in case all route costs are infinite
#                p['routes'][m]={}
#                best_portal_route_time=float('inf')
#                for portal in range(len(portals['features'])):
##                    portal_coord=portals['features'][portal]['properties']['centroid']
#                    portal_node_list=portals['features'][portal]['properties']['closest_nodes']
##                    internal_portal_route=approx_route_costs(home_coord, portal_coord)[m]
#                    internal_portal_route=internal_route_costs(home_node_list, portal_node_list, 
#                         sim_net_floyd_result, nodes_to_link_attributes)[m]
#                    external_portal_route=ext_route_costs[mode_graphs[m]][str(p['work_geoid'])][str(portal)]
#                    external_time=sum([external_portal_route[c] for c in external_portal_route])
#                    full_portal_route={c: internal_portal_route['route'][c] + external_portal_route[c] for
#                                       c in ['driving', 'walking', 'waiting','cycling', 'pt']}
#                    total_portal_route_time=sum([full_portal_route[c] for c in full_portal_route])
#                    if total_portal_route_time<best_portal_route_time:
#                        best_portal=portal
#                        best_route=full_portal_route
#                        best_external_time=external_time
#                        best_portal_route_time=total_portal_route_time                    
#                p['routes'][m]['portal']=best_portal
#                p['routes'][m]['external_time']= 0
#                p['routes'][m]['route']=best_route
#                
#def predict_modes(persons):
#    """ takes list of person objects and 
#    predicts transport modes for each person's commute
#    modifies in place
#    """
#    feature_df=pd.DataFrame(persons)  
##    feature_df['bach_degree_yes']=feature_df['SCHL']>20
##    feature_df['bach_degree_no']=~feature_df['bach_degree_yes']
#    for feat in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 
#                 'bach_degree', 'race', 'cars']:
#        new_dummys=pd.get_dummies(feature_df[feat], prefix=feat)
#        feature_df=pd.concat([feature_df, new_dummys],  axis=1)
#    # TODO: better method of predicting travel times
#    # routing engine or feedback from simulation
#    feature_df['drive_time_minutes']=  feature_df.apply(lambda row: row['routes'][0]['route']['driving'], axis=1)     
#    feature_df['cycle_time_minutes']=  feature_df.apply(lambda row: row['routes'][1]['route']['cycling'], axis=1)     
#    feature_df['walk_time_minutes']=  feature_df.apply(lambda row: row['routes'][2]['route']['walking'], axis=1)     
#    feature_df['PT_time_minutes']=  feature_df.apply(lambda row: row['routes'][3]['route']['pt'], axis=1)
#    feature_df['walk_time_PT_minutes']=feature_df.apply(lambda row: row['routes'][3]['route']['walking'], axis=1)  
#    feature_df['drive_time_PT_minutes']=0 
#    # TODO: below should come directly from the path-finding
#    feature_df['network_dist_km']=feature_df.apply(lambda row: row['drive_time_minutes']*30/60, axis=1) 
#    # TODO: change below if modelling housing sales as well
#    feature_df['tenure_owned']=False
#    feature_df['tenure_other']=False
#    feature_df['purpose_HBW']=1
#    feature_df['purpose_NHB']=0
#    feature_df['purpose_HBO']=0
#    feature_df['race_asian']=0
#    for rff in rf_features:
#        # feature_df[']
#        assert rff in feature_df.columns, str(rff) +' not in data.'
##    assert all([rff in feature_df.columns for rff in rf_features]
##    ),"Features in table dont match features in RF model"   
#    feature_df=feature_df[rf_features]#reorder columns to match rf model
#    mode_probs=mode_rf.predict_proba(feature_df)
#    for i,p in enumerate(persons): 
#        chosen_mode=int(np.random.choice(range(4), size=1, replace=False, p=mode_probs[i])[0])
#        p['mode']=chosen_mode
#        if p['home_sim']['type']=='portal': 
#            p['home_sim']['ind']=p['routes'][chosen_mode]['portal']
#            p['home_sim']['ll']=portals['features'][p['home_sim']['ind']]['properties']['centroid']
#            p['work_sim']['ll']=geogrid['features'][p['work_sim']['ind']]['properties']['centroid']
#        elif p['work_sim']['type']=='portal': 
#            p['work_sim']['ind']=p['routes'][chosen_mode]['portal']
#            p['work_sim']['ll']=portals['features'][p['work_sim']['ind']]['properties']['centroid']
#            p['home_sim']['ll']=geogrid['features'][p['home_sim']['ind']]['properties']['centroid']
#        else:
#            p['home_sim']['ll']=geogrid['features'][p['home_sim']['ind']]['properties']['centroid']
#            p['work_sim']['ll']=geogrid['features'][p['work_sim']['ind']]['properties']['centroid']
#        p['external_time']=p['routes'][chosen_mode]['external_time']
        
        
def sample_activity_schedules(persons):
    # TODO predict rather than random sample
    for p in persons:
        motif=random.choice(motif_sample_obj)
        activities=[motif['P{}'.format(str(period).zfill(3))] for period in range(24)]
        p['activities']=activities
        p['motif_name']=motif['cluster_name']
        p['motif_id']=motif['cluster']
        
def get_standard_lu_from_base(land_use_input):
    if land_use_input is None:
        return None
    elif land_use_input=='None':
        return None
    else:
        return base_lu_to_lu[land_use_input]
    
def get_standard_lu_from_input_lu(input_index):
    try: 
        input_lu_name=lu_inputs[str(input_index)]
    except:
        return None
    possible_standard_lus=lu_input_to_lu_standard[input_lu_name]
    return np.random.choice([k for k in possible_standard_lus],
                                 1, p=[v for v in possible_standard_lus.values()])[0]
    
#def post_od_data(persons, destination_address):
#    od_str=json.dumps([{'home_ll': p['home_sim']['ll'],
#                       'work_ll': p['work_sim']['ll'],
#                       'home_sim': p['home_sim'],
#                       'work_sim': p['work_sim'],
#                       'type': p['type'],
#                       'mode': p['mode'],
#                       'activities': p['activities'],
#                       'motif': p['motif_name'],
##                       'activity_start_times':p['start_times'],
#                       'start_time': 6*3600+random.randint(0,3*3600)+p['external_time']
#                       } for p in persons if len(p['activities'])>1])
#    try:
#        r = requests.post(destination_address, data = od_str)
#        print('OD: {}'.format(r))
#    except requests.exceptions.RequestException as e:
#        print('Couldnt send to cityio')
    
              
def create_long_record_puma(person, puma):
    # TODO: dont use work_ll- lookup the location from geogrid locations and zone locations
    """ takes a puma object and a household object and 
    creates a row for the MNL long data frame 
    """
    return   {'puma_pop_per_sqm': puma['puma_pop_per_sqm'],
              'income_disparity': np.abs(person['HINCP'] - puma['med_income']),
              'work_dist': get_haversine_distance(person['work_ll'], puma['centroid']),
              'media_norm_rent': puma['media_norm_rent'],
              'num_houses': puma['num_houses'],
#              'entertainment_den': puma['entertainment_den'],
#              'medical_den': puma['medical_den'],
#              'school_den': puma['school_den'],
              'custom_id': person['person_id'],
              'choice_id': puma['puma']} 

def create_long_record_house(person, house, choice_id):
    """ takes a house object and a household object and 
    creates a row for the MNL long data frame 
    """
    beds=min(3, max(1, house['beds']))
    norm_rent=(house['rent']-rent_normalisation['mean'][str(int(beds))])/rent_normalisation['std'][str(int(beds))]
    record = {'norm_rent': norm_rent,
            'built_since_jan2010': house['built_since_jan2010'],
            'bedrooms': house['beds'],
            'income': person['HINCP'],
            'custom_id': person['person_id'],
            'choice_id': choice_id,
            'actual_house_id':house['house_id']} 
    nPersons = 0
    if person['workers'] == 'one':
        nPersons += 1
    elif person['workers'] == 'two or more':
        nPersons += 2
    if person['children'] == 'yes':
        nPersons += 1
    if nPersons == 0:
        nPersons = 1
    record['nPersons'] = nPersons
    return record
    
#
def utility_to_prob(v):
    """ takes a utility vector and predicts probability 
    """
    v = v - v.mean()
    v[v>700] = 700
    v[v<-700] = -700
    expV = np.exp(v)
    p = expV / expV.sum()
    return p
    
def unique_ele_and_keep_order(seq):
    """ same as list(set(seq)) while keep element order 
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
    
def pylogit_pred(data, modelDict, customIDColumnName, even=True):
    """ predicts probabilities for pylogit models,
    this function is needed as the official 'predict' method cannot be used when the choice sets 
    in predicting is not exactly the same as in trainning,
    argument even: whether each choice situation has the same number of alternatives
    """
    # fectch variable names and parameters 
    if modelDict['just_point']:
        params, varnames = modelDict['params'], modelDict['var_names']
    else:
        params, varnames = list(modelDict['model'].coefs.values), list(modelDict['model'].coefs.index)
    # calc utilities
    data['utility'] = 0
    for varname, param in zip(varnames, params):
        data['utility'] += data[varname] * param
    # calc probabilities given utilities
    # if every choice situation has the same number of alternatives, use matrix, otherwise use list comprehension
    if even:
        numChoices = len(set(data[customIDColumnName]))
        v = np.array(data['utility']).copy().reshape(numChoices, -1)
        v = v - v.mean(axis=1, keepdims=True)  
        v[v>700] = 700
        v[v<-700] = -700
        expV = np.exp(v)
        p = expV / expV.sum(axis=1, keepdims=True)
        return p.flatten()
    else:
        uniqueCustomIDs = unique_ele_and_keep_order(data[customIDColumnName])
        vArrayList = [np.array(data.loc[data[customIDColumnName]==id, 'utility']) for id in uniqueCustomIDs]
        pArrayList = [utility_to_prob(v) for v in vArrayList]
        return [pElement for pArray in pArrayList for pElement in pArray ]
    
def home_location_choices(houses, persons):
    """ takes the house and person objects
    finds the vacant houses and homeless persons
    chooses a housing unit for each person
    modifies the house and person objects in place
    """
    # TODO: 'tui' to 'grid'
    # preparing
    valid_pumas = list(set([h['puma10'] for h in houses]))
    valid_puma_objs = [puma_obj_dict[puma] for puma in valid_pumas]
    puma_to_houses = {puma: [h for h in houses if h['puma10']==puma] for puma in valid_pumas}
    for h in houses:        # updating number of houses in each puma given new houses
        if h['home_geoid'].startswith('grid'):
            puma_index = valid_pumas.index(h['puma10'])
            valid_puma_objs[puma_index]['num_houses'] += 1
    # stage1: PUMA choice
    long_data_puma = []
    for p in persons:
        for puma in valid_puma_objs:
            this_sample_long_record_puma = create_long_record_puma(p, puma)
            long_data_puma.append(this_sample_long_record_puma)
    long_df_puma = pd.DataFrame(long_data_puma)
    home_loc_mnl_puma = home_loc_logit['home_loc_mnl_PUMAs']
    long_df_puma['predictions'] = pylogit_pred(long_df_puma, home_loc_mnl_puma, 'custom_id', even=True)   
    if top_n_pumas is None:
        custom_specific_long_df_puma = {custom_id: group for custom_id, group in long_df_puma[['custom_id', 'choice_id', 'predictions']].groupby('custom_id')}
    else:
        long_df_puma_sorted = long_df_puma[['custom_id', 'choice_id', 'predictions']].sort_values(['custom_id','predictions'], ascending=[True, False])
        custom_specific_long_df_puma = {custom_id: group.iloc[:top_n_pumas,:] for custom_id, group in long_df_puma_sorted.groupby('custom_id')}
    for p_ind in set(long_df_puma['custom_id']):
        if top_n_pumas is None:
            house_puma=np.random.choice(custom_specific_long_df_puma[p_ind]['choice_id'], p=custom_specific_long_df_puma[p_ind]['predictions'])
        else:
            house_puma=np.random.choice(custom_specific_long_df_puma[p_ind]['choice_id'], 
                                        p=custom_specific_long_df_puma[p_ind]['predictions'] / custom_specific_long_df_puma[p_ind]['predictions'].sum())
        persons[p_ind]['house_puma'] = house_puma
    # stage2: housing unit choice
    long_data_house = []
    even = True  # use "even" to monitor if every choice situation has the same number of alternatives
    for p in persons:
        houses_in_puma = puma_to_houses[p['house_puma']]
        if len(houses_in_puma) < 9:
            house_alts = houses_in_puma 
            even = False
        else:
            house_alts = random.sample(houses_in_puma, 9)
        for hi, h in enumerate(house_alts):
            this_sample_long_record_house = create_long_record_house(p, h, hi+1)
            long_data_house.append(this_sample_long_record_house)             
    long_df_house = pd.DataFrame(long_data_house)
    long_df_house.loc[long_df_house['norm_rent'].isnull(), 'norm_rent']=0
    long_df_house['income_norm_rent'] = long_df_house['income'] * long_df_house['norm_rent']
    long_df_house['income_bedrooms'] = long_df_house['income'] * long_df_house['bedrooms']
    long_df_house['nPerson_bedrooms'] = long_df_house['nPersons'] * long_df_house['bedrooms']
    home_loc_mnl_house = home_loc_logit['home_loc_mnl_hh']
    long_df_house['predictions'] = pylogit_pred(long_df_house, home_loc_mnl_house, 'custom_id', even=even)
    custom_specific_long_df_house = {custom_id: group for custom_id, group in long_df_house[['custom_id', 'actual_house_id', 'predictions']].groupby('custom_id')}
    for p_ind in set(long_df_house['custom_id']):
        house_id = np.random.choice(custom_specific_long_df_house[p_ind]['actual_house_id'], 
                                    p=custom_specific_long_df_house[p_ind]['predictions'])             
        persons[p_ind]['house_id']=house_id
        persons[p_ind]['home_geoid']=houses[house_id]['home_geoid']

#def shannon_equitability(species_pop, species_set):
#    diversity=0
#    pop_size=len(species_pop)
#    if pop_size>0:
#        for species in species_set:
#            pj=species_pop.count(species)/len(species_pop)
#            if not pj==0:
#                diversity+= -pj*np.log(pj)
#        equitability=diversity/np.log(len(species_set))
#        return equitability
#    else:
#        return 0
#            
#def get_pop_diversity(persons):
#    diversity={}
#    dims=['age', 'income']
#    for d in dims:
#        dim_all_persons=[p[d] for p in persons]
#        diversity[d]=shannon_equitability(dim_all_persons, set(dim_all_persons))
#    return diversity
        
#def get_lu_diversity(grid_data):
#    # TODO: incorporate num floors
#    lu_diversity={}
#    housing_pop=[lu for lu in grid_data if lu in housing_lus]
#    lu_diversity['housing']=shannon_equitability(housing_pop, set(housing_lus))
#    office_pop=[lu for lu in grid_data if lu in employment_lus]
#    lu_diversity['office']=shannon_equitability(office_pop, set(employment_lus))
#    return lu_diversity

#def post_diversity_indicators(pop_diversity, lu_diversity, destination_address):
#    all_diversity={}
#    for var in pop_diversity:
#        all_diversity[var]=pop_diversity[var]
#    for var in lu_diversity:
#        all_diversity[var]=lu_diversity[var]
#    try:
#        r = requests.post(destination_address, data = json.dumps(all_diversity))
#        print('Diversity Indicators: {}'.format(r))
#    except requests.exceptions.RequestException as e:
#        print('Couldnt send diversity indicators to cityio')
        
#def get_path_from_fw_multi(fw_result, sim_net_list, from_node_list, to_node_list):
#    for fn in from_node_list:
#        for tn in to_node_list:
#            try:
#                return get_path_from_fw(fw_result, sim_net_list, fn, tn)
#            except:
#                pass
#    print('No path found')
#    return None, None, None
        
def get_node_path_from_fw_try_multi(sim_net_floyd_result, from_list, to_list):
    for fn in from_list:
        for tn in to_list:
            try: 
                node_path=get_node_path_from_fw(sim_net_floyd_result, 
                                                fn, tn)
                return node_path
            except:
                pass
#    print('No path found')
    return None     
  
def get_node_path_from_fw(sim_net_floyd_result, from_node, to_node):
    if from_node==to_node:
        return []
    pred=to_node
    path=[pred]
    while not pred==from_node:
        pred=sim_net_floyd_result[from_node][pred]
        path.insert(0,pred)
    return path
        
def get_path_coords_distances(nodes_to_link_attributes, path):
    coords, distances =[], []
    if len(path)>1:
        for node_ind in range(len(path)-1):
            from_node=path[node_ind]
            to_node=path[node_ind+1]
            link_attributes=nodes_to_link_attributes['{}_{}'.format(from_node, to_node)]
            distances+=[link_attributes['distance']]
            coords+=[[reduce_precision(link_attributes['from_coord'][0], 5),
                     reduce_precision(link_attributes['from_coord'][1], 5)]]
        coords+= [link_attributes['to_coord']]
        # add the final coordinate of the very last segment
    return coords, distances

def internal_route_costs(from_node_list, to_node_list, 
                         sim_net_floyd_result, nodes_to_link_attributes):
    path=get_node_path_from_fw_try_multi(sim_net_floyd_result, from_node_list, to_node_list)
    if path is None:
        coords, distances, total_distance=[], [], float('1e10')
    else:
        coords, distances=get_path_coords_distances(nodes_to_link_attributes, path)
        total_distance=sum(distances)
    routes={}
    for mode in range(4):
        routes[mode]={'route': {'driving':0, 'walking':0, 'waiting':0,
                      'cycling':0, 'pt':0}, 'external_time':0}
    routes[0]['route']['driving']=(total_distance/SPEEDS_MET_S[0])/60
    routes[1]['route']['cycling']=(total_distance/SPEEDS_MET_S[1])/60
    routes[2]['route']['walking']=(total_distance/SPEEDS_MET_S[2])/60
    routes[3]['route']['pt']=(total_distance/SPEEDS_MET_S[3])/60
    routes[3]['route']['walking']=(200/SPEEDS_MET_S[2])/60
    routes['total_distance'] = total_distance
    routes['node_path'] = path
    routes['cum_dist']=np.cumsum(distances)
    routes['coords']=coords
    return routes
    
    
def external_route_costs(out_geoid, grid_node_list, direction):
    """
    calcuating route costs between a in-site location (grid_node_list) and 
    a outside location (out_geoid)
    directions = 'in' (origin is outside) or 'out' (destination is outside)
    """
    routes = {}
    portal_specific_internal_routes = {portal:{} for portal in range(len(portals['features']))} 
    for portal in portal_specific_internal_routes:
        portal_node_list=portals['features'][portal]['properties']['closest_nodes']
        if direction == 'in':
            internal_portal_route_all_modes = internal_route_costs(portal_node_list, grid_node_list, 
                sim_net_floyd_result, nodes_to_link_attributes)
        elif direction == 'out':
            internal_portal_route_all_modes = internal_route_costs(grid_node_list, portal_node_list, 
                sim_net_floyd_result, nodes_to_link_attributes)
        portal_specific_internal_routes[portal] = internal_portal_route_all_modes
    for m in range(4):
        routes[m] = {}
        best_portal_route_time = np.inf
        best_portal=None
        for portal in portal_specific_internal_routes:
            internal_portal_route = portal_specific_internal_routes[portal][m]
            external_portal_route = ext_route_costs[mode_graphs[m]][str(out_geoid)][str(portal)]
            external_time = sum([external_portal_route[c] for c in external_portal_route])
            full_portal_route={c: internal_portal_route['route'][c] + external_portal_route[c] for
                               c in ['driving', 'walking', 'waiting','cycling', 'pt']}
            total_portal_route_time=sum([full_portal_route[c] for c in full_portal_route])
            if total_portal_route_time<best_portal_route_time:
                best_portal=portal
                best_route=full_portal_route
                best_interanl_route = internal_portal_route
                best_external_time=external_time
                best_portal_route_time=total_portal_route_time  
        if best_portal is not None:
            routes[m]['portal']=best_portal
            routes[m]['external_time']= int(best_external_time*60)
            routes[m]['route']=best_route
            routes[m]['internal_route'] = best_interanl_route
            routes[m]['node_path'] =  portal_specific_internal_routes[best_portal]['node_path']
            routes[m]['cum_dist'] =  portal_specific_internal_routes[best_portal]['cum_dist']
            routes[m]['coords'] =  portal_specific_internal_routes[best_portal]['coords']
    return routes


def find_destination(persons, land_uses, sampleN=15):
    """ 
    takes the person objects, choosing the place for each of his activities, 
    modifies in place: all results are stored in "activity_objs"
    work/home: use work_sim / home_sim; 
    others: if there are available meta grids in-site, use huff model to make choice; 
            otherwise randomly choose a external zone which contains the corresponding amenities
    
    Arguments:
    -------------------------------------
    persons: a list of person objects
    land_uses: a lookup dict, key: land use code, value: meta grid indices
    sampleN: the number of alternatives when choosing among meta grids, sampleN=None for all meta grids
    """
    # TODO: meta_grid to geogrid
    # ensure the input list of land uses is the definitve list
    # 'type': 'portal', 'geoid': geoid ??
    for p_id, person in enumerate(persons):
        activities_hourly = person['activities']
        activity_objs = [{'t': t*3600+ np.random.randint(3600),  'activity': a} for t, a in enumerate(activities_hourly) if t == 0 or a != activities_hourly[t-1]]
        for a_id, a_object in enumerate(activity_objs):
            t, a = a_object['t'], a_object['activity']
            if a == 'H':
                place = person['home_sim']
                if place['type']=='portal': place['geo_id']=person['home_geoid']
            elif a == 'W':
                place = person['work_sim']
                if place['type']=='portal': place['geo_id']=person['work_geoid']
            else:
                lu_config = activities_to_lu.get(activity_full_name[a], None)
                if len(lu_config) == 1:
                    lu_type = list(lu_config)[0]
                else:
                    lu_type = np.random.choice(list(lu_config), p=list(lu_config.values()))
                possible_lus = land_uses.get(lu_type, [])
                if ((sampleN) and len(possible_lus)>0):
                    possible_lus = np.random.choice(possible_lus, size=min(sampleN, len(possible_lus)), replace=False)
                possible_lus_ll = [geogrid['features'][idx]['properties']['centroid'] for idx in possible_lus]
                if len(possible_lus) > 1:
                    if a_id > 0:
                        last_place_sim = activity_objs[a_id-1]['place_sim']
                    else:
                        last_place_sim = person['home_sim']   # in case that "Home" is not the first activity, should not happen
                        last_place_sim['geo_id'] = person['home_geoid']
                    if last_place_sim['type'] == 'geogrid':
                        last_node_list = geogrid['features'][last_place_sim['ind']]['properties']['closest_nodes']
                        dist = [internal_route_costs(last_node_list, geogrid['features'][this_grid]['properties']['closest_nodes'], 
                            sim_net_floyd_result, nodes_to_link_attributes)['total_distance'] for this_grid in possible_lus]
                    else:
                        # too much time to calculate network distance between a outside geoid and in-site metagrid, use straighline for approx.
                        # dist = [external_routes(last_place_sim['geo_id'], geogrid['features'][this_grid]['properties']['closest_nodes'],
                            # direction='in')[0]['route']['driving']*30/60 for this_grid in possible_lus]
                        dist = [get_haversine_distance(all_geoid_centroids[last_place_sim['geo_id']], this_grid_ll) 
                            for this_grid_ll in possible_lus_ll]
                    prob, chosen_idx = huff_model(dist, beta=2, predict_y=True, topN=5, alt_names=possible_lus)
                    place = {'type': 'geogrid', 'ind': chosen_idx[0], 'll': geogrid['features'][chosen_idx[0]]['properties']['centroid']}
                elif len(possible_lus) == 1:
                    place = {'type': 'geogrid', 'ind': possible_lus[0], 'll': possible_lus_ll[0]}
                elif len(possible_lus) == 0:
                    # no available land use in site, randomly find a destination outside
                    geo_id = np.random.choice(external_lu[lu_type])
                    place = {'type': 'portal', 'geo_id': geo_id}
            if 'ind' in place:
                place['ind'] = int(place['ind'])     # np.int32 will cause error for json.dumps()
            activity_objs[a_id]['place_sim'] = place
        person['activity_objs'] = activity_objs
        person['start_times'] = [activity_objs[i]['t'] for i in range(1, len(activity_objs))]
        if len(person['start_times']) > 0:
            person['start_times'].append(person['start_times'][0])  # assuming the person will repeat the schedule the next day
        

def generate_ods(persons):
    """ 
    takes list of person objects and return a list of od objects
    an od object is a dict with all attributes of its affiliated person, plus:
        'o_loc', 'd_loc': location information (type, ind, ll, geo_id) of origin and destination
        'o_activity', 'd_activity': activities at origin and destination
        'activity_routes': routes (costs) returned by interal_route_costs or external_route_costs
        'mode': to be predicted by "predict_modes_for_activities"
        'internal_time_sec', 'external_time_sec': interal and external time in seconds from o to d
        'node_path': a list of network nodes from o to d
        ...
    """
    # TODO: does trip purpose get used
    # TODO: meta_grid to geogrid
    all_ods = [dict(p, o_loc=p['activity_objs'][idx]['place_sim'], d_loc=p['activity_objs'][idx+1]['place_sim'],
            o_activity=p['activity_objs'][idx]['activity'], d_activity=p['activity_objs'][idx+1]['activity'], 
            start_time=p['start_times'][idx], stay_until_time=p['start_times'][idx+1], od_id=idx)
            for p in persons for idx in range(0, len(p['activity_objs'])-1)]
    valid_ods = []
    for od in all_ods:
        if od['o_loc']['type'] == 'geogrid' and od['d_loc']['type'] == 'geogrid':
            o_node_list = geogrid['features'][od['o_loc']['ind']]['properties']['closest_nodes']
            d_node_list = geogrid['features'][od['d_loc']['ind']]['properties']['closest_nodes']
            activity_routes = internal_route_costs(o_node_list, d_node_list, 
                sim_net_floyd_result, nodes_to_link_attributes)
        elif od['o_loc']['type'] == 'geogrid' and od['d_loc']['type'] == 'portal':
            o_node_list = geogrid['features'][od['o_loc']['ind']]['properties']['closest_nodes']
            out_geoid = od['d_loc']['geo_id']
            activity_routes = external_route_costs(out_geoid, o_node_list, direction='out')
        elif od['o_loc']['type'] == 'portal' and od['d_loc']['type'] == 'geogrid':
            d_node_list = geogrid['features'][od['d_loc']['ind']]['properties']['closest_nodes']
            out_geoid = od['o_loc']['geo_id']
            activity_routes = external_route_costs(out_geoid, d_node_list, direction='in')
        else:
            continue    # omit ods with both o and d outside
        if len(activity_routes[0])>0:
            # ommits ODs where all routes were inf cost
            purpose_HBW, purpose_HBO, purpose_NHB = 0, 0, 0
            if od['o_activity'] + od['d_activity'] in ['HW', 'WH']:
                od['purpose'] = 'HBW'
                purpose_HBW = 1
            elif 'H' in [od['o_activity'] , od['d_activity']]:
                od['purpose'] = 'HBO'
                purpose_HBO = 1
            else:
                od['purpose'] = 'NHB'
                purpose_NHB = 1
            assert purpose_HBW + purpose_HBO + purpose_NHB == 1
            od['purpose_HBW'] = purpose_HBW
            od['purpose_HBO'] = purpose_HBO
            od['purpose_NHB'] = purpose_NHB 
            od['activity_routes'] = activity_routes
            valid_ods.append(od)
    return valid_ods


def predict_modes_for_activities(ods, persons=[]):
    """ 
    takes a list of od objects and predicts transport modes for each od, modifies in place
    
    Arguments:
    ---------------------------------
    ods: a list of od objects, returned by "generate_ods"
    persons: a list of persons who generate ods, modified in place to add new information of ods
    """
    # TODO check the lookup with new persons
    person_lookup = {p['person_id']: p for p in persons}
    feature_df=pd.DataFrame(ods)  
    for feat in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 
                 'bach_degree', 'race', 'cars']:
        new_dummys=pd.get_dummies(feature_df[feat], prefix=feat)
        feature_df=pd.concat([feature_df, new_dummys],  axis=1)
#    feature_df=feature_df.loc[len(feature_df['activity_routes'][0])>0]
    feature_df['drive_time_minutes'] = feature_df.apply(lambda row: row['activity_routes'][0]['route']['driving'], axis=1)     
    feature_df['cycle_time_minutes'] = feature_df.apply(lambda row: row['activity_routes'][1]['route']['cycling'], axis=1)     
    feature_df['walk_time_minutes'] = feature_df.apply(lambda row: row['activity_routes'][2]['route']['walking'], axis=1)     
    feature_df['PT_time_minutes'] = feature_df.apply(lambda row: row['activity_routes'][3]['route']['pt'], axis=1)
    feature_df['walk_time_PT_minutes'] = feature_df.apply(lambda row: row['activity_routes'][3]['route']['walking'], axis=1)  
    feature_df['drive_time_PT_minutes']=0 
    feature_df['network_dist_km']=feature_df.apply(lambda row: row['drive_time_minutes']*30/60, axis=1) 
    feature_df['tenure_owned']=False
    feature_df['tenure_other']=False
    feature_df['race_asian']=0
    for rff in rf_features:
        assert rff in feature_df.columns, str(rff) +' not in data.'
    feature_df=feature_df[rf_features] #reorder columns to match rf model
    
    mode_probs=mode_rf.predict_proba(feature_df)
    for i,od in enumerate(ods): 
        chosen_mode=int(np.random.choice(range(4), size=1, replace=False, p=mode_probs[i])[0])
        od['mode']=chosen_mode
        # TODO why 'v'?
        if od['o_loc']['type'] == 'geogrid' and od['d_loc']['type'] == 'geogrid':
            internal_route_mode = od['activity_routes'][chosen_mode]['route']
            external_time_sec = 0
            node_path = od['activity_routes']['node_path']
            cum_dist = od['activity_routes']['cum_dist']
            coords = od['activity_routes']['coords']
            time_to_enter_site=0
        elif od['o_loc']['type'] == 'portal' and od['d_loc']['type'] == 'geogrid':     #travel in
            internal_route_mode = od['activity_routes'][chosen_mode]['internal_route']['route']
            external_time_sec = od['activity_routes'][chosen_mode]['external_time']
            node_path = od['activity_routes'][chosen_mode]['node_path']
            od['o_loc']['ind'] = od['activity_routes'][chosen_mode]['portal']
            od['o_loc']['ll'] = portals['features'][od['activity_routes'][chosen_mode]['portal']]['properties']['centroid']
            cum_dist = od['activity_routes'][chosen_mode]['cum_dist']
            coords = od['activity_routes'][chosen_mode]['coords']
            time_to_enter_site=od['activity_routes'][chosen_mode]['external_time']
        elif od['o_loc']['type'] == 'geogrid' and od['d_loc']['type'] == 'portal':     #travel out  
            internal_route_mode = od['activity_routes'][chosen_mode]['internal_route']['route']
            external_time_sec = od['activity_routes'][chosen_mode]['external_time'] #or use external_time_sec=0?
            node_path = od['activity_routes'][chosen_mode]['node_path']
            cum_dist = od['activity_routes'][chosen_mode]['cum_dist']
            coords = od['activity_routes'][chosen_mode]['coords']
            time_to_enter_site=0
            od['d_loc']['ind'] = od['activity_routes'][chosen_mode]['portal']
            od['d_loc']['ll'] = portals['features'][od['activity_routes'][chosen_mode]['portal']]['properties']['centroid']
                        
        if chosen_mode == 0:
            internal_time_sec = int(internal_route_mode['driving']*60)
        elif chosen_mode == 1:
            internal_time_sec = int(internal_route_mode['cycling']*60)
        elif chosen_mode == 2:
            internal_time_sec = int(internal_route_mode['walking']*60)
        elif chosen_mode == 3:
            internal_time_sec = int((internal_route_mode['pt'] + internal_route_mode['walking'])*60)
        
        od['internal_time_sec'] = internal_time_sec
        od['external_time_sec'] = external_time_sec
        
        if od['person_id'] in person_lookup:
            person_lookup[od['person_id']]['activity_objs'][od['od_id']+1]['mode'] = chosen_mode
            person_lookup[od['person_id']]['activity_objs'][od['od_id']+1]['internal_time_sec'] = internal_time_sec
            person_lookup[od['person_id']]['activity_objs'][od['od_id']+1]['external_time_sec'] = external_time_sec
            person_lookup[od['person_id']]['activity_objs'][od['od_id']+1]['node_path'] = node_path
#            person_lookup[od['person_id']]['activity_objs'][od['od_id']+1]['cum_dist'] = cum_dist
            person_lookup[od['person_id']]['activity_objs'][od['od_id']+1]['coords'] = coords
            person_lookup[od['person_id']]['activity_objs'][od['od_id']+1]['cum_time_from_act_start']=[
                    time_to_enter_site]+[time_to_enter_site+ cd/SPEEDS_MET_S[chosen_mode] for cd in cum_dist]

def generate_detailed_schedules(persons):
    """ 
    takes a list of person objects and generate the attribute of 'sched_objs', modifies in place
    'sched_objs' is a list of detailed schedule objects to describe status of the person in different periods of a day
    a detailed sched object is dict for a certain period:
    {
        't': start_time in seconds for this period,
        'period': [start_time,  end_time]
        'status': 'stay' (stay in site) or 'trip' (trip in site) or 'out' (stay or trip outside),
        'activity': activity code if the person is doing an activity, None if the person is on trip
        'mode': 0~3 if the person is on trip, None if the person is doing an activity
        ...
    }
    if the person is doing an activity, the dict also contains 'place_sim' for location information
    if the person is on trip, the dict also contains 'o_loc' & 'd_loc' for location information, and 'node_path' for network path inforatmion
    """
    for p in persons:
        activity_objs = p['activity_objs']
        
        # the 1st activity: no trip
        sched_objs = [copy.deepcopy(activity_objs[0])]
        if sched_objs[0]['place_sim']['type'] == 'geogrid':
            sched_objs[0]['status'] = 'stay' 
        else:
            sched_objs[0]['status'] = 'out' 
        if len(activity_objs) == 1:
            sched_objs[0]['period'] = [activity_objs[0]['t'], 86400]
            p['sched_objs'] = sched_objs
            continue
        sched_objs[0]['period'] = [activity_objs[0]['t'], activity_objs[1]['t']]
        
        # following activities and trips
        for a_id in range(1, len(activity_objs)):
            stay_until_time_sec = activity_objs[a_id+1]['t'] if a_id < len(activity_objs)-1 else 86400
            if activity_objs[a_id-1]['place_sim']['type'] == 'geogrid':
                depart_time_sec = activity_objs[a_id]['t']
                arrive_time_sec = depart_time_sec + activity_objs[a_id]['internal_time_sec']
                if arrive_time_sec > stay_until_time_sec - 5*60:
                    arrive_time_sec = stay_until_time_sec - 5*60    # allow at least 5 mins stay for this activity
                internal_trip_obj = {'t': depart_time_sec, 'period': [depart_time_sec, arrive_time_sec], 
                    'status': 'trip', 'o_loc': activity_objs[a_id-1]['place_sim'], 'd_loc': activity_objs[a_id]['place_sim'],
                    'node_path': activity_objs[a_id]['node_path'], 'mode': activity_objs[a_id]['mode']}
                sched_objs.append(internal_trip_obj)
            elif activity_objs[a_id-1]['place_sim']['type'] == 'portal' and activity_objs[a_id]['place_sim']['type'] == 'geogrid':
                external_depart_time_sec = activity_objs[a_id]['t']
                internal_depart_time_sec = external_depart_time_sec + activity_objs[a_id]['external_time_sec']
                arrive_time_sec = internal_depart_time_sec + activity_objs[a_id]['internal_time_sec']
                if arrive_time_sec > stay_until_time_sec - 5*60:
                    arrive_time_sec = stay_until_time_sec - 5*60
                if internal_depart_time_sec >= arrive_time_sec:
                    internal_depart_time_sec = arrive_time_sec - 60  # allow at least 1min internal trip 
                external_trip_obj = {'t': external_depart_time_sec, 'period': [external_depart_time_sec, internal_depart_time_sec],
                    'status': 'out', 'mode': activity_objs[a_id]['mode']}
                internal_trip_obj = {'t': internal_depart_time_sec, 'period': [internal_depart_time_sec, arrive_time_sec],
                    'status': 'trip', 'o_loc': activity_objs[a_id-1]['place_sim'], 'd_loc': activity_objs[a_id]['place_sim'],
                    'node_path': activity_objs[a_id]['node_path'], 'mode': activity_objs[a_id]['mode']}
                sched_objs.extend([external_trip_obj, internal_trip_obj])
            elif activity_objs[a_id-1]['place_sim']['type'] == 'portal' and activity_objs[a_id]['place_sim']['type'] == 'portal':
                arrive_time_sec = activity_objs[a_id]['t']  #just assuming 'portal-portal' trip is external and omit it
            stay_obj = {key: activity_objs[a_id][key] for key in ['activity', 'place_sim']}
            stay_obj['t'] = arrive_time_sec
            stay_obj['period'] = [arrive_time_sec, stay_until_time_sec]
            if stay_obj['place_sim']['type'] == 'geogrid':
                stay_obj['status'] = 'stay'
            else:
                stay_obj['status'] = 'out'
            sched_objs.append(stay_obj)  
        p['sched_objs'] = sched_objs
                        
        
def huff_model(dist, attract=None, alt_names=None, alpha=1, beta=2, predict_y=False, topN=None):
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


def post_sched_data(persons, destination_address):
    sched_str = json.dumps([{'person_id': p['person_id'], 'sched_objs': p['sched_objs']}
        for p in persons])
    ## online post always return 413, have to save local file
    # with open('./sched.json', 'w') as f:
        # f.write(sched_str)
    try:
        r = requests.post(destination_address, data = sched_str)
        print('Detailed schedule: {}'.format(r))
    except requests.exceptions.RequestException as e:
        print('Couldnt send to cityio')
        
def reduce_precision(number, decimal_places):
    return int(number*10**decimal_places)/10**decimal_places

def create_trips_layer_v7(persons):
    trips=[]
    for p in persons:
        for a in p['activity_objs']:
            if 'coords' in a:
                if len(a['coords'])>1:
                    segments=[a['coords'][i]+[a['t']+ a['cum_time_from_act_start'][i]]
                       for i in range(len(a['coords']))]  
                    trips.append({'mode': [a['mode'], p['type']],
                                 'segments': segments})
    return trips
    
    
def create_trips_layer(persons):
    trips=[]
    for p in persons:
        for a in p['activity_objs']:
            if 'coords' in a:
                if len(a['coords'])>1:
                    trips.append({'path': a['coords'],
                                  'timestamps': [int(ti + a['t']) for ti in a['cum_time_from_act_start']],
                                  'mode': [a['mode'], p['type']]})
    return trips

def post_trips_data(trips_data, destination_address):
    trips_str = json.dumps(trips_data)
    try:
        r = requests.post(destination_address, data = trips_str)
        print('Trips Layer: {}'.format(r))
    except requests.exceptions.RequestException as e:
        print('Couldnt send to cityio')

def create_stay_data(persons):
    stays=[]
    for p in persons:
        for a_ind in range(len(p['activity_objs'])):
            if not p['activity_objs'][a_ind]['place_sim']['type']=='portal':
                if a_ind==0:
                    # first activity- no travel time t get there
                    start_time=p['activity_objs'][a_ind]['t']
                else:
                    start_time=p['activity_objs'][a_ind]['t']+ p['activity_objs'][a_ind]['cum_time_from_act_start'][-1]
                if len(p['activity_objs'])>(a_ind+1):
                    end_time=p['activity_objs'][a_ind+1]['t']
                else:
                    # last activity
                    end_time=24*60*60
                stays.append({'start_time': start_time,
                             'end_time': end_time,
                             'coords': p['activity_objs'][a_ind]['place_sim']['ll']})
    return stays
                
    
def get_realtime_route_position(node_path, period, current_time):
    """ 
    when a person is traveling on a route, predict his realtime lon-lat position,
    assuming the speed is known and constant
    
    Arguments:
    --------------------------------------------
    node_path: a list of network nodes to represent the route
    period: expected period ([start_time, end_time]) for passing this route
    current_time: time for prediction
    
    Return:
    --------------------------------------------
    ll: lon-lat position at current_time
    """
    segment_length = np.asarray([nodes_to_link_attributes['{}_{}'.format(node_path[i], node_path[i+1])][
        'distance'] for i in range(len(node_path)-1)])
    segment_length_ratio = segment_length.cumsum() / segment_length.sum()
    time_ratio = (current_time-period[0]) / (period[1]-period[0])
    try:
        segment_idx = next(idx for idx, ratio in enumerate(segment_length_ratio) if ratio>=time_ratio)
    except:
        segment_idx = 0
        print('"get_realtime_route_position" encount an error: ')
        print('segment_length_ratio: ', segment_length_ratio)
        print('time_ratio: ', time_ratio)
    if segment_length_ratio[segment_idx] > segment_length_ratio[segment_idx-1]:
        in_segment_ratio = (time_ratio-segment_length_ratio[segment_idx-1]) / (segment_length_ratio[segment_idx]-segment_length_ratio[segment_idx-1])
    else:
        in_segment_ratio = 0
    segment_start_ll = LatLongDict[node_path[segment_idx]]
    segment_end_ll = LatLongDict[node_path[segment_idx+1]]
    ll = [segment_start_ll[i] + (segment_end_ll[i]-segment_start_ll[i]) * in_segment_ratio for i in [0,1]]
    return ll


def get_realtime_agents(persons, current_time):
    """ 
    takes a list of person objects, and generate a list of agent objects at certain time
    an agent object is dict with minimum realtime information: "status", "activity", "mode", "ll"
    """
    agents = []
    generic_keys = ['status', 'activity', 'mode']
    for person in persons:
        for sched in person['sched_objs']:
            if current_time>=sched['period'][0] and current_time<=sched['period'][1]:
                break
        agent = {key: sched.get(key, None) for key in generic_keys}
        if sched['status'] == 'stay':
            agent['ll'] = sched['place_sim']['ll']
        elif sched['status'] == 'trip':
            if len(sched['node_path']) > 1:
                agent['ll'] = get_realtime_route_position(
                    sched['node_path'], sched['period'], current_time)
            else:
                # node_path may be empty when o and d are too close
                agent['ll'] = sched['d_loc']['ll']
        agents.append(agent)
    return agents
    
def draw_agents_inital(agents, ax, time_sec=None):
    """ 
    takes a list of agent objects, initializes the visulization: 
    road network as background, and matplotlib objects for different kinds of agents.
    return these matplotlib objects for update
    """
    # background
    from_coords, to_coords = [], []
    for key, value in nodes_to_link_attributes.items():
        from_coords.append(value['from_coord'])
        to_coords.append(value['to_coord'])
    from_coords, to_coords = np.asarray(from_coords), np.asarray(to_coords)
    x_coords = np.asarray([[from_coord[0], to_coord[0]] for from_coord, to_coord in zip(from_coords, to_coords)]).transpose()
    y_coords = np.asarray([[from_coord[1], to_coord[1]] for from_coord, to_coord in zip(from_coords, to_coords)]).transpose()
    ax.plot(x_coords, y_coords, 'b-', linewidth=1) 

    driving_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==0] 
    cycling_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==1]
    walk_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==2]
    pt_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==3]
    stay_agents = [agent['ll'] for agent in agents if agent['status']=='stay']
    driving_agents = np.asarray(driving_agents)
    cycling_agents = np.asarray(cycling_agents)
    walk_agents = np.asarray(walk_agents)
    pt_agents = np.asarray(pt_agents)
    stay_agents = np.asarray(stay_agents)
    
    if len(driving_agents) > 0:
        driving_points, = ax.plot(driving_agents[:,0], driving_agents[:,1], 'ro', label='driving') 
    else:
        driving_points, = ax.plot([], [], 'ro', label='driving')
    if len(cycling_agents) > 0:
        cycling_points, = ax.plot(cycling_agents[:,0], cycling_agents[:,1], 'yo', label='cycling')
    else:
        cycling_points, = ax.plot([], [], 'yo', label='cycling')
    if len(walk_agents) > 0:
        walk_points, = ax.plot(walk_agents[:,0], walk_agents[:,1], 'co', label='walk')
    else:
        walk_points, = ax.plot([], [], 'co', label='walk')
    if len(pt_agents) > 0:
        pt_points, = ax.plot(pt_agents[:,0], pt_agents[:,1], 'mo', label='pt')
    else:
        pt_points, = ax.plot([], [], 'mo', label='pt')
    if len(stay_agents) > 0:
        stay_points, = ax.plot(stay_agents[:,0], stay_agents[:,1], 'ks', label='stay')
    else:
        stay_points, = ax.plot([], [], 'ks', label='stay')
    ax.legend()
    if time_sec:
        hour = int(time_sec / 3600)
        minute = int((time_sec-hour*3600)/60)
        second = time_sec - 3600*hour - 60*minute
        time_str = ':'.join([str(hour).zfill(2), str(minute).zfill(2), str(second).zfill(2)])
        ax.set_title('Current time: {}'.format(time_str))
    points = {'driving_points': driving_points, 'cycling_points': cycling_points, 'walk_points': walk_points, 'pt_points': pt_points, 'stay_points': stay_points}
    print('People counts at {}: driving={}, cycling={}, walk={}, pt={}, stay={}'.format(time_str, 
        len(driving_agents), len(cycling_agents), len(walk_agents), len(pt_agents),  len(stay_agents)))
    return points
    
def draw_agents_update(agents, points, time_sec=None):
    """ 
    takes a list of agent objects and a dict of matplotlib objects for different kinds of agents,
    update their locations and status
    """
    driving_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==0] 
    cycling_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==1]
    walk_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==2]
    pt_agents = [agent['ll'] for agent in agents if agent['status']=='trip' and agent['mode']==3]
    stay_agents = [agent['ll'] for agent in agents if agent['status']=='stay']
    driving_agents = np.asarray(driving_agents)
    cycling_agents = np.asarray(cycling_agents)
    walk_agents = np.asarray(walk_agents)
    pt_agents = np.asarray(pt_agents)
    stay_agents = np.asarray(stay_agents)
    driving_points = points['driving_points']
    cycling_points = points['cycling_points']
    walk_points = points['walk_points']
    pt_points = points['pt_points']
    stay_points = points['stay_points']
    if len(driving_agents) > 0:
        driving_points.set_data(driving_agents[:,0], driving_agents[:,1])
    else:
        driving_points.set_data([], [])
    if len(cycling_agents) > 0:
        cycling_points.set_data(cycling_agents[:,0], cycling_agents[:,1])
    else:
        cycling_points.set_data([], [])
    if len(walk_agents) > 0:
        walk_points.set_data(walk_agents[:,0], walk_agents[:,1])
    else:
        walk_points.set_data([], [])
    if len(pt_agents) > 0:
        pt_points.set_data(pt_agents[:,0], pt_agents[:,1])
    else:
        pt_points.set_data([], [])
    if len(stay_agents) > 0:
        stay_points.set_data(stay_agents[:,0], stay_agents[:,1])
    else:
        stay_points.set_data([], [])
    if time_sec:
        hour = int(time_sec / 3600)
        minute = int((time_sec-hour*3600)/60)
        second = time_sec - 3600*hour - 60*minute
        time_str = ':'.join([str(hour).zfill(2), str(minute).zfill(2), str(second).zfill(2)])
        ax.set_title('Current time: {}'.format(time_str))
    else:
        ax.set_title('')
    print('People counts at {}: driving={}, cycling={}, walk={}, pt={}, stay={}'.format(time_str, 
        len(driving_agents), len(cycling_agents), len(walk_agents), len(pt_agents),  len(stay_agents)))



# =============================================================================
# Constants
# =============================================================================

ALL_ZONES_PATH='./scripts/cities/'+city+'/clean/model_area.geojson'
SIM_ZONES_PATH='./scripts/cities/'+city+'/clean/sim_zones.json'
# Synthpop results
SIM_POP_PATH='./scripts/cities/'+city+'/clean/sim_pop.json'
VACANT_PATH='./scripts/cities/'+city+'/clean/vacant.json'
FLOATING_PATH='./scripts/cities/'+city+'/clean/floating.json'
# Mode choice model
FITTED_MODE_MODEL_PATH='./scripts/cities/'+city+'/models/trip_mode_rf.p'
RF_FEATURES_LIST_PATH='./scripts/cities/'+city+'/models/rf_features.json'
# Home location choice model
FITTED_HOME_LOC_MODEL_PATH='./scripts/cities/'+city+'/models/home_loc_logit.p'
RENT_NORM_PATH='./scripts/cities/'+city+'/models/rent_norm.json'

#ACTIVITY_SCHED_PATH='./scripts/cities/'+city+'/clean/person_sched_weekday.csv'
MOTIF_SAMPLE_PATH='./scripts/cities/'+city+'/clean/motif_samples.csv'

#Road network graph
PORTALS_PATH='./scripts/cities/'+city+'/clean/portals.geojson'
ROUTE_COSTS_PATH='./scripts/cities/'+city+'/clean/route_costs.json'
FLOYD_PREDECESSOR_PATH='./scripts/cities/'+city+'/clean/fw_result.json'
INT_NET_DF_FLOYD_PATH='./scripts/cities/'+city+'/clean/sim_net_df_floyd.csv'
INT_NET_COORDINATES_PATH='./scripts/cities/'+city+'/clean/sim_net_node_coords.json'

GEOGRID_SAMPLE_PATH='./scripts/cities/'+city+'/clean/geogrid.geojson'

PUMA_SHAPE_PATH='./scripts/cities/'+city+'/raw/PUMS/pumas.geojson'
PUMAS_INCLUDED_PATH='./scripts/cities/'+city+'/raw/PUMS/pumas_included.json'
PUMA_ATTR_PATH = './scripts/cities/'+city+'/models/puma_attr.json'
EXTERNAL_LU_PATH = './scripts/cities/'+city+'/clean/external_lu.json'

MAPPINGS_PATH = './scripts/cities/'+city+'/mappings'


# the graph used by each mode
mode_graphs={0:'driving',
             1:'cycling',
             2:'walking',
             3:'pt'}

SPEEDS_MET_S={0:30/3.6,
        1:15/3.6,
        2:4.8/3.6,
        3: 20/3.6 
        }

kgCO2PerMet={0: 0.45*0.8708/0.00162,
                    1: 0,
                    2: 0,
                    3: 0.45*0.2359/0.00162}#from lbs/mile to US tonnes/m
# TODO: put the below in a text file
# TODO: number of of people per housing cell should vary by type
housing_lus={'Residential_Affordable':{'rent': 800, 'beds': 2, 'built_since_jan2010': True, 
                  'puma_pop_per_sqmeter': 0.000292, 'puma_med_income': 60000,
                  'pop_per_sqmile': 5000, 'tenure': 'rented'},
             'Residential_Market_Rate':{'rent': 1500, 'beds': 2, 'built_since_jan2010': True, 
                  'puma_pop_per_sqmeter': 0.000292, 'puma_med_income': 60000,
                  'pop_per_sqmile': 5000, 'tenure': 'rented'}}
# TODO: number of each employment sector for each building type
employment_lus= {"Office_High_Density":{},"Office_Low_Density":{},
                 "Industrial":{},"Makerspace":{}}

#land_use_codes={'home': ['R'], 'work': ['M', 'B']}

NEW_PERSONS_PER_BLD=1

# #cityIO grid data
table_name_map={'Boston':"mocho",
     'Hamburg':"grasbrook",
     'Detroit': "corktown"}
host='https://cityio.media.mit.edu/'
CITYIO_GET_URL=host+'api/table/'+table_name_map[city]
UPDATE_FREQ=1 # seconds

#CITYIO_SAMPLE_PATH='scripts/cities/'+city+'/clean/sample_cityio_data.json' #cityIO backup data

# destination for output files
CITYIO_POST_URL=host+'api/table/update/'+table_name_map[city]+'/'

# activity full name lookup
# "S" is short for "Buy services" before and "Shopping" now
activity_full_name = {'H': 'Home', 'W': 'Work', 'C': 'College', 'D': 'Drop-off', 'G': 'Groceries',
    'S': 'Shopping', 'E': 'Eat', 'R': 'Recreation', 'X': 'Exercise', 'V': 'Visit', 'P': 'Health', 'Z':'Religion'}

# =============================================================================
# Load Data
# =============================================================================
# load the pre-calibrated mode choice model
mode_rf=pickle.load( open( FITTED_MODE_MODEL_PATH, "rb" ) )
rf_features=json.load(open(RF_FEATURES_LIST_PATH, 'r'))
# load the pre-calibrated home location choice model
home_loc_logit=pickle.load( open( FITTED_HOME_LOC_MODEL_PATH, "rb" ) )
rent_normalisation=json.load(open(RENT_NORM_PATH))

# load the external route costs
ext_route_costs=json.load(open(ROUTE_COSTS_PATH))

sample_motifs=pd.read_csv(MOTIF_SAMPLE_PATH)
motif_sample_obj=sample_motifs.to_dict(orient='records')

# load the land use and activity mappings

lu_inputs=json.load(open(MAPPINGS_PATH+'/lu_inputs.json'))
lu_input_to_lu_standard=json.load(open(MAPPINGS_PATH+'/lu_input_to_lu_standard.json'))
activities_to_lu=json.load(open(MAPPINGS_PATH+'/activities_to_lu.json'))
base_lu_to_lu=json.load(open(MAPPINGS_PATH+'/base_lu_to_lu.json'))
#lu_standard=json.load(open(MAPPINGS_PATH+'/lu_standard.json'))
external_lu = json.load(open(EXTERNAL_LU_PATH))

all_zones=json.load(open(ALL_ZONES_PATH))
sim_zones=json.load(open(SIM_ZONES_PATH))
portals=json.load(open(PORTALS_PATH))

# precomputed shortest paths for internal simulation networks
sim_net_floyd_result=json.load(open(FLOYD_PREDECESSOR_PATH))
sim_net_floyd_df=pd.read_csv(INT_NET_DF_FLOYD_PATH)


# =============================================================================
# Pre-Processing
# =============================================================================

# Processing of the Floyd Warshall results and graph
# Create mapping from nodes to link attributes to speed up queries
nodes_to_link_attributes={}
LatLongDict = {}
for ind, row in sim_net_floyd_df.iterrows():
    nodes_to_link_attributes['{}_{}'.format(row['aNodes'], row['bNodes'])]={
        'distance': row['distance'],
        'from_coord': [float(row['aNodeLon']), float(row['aNodeLat'])],
        'to_coord': [float(row['bNodeLon']), float(row['bNodeLat'])]}
    if row['aNodes'] not in LatLongDict:
       LatLongDict[str(row['aNodes'])]  = [float(row['aNodeLon']), float(row['aNodeLat'])]
    if row['bNodes'] not in LatLongDict:
       LatLongDict[str(row['bNodes'])]  = [float(row['bNodeLon']), float(row['bNodeLat'])]


sim_net_map_node_lls=json.load(open(INT_NET_COORDINATES_PATH))
sim_node_ids=[node for node in sim_net_map_node_lls]
sim_node_lls=[sim_net_map_node_lls[node] for node in sim_node_ids]
internal_nodes_kdtree=spatial.KDTree(np.array(sim_node_lls))


# add centroids and closest sim network nodes to portals
for p in portals['features']:
    centroid=approx_shape_centroid(p['geometry'])
    p['properties']['centroid']=centroid
    p['properties']['closest_nodes']=[sim_node_ids[n_ind] for n_ind in 
      internal_nodes_kdtree.query(centroid, 3)[1]]
      

if city=='Hamburg':
    geoid_order_all=[f['properties']['GEO_ID'] for f in all_zones['features']]
    sim_area_zone_list=sim_zones
else:
    geoid_order_all=[f['properties']['GEO_ID'].split('US')[1] for f in all_zones['features']]
    sim_area_zone_list=[z.split('US')[1] for z in sim_zones]

all_geoid_centroids={}
for ind, geo_id in enumerate(geoid_order_all):
    centroid=approx_shape_centroid(all_zones['features'][ind]['geometry'])
#    all_geoid_centroids[geo_id]=[centroid.x, centroid.y]
    all_geoid_centroids[geo_id]=list(centroid)

# =============================================================================
# Pre-Processing of spatial grid data
# =============================================================================
    
# Full geogrid geojson      
try:
    with urllib.request.urlopen(CITYIO_GET_URL+'/GEOGRID') as url:
    #get the latest grid data
        geogrid=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    geogrid=json.load(open(GEOGRID_SAMPLE_PATH))
    

# and a dict of static land uses to their locations in the geogrid
static_land_uses={}
for fi, f in enumerate(geogrid['features']):
    this_land_use_input=f['properties']['land_use']
    this_land_use_standard=get_standard_lu_from_base(this_land_use_input)
    if this_land_use_standard in static_land_uses:
        static_land_uses[this_land_use_standard].append(fi)
    else:
        static_land_uses[this_land_use_standard]=[fi]

    
# add centroids and closest nodes in sim network to geogrid_cells
for cell in geogrid['features']:
    centroid=approx_shape_centroid(cell['geometry'])
    cell['properties']['centroid']=centroid
    cell['properties']['closest_nodes']=[sim_node_ids[n_ind] for n_ind in 
      internal_nodes_kdtree.query(centroid, 3)[1]]

# create a lookup from interactive grid to puma
puma_shape=json.load(open(PUMA_SHAPE_PATH))
puma_order=[f['properties']['PUMACE10'] for f in puma_shape['features']]
puma_included=json.load(open(PUMAS_INCLUDED_PATH)) 
puma_path_dict = {}
# if the shape type is "Polygon", [0][0] would return only a point
for feature in puma_shape['features']:
    if feature["properties"]["GEOID10"][2:] in puma_included:
        if feature['geometry']['type'] == 'Polygon':
            puma_path_dict[feature["properties"]["GEOID10"][2:]] = mplPath.Path(feature["geometry"]["coordinates"][0])
        elif feature['geometry']['type'] == 'MultiPolygon':
            puma_path_dict[feature["properties"]["GEOID10"][2:]] = mplPath.Path(feature["geometry"]["coordinates"][0][0])
geogrid_to_puma = {'grid'+str(grid_id): None for grid_id in range(len(geogrid['features']))}
for grid_id, grid_feat in enumerate(geogrid['features']):
    for puma_id, puma_path in puma_path_dict.items():
        if puma_path.contains_point((grid_feat['properties']['centroid'])):
            geogrid_to_puma['grid'+str(grid_id)] = puma_id
            break
            
# create puma objects
puma_df = pd.DataFrame(json.load(open(PUMA_ATTR_PATH, 'r')))
puma_obj_dict = {}
for puma in puma_df.index:
    this_obj = puma_df.loc[puma].to_dict()
    this_obj['puma'] = puma
    centroid = approx_shape_centroid(puma_shape['features'][puma_order.index(puma)]['geometry'])
    this_obj['centroid'] = centroid
    puma_obj_dict[puma] = this_obj

sim_area_zone_list+=['grid'+str(i) for i in range(len(geogrid['features']))]
                   
# =============================================================================
# Population
# =============================================================================
# TODO: option to sample from base pop
# load sim_persons
base_sim_persons=json.load(open(SIM_POP_PATH))
# TODO: checl how the id is used
for idx, person in enumerate(base_sim_persons):
    person['person_id'] = 'b'+str(idx)
# load floaters
base_floating_persons=json.load(open(FLOATING_PATH))
# load vacant houses
base_vacant_houses=json.load(open(VACANT_PATH))
for h in base_vacant_houses:
    h['centroid']=all_geoid_centroids[h['home_geoid']]


if base_sim_persons: 
    get_simulation_locations(base_sim_persons)
    get_LLs(base_sim_persons, ['home', 'work'])
    get_person_type(base_sim_persons)
    sample_activity_schedules(base_sim_persons)
    # 'Residential' in 'activities_to_lu', but not in lu_standard
    static_land_uses_tmp = copy.deepcopy(static_land_uses)
    static_land_uses_tmp['Residential'] = static_land_uses_tmp.get('Residential_Affordable', []) + static_land_uses_tmp.get('Residential_Market_Rate', [])
    find_destination(base_sim_persons, land_uses=static_land_uses_tmp)
    ods = generate_ods(base_sim_persons) # TODO this function takes longest
    predict_modes_for_activities(ods,base_sim_persons)
    trips=create_trips_layer(base_sim_persons)
    # stays=create_stay_data(base_sim_persons)
    post_trips_data(trips, CITYIO_POST_URL+'ABM')
#    generate_detailed_schedules(base_sim_persons)
#    post_od_data(base_sim_persons, CITYIO_POST_URL+'od')

if base_floating_persons:
    get_LLs(base_floating_persons, ['work'])
#    create_trips(base_sim_persons)
#    post_trips_data(base_sim_persons, CITYIO_POST_URL+'trips')

# =============================================================================
# Handle Interactions
# =============================================================================

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.axis('off')
#ax.axis('equal')
#plt.ion()

lastId=0
while True:
#check if grid data changed
    try:
        with urllib.request.urlopen(CITYIO_GET_URL+'/meta/hashes/GEOGRIDDATA') as url:
            hash_id=json.loads(url.read().decode())
    except:
        print('Cant access cityIO')
        hash_id=1
    if hash_id==lastId:
        sleep(1)
    else:
        print('Change in grid detected. Fetching the new grid data')
        try:
            with urllib.request.urlopen(CITYIO_GET_URL+'/GEOGRIDDATA') as url:
                cityIO_grid_data=json.loads(url.read().decode())
        except:
            print('Read hash coudnt read grid data')
        start_time=time.time()
        lastId=hash_id
        # TODO: use actual inputs below
## =============================================================================
##         FAKE DATA FOR SCENAIO EXPLORATION
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(3,5,len(geogrid['features']))] # all employment
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(1,3,len(geogrid['features']))] # all housing
        cityIO_grid_data=[random.randint(0, 6) for i in range(len(geogrid['features']))] # random mix
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(2,4,len(geogrid['features']))] # affordable + employment
## =============================================================================
        new_houses=[]
        new_persons=[]
#        new_households=[]  
        # tui_grid_land_uses to geogrid_land_uses
        geogrid_land_uses=[get_standard_lu_from_input_lu(g) for g in cityIO_grid_data]
        # TODO: also web_grid_land_uses
        # adding new land use information for interactive grids to static_land_uses
        overall_land_uses = copy.deepcopy(static_land_uses)
        for geogrid_idx, geogrid_lu in enumerate(geogrid_land_uses):
            # TODO: no lookup- use geogrid_land_uses directly
            overall_land_uses.setdefault(geogrid_lu, []).append(geogrid_idx)
            # append the new locations for each land uses, but if the lu does not yet exist in the dict,
            # add it with a value of []
        overall_land_uses_tmp = copy.deepcopy(overall_land_uses)
        overall_land_uses_tmp['Residential'] = overall_land_uses_tmp.get('Residential_Affordable', []) + overall_land_uses_tmp.get('Residential_Market_Rate', [])
        for ht in housing_lus:
            ht_locs=[i for i in range(len(geogrid_land_uses)) if geogrid_land_uses[i]==ht]
            for htl in ht_locs:
                add_house=housing_lus[ht].copy()
                add_house['home_geoid']='grid'+str(htl)
                add_house['centroid']=geogrid['features'][htl]['properties']['centroid']
                new_houses.append(add_house)        
#        # for each office unit, add a (homeless) person
        random.seed(0)
        for et in employment_lus:
            et_locs=[i for i in range(len(geogrid_land_uses)) if geogrid_land_uses[i]==et]
            for etl in et_locs:
                for i in range(NEW_PERSONS_PER_BLD):
                    add_person=random.choice(base_floating_persons).copy()
                    add_person['work_geoid']='grid'+str(etl)
                    new_persons.append(add_person)
        get_LLs(new_persons, ['work'])
        floating_persons=base_floating_persons+new_persons
        vacant_houses=base_vacant_houses+new_houses
        for ip, p in enumerate(floating_persons):
            p['person_id']=ip
        for ih, h in enumerate(vacant_houses):
            h['house_id']=ih
        # add PUMA info for new housing units
        for h in vacant_houses:
            if 'puma10' not in h:
                h['puma10'] = geogrid_to_puma[h['home_geoid']]
            h['puma10'] = str(h['puma10']).zfill(5)
        top_n_pumas = 5
        home_location_choices(vacant_houses, floating_persons)
        # new_sim_people = people living/working in simzone
        new_sim_persons=[p for p in floating_persons if
                         (p['home_geoid'] in sim_area_zone_list or
                          p['work_geoid'] in sim_area_zone_list)]
        get_LLs(new_sim_persons, ['home', 'work'])
        get_simulation_locations(new_sim_persons)
        get_person_type(new_sim_persons)
        sample_activity_schedules(new_sim_persons)     
        find_destination(new_sim_persons, land_uses=overall_land_uses_tmp)
        new_ods = generate_ods(new_sim_persons)
        predict_modes_for_activities(new_ods, new_sim_persons)
        trips=create_trips_layer(base_sim_persons+new_sim_persons)
        # stays=create_stay_data(base_sim_persons)
        post_trips_data(trips, CITYIO_POST_URL+'ABM')
#        generate_detailed_schedules(new_sim_persons)       
#        post_od_data(base_sim_persons+ new_sim_persons, CITYIO_POST_URL+'od')
        # post_sched_data(base_sim_persons+ new_sim_persons, CITYIO_POST_URL+'sched')    #always return 413
        finish_time=time.time()
        print('Response time: '+ str(finish_time-start_time))
        
        # visualization
#        current_time_sec = 5
#        agents = get_realtime_agents(base_sim_persons+ new_sim_persons, current_time_sec)
#        points = draw_agents_inital(agents, ax, current_time_sec)
#        plt.pause(0.2)
#        for current_time_sec in [hour*3600+5 for hour in range(1,24)]:
#            agents = get_realtime_agents(base_sim_persons+ new_sim_persons, current_time_sec)
#            draw_agents_update(agents, points, current_time_sec)
#            plt.pause(0.2)
            
        sleep(0.2)
