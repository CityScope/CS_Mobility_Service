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

city=sys.argv[1]
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
    For each agent, based on their actual home and work zones (geoids or int grid cells)
    a "simulation" home and a workplace are assigned which may be metagrid cells or portals
    """
    for p in persons:  
        for place in ['home', 'work']:
            geoid=p[place+'_geoid']
            if 'g' in str(geoid):
                p[place+'_sim']={'type': 'meta_grid', 
                                 'ind': int_to_meta_grid[int(geoid[1:])]}
            elif p[place+'_geoid'] in sim_area_zone_list:
                if 'place'=='work':
                    relevant_land_use_codes=employment_lus
                else:
                    relevant_land_use_codes=housing_lus
                possible_locations=[static_land_uses[rlu] for rlu in relevant_land_use_codes if rlu in static_land_uses]
                possible_locations=[item for sublist in possible_locations for item in sublist]
                p[place+'_sim']={'type': 'meta_grid', 
                                 'ind': random.choice(possible_locations)}
            else:
                p[place+'_sim']={'type': 'portal'}


def get_LLs(persons, places):
    """ takes a list of person objects and 
    finds home and work coordinates for them
    modifies in place
    """
    for p in persons:  
        for place in places:
            geoid=p[place+'_geoid']
            if 'g' in str(geoid):
                ll=grid_points_ll[int(geoid[1:])]
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
    
        
def get_route_costs(persons):
    """ takes a list of person objects 
    and finds the travel time costs of travel by each mode
    modifies in place
    """
    for p in persons:
        p['routes']={}
        if ((p['home_sim']['type']=='meta_grid') and  (p['work_sim']['type']=='meta_grid')):
            p['type']=0 # lives and works on site
#            home_coord=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
#            work_coord=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
            home_node_list=meta_grid['features'][p['home_sim']['ind']]['properties']['closest_nodes']
            work_node_list=meta_grid['features'][p['work_sim']['ind']]['properties']['closest_nodes']

#            p['routes']=approx_route_costs(home_coord, work_coord)
            p['routes']=internal_route_costs(home_node_list, work_node_list, 
                         sim_net_floyd_result, nodes_to_link_attributes)
        elif p['work_sim']['type']=='meta_grid':
            p['type']=1 # commute_in
#            work_coord=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
            work_node_list=meta_grid['features'][p['work_sim']['ind']]['properties']['closest_nodes']
            for m in range(4):
                p['routes'][m]={}
                best_portal_route_time=float('inf')
                for portal in range(len(portals['features'])):
#                    portal_coord=portals['features'][portal]['properties']['centroid']
                    portal_node_list=portals['features'][portal]['properties']['closest_nodes']
#                    internal_portal_route=approx_route_costs(portal_coord, work_coord)[m]
                    internal_portal_route=internal_route_costs(portal_node_list, work_node_list, 
                         sim_net_floyd_result, nodes_to_link_attributes)[m]
                    external_portal_route=ext_route_costs[mode_graphs[m]][str(p['home_geoid'])][str(portal)]
                    external_time=sum([external_portal_route[c] for c in external_portal_route])
                    full_portal_route={c: internal_portal_route['route'][c] + external_portal_route[c] for
                                       c in ['driving', 'walking', 'waiting','cycling', 'pt']}
                    total_portal_route_time=sum([full_portal_route[c] for c in full_portal_route])
                    if total_portal_route_time<best_portal_route_time:
                        best_portal=portal
                        best_route=full_portal_route
                        best_external_time=external_time
                        best_portal_route_time=total_portal_route_time                    
                p['routes'][m]['portal']=best_portal
                p['routes'][m]['external_time']= int(best_external_time*60)
                p['routes'][m]['route']=best_route
        elif p['home_sim']['type']=='meta_grid':
            p['type']=2 # commute_out
#            home_coord=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
            home_node_list=meta_grid['features'][p['home_sim']['ind']]['properties']['closest_nodes']

            for m in range(4):
                p['routes'][m]={}
                best_portal_route_time=float('inf')
                for portal in range(len(portals['features'])):
#                    portal_coord=portals['features'][portal]['properties']['centroid']
                    portal_node_list=portals['features'][portal]['properties']['closest_nodes']
#                    internal_portal_route=approx_route_costs(home_coord, portal_coord)[m]
                    internal_portal_route=internal_route_costs(home_node_list, portal_node_list, 
                         sim_net_floyd_result, nodes_to_link_attributes)[m]
                    external_portal_route=ext_route_costs[mode_graphs[m]][str(p['work_geoid'])][str(portal)]
                    external_time=sum([external_portal_route[c] for c in external_portal_route])
                    full_portal_route={c: internal_portal_route['route'][c] + external_portal_route[c] for
                                       c in ['driving', 'walking', 'waiting','cycling', 'pt']}
                    total_portal_route_time=sum([full_portal_route[c] for c in full_portal_route])
                    if total_portal_route_time<best_portal_route_time:
                        best_portal=portal
                        best_route=full_portal_route
                        best_external_time=external_time
                        best_portal_route_time=total_portal_route_time                    
                p['routes'][m]['portal']=best_portal
                p['routes'][m]['external_time']= 0
                p['routes'][m]['route']=best_route
                
def predict_modes(persons):
    """ takes list of person objects and 
    predicts transport modes for each person's commute
    modifies in place
    """
    feature_df=pd.DataFrame(persons)  
#    feature_df['bach_degree_yes']=feature_df['SCHL']>20
#    feature_df['bach_degree_no']=~feature_df['bach_degree_yes']
    for feat in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 
                 'bach_degree', 'race', 'cars']:
        new_dummys=pd.get_dummies(feature_df[feat], prefix=feat)
        feature_df=pd.concat([feature_df, new_dummys],  axis=1)
    # TODO: better method of predicting travel times
    # routing engine or feedback from simulation
    feature_df['drive_time_minutes']=  feature_df.apply(lambda row: row['routes'][0]['route']['driving'], axis=1)     
    feature_df['cycle_time_minutes']=  feature_df.apply(lambda row: row['routes'][1]['route']['cycling'], axis=1)     
    feature_df['walk_time_minutes']=  feature_df.apply(lambda row: row['routes'][2]['route']['walking'], axis=1)     
    feature_df['PT_time_minutes']=  feature_df.apply(lambda row: row['routes'][3]['route']['pt'], axis=1)
    feature_df['walk_time_PT_minutes']=feature_df.apply(lambda row: row['routes'][3]['route']['walking'], axis=1)  
    feature_df['drive_time_PT_minutes']=0 
    # TODO: below should come directly from the path-finding
    feature_df['network_dist_km']=feature_df.apply(lambda row: row['drive_time_minutes']*30/60, axis=1) 
    # TODO: change below if modelling housing sales as well
    feature_df['tenure_owned']=False
    feature_df['tenure_other']=False
    feature_df['purpose_HBW']=1
    feature_df['purpose_NHB']=0
    feature_df['purpose_HBO']=0
    feature_df['race_asian']=0
    for rff in rf_features:
        # feature_df[']
        assert rff in feature_df.columns, str(rff) +' not in data.'
#    assert all([rff in feature_df.columns for rff in rf_features]
#    ),"Features in table dont match features in RF model"   
    feature_df=feature_df[rf_features]#reorder columns to match rf model
    mode_probs=mode_rf.predict_proba(feature_df)
    for i,p in enumerate(persons): 
        chosen_mode=int(np.random.choice(range(4), size=1, replace=False, p=mode_probs[i])[0])
        p['mode']=chosen_mode
        if p['home_sim']['type']=='portal': 
            p['home_sim']['ind']=p['routes'][chosen_mode]['portal']
            p['home_sim']['ll']=portals['features'][p['home_sim']['ind']]['properties']['centroid']
            p['work_sim']['ll']=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
        elif p['work_sim']['type']=='portal': 
            p['work_sim']['ind']=p['routes'][chosen_mode]['portal']
            p['work_sim']['ll']=portals['features'][p['work_sim']['ind']]['properties']['centroid']
            p['home_sim']['ll']=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
        else:
            p['home_sim']['ll']=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
            p['work_sim']['ll']=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
        p['external_time']=p['routes'][chosen_mode]['external_time']
        
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
    
def post_od_data(persons, destination_address):
    od_str=json.dumps([{'home_ll': p['home_sim']['ll'],
                       'work_ll': p['work_sim']['ll'],
                       'home_sim': p['home_sim'],
                       'work_sim': p['work_sim'],
                       'type': p['type'],
                       'mode': p['mode'],
                       'activities': p['activities'],
                       'motif': p['motif_name'],
#                       'activity_start_times':p['start_times'],
                       'start_time': 6*3600+random.randint(0,3*3600)+p['external_time']
                       } for p in persons if len(p['activities'])>1])
    try:
        r = requests.post(destination_address, data = od_str)
        print('OD: {}'.format(r))
    except requests.exceptions.RequestException as e:
        print('Couldnt send to cityio')
    
              
def create_long_record_puma(person, puma):
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
    # preparing
    valid_pumas = list(set([h['puma10'] for h in houses]))
    valid_puma_objs = [puma_obj_dict[puma] for puma in valid_pumas]
    puma_to_houses = {puma: [h for h in houses if h['puma10']==puma] for puma in valid_pumas}
    for h in houses:        # updating number of houses in each puma given new houses
        if h['home_geoid'].startswith('g'):
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

def shannon_equitability(species_pop, species_set):
    diversity=0
    pop_size=len(species_pop)
    if pop_size>0:
        for species in species_set:
            pj=species_pop.count(species)/len(species_pop)
            if not pj==0:
                diversity+= -pj*np.log(pj)
        equitability=diversity/np.log(len(species_set))
        return equitability
    else:
        return 0
            
def get_pop_diversity(persons):
    diversity={}
    dims=['age', 'income']
    for d in dims:
        dim_all_persons=[p[d] for p in persons]
        diversity[d]=shannon_equitability(dim_all_persons, set(dim_all_persons))
    return diversity
        
def get_lu_diversity(grid_data):
    # TODO: incorporate num floors
    lu_diversity={}
    housing_pop=[lu for lu in grid_data if lu in housing_lus]
    lu_diversity['housing']=shannon_equitability(housing_pop, set(housing_lus))
    office_pop=[lu for lu in grid_data if lu in employment_lus]
    lu_diversity['office']=shannon_equitability(office_pop, set(employment_lus))
    return lu_diversity

def post_diversity_indicators(pop_diversity, lu_diversity, destination_address):
    all_diversity={}
    for var in pop_diversity:
        all_diversity[var]=pop_diversity[var]
    for var in lu_diversity:
        all_diversity[var]=lu_diversity[var]
    try:
        r = requests.post(destination_address, data = json.dumps(all_diversity))
        print('Diversity Indicators: {}'.format(r))
    except requests.exceptions.RequestException as e:
        print('Couldnt send diversity indicators to cityio')
        
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
    print('No path found')
    return []     
  
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
            coords+=[link_attributes['from_coord']]
        coords+= [link_attributes['to_coord']]
        # add the final coordinate of the very last segment
    return coords, distances

def internal_route_costs(from_node_list, to_node_list, 
                         sim_net_floyd_result, nodes_to_link_attributes):
    approx_speeds_met_s={'driving':30/3.6,
        'cycling':15/3.6,
        'walking':4/3.6,
        'pt': 20/3.6 
        }
#     from_node_list= [sim_node_ids[n_ind] for n_ind in int_nodes_kdtree.query(start_coord, 3)[1]]
#     to_node_list= [sim_node_ids[n_ind] for n_ind in int_nodes_kdtree.query(end_coord, 3)[1]]
    path=get_node_path_from_fw_try_multi(sim_net_floyd_result, from_node_list, to_node_list)
    coords, distances=get_path_coords_distances(nodes_to_link_attributes, path)
    total_distance=sum(distances)
    routes={}
    for mode in range(4):
        routes[mode]={'route': {'driving':0, 'walking':0, 'waiting':0,
                      'cycling':0, 'pt':0}, 'external_time':0}
    routes[0]['route']['driving']=(total_distance/approx_speeds_met_s['driving'])/60
    routes[1]['route']['cycling']=(total_distance/approx_speeds_met_s['cycling'])/60
    routes[2]['route']['walking']=(total_distance/approx_speeds_met_s['walking'])/60
    routes[3]['route']['pt']=(total_distance/approx_speeds_met_s['pt'])/60
    routes[3]['route']['walking']=(200/approx_speeds_met_s['walking'])/60
    return routes

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

META_GRID_SAMPLE_PATH='./scripts/cities/'+city+'/clean/meta_grid.geojson'
GRID_INT_SAMPLE_PATH='./scripts/cities/'+city+'/clean/grid_interactive.geojson'

PUMA_SHAPE_PATH='./scripts/cities/'+city+'/raw/PUMS/pumas.geojson'
PUMAS_INCLUDED_PATH='./scripts/cities/'+city+'/raw/PUMS/pumas_included.json'
PUMA_ATTR_PATH = './scripts/cities/'+city+'/models/puma_attr.json'

MAPPINGS_PATH = './scripts/cities/'+city+'/mappings'


# the graph used by each mode
mode_graphs={0:'driving',
             1:'cycling',
             2:'walking',
             3:'pt'}

SPEEDS_MET_S={'driving':30/3.6,
        'cycling':15/3.6,
        'walking':4.8/3.6,
        'pt': 4.8/3.6 # only used for grid use walking speed for pt
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

NEW_PERSONS_PER_BLD=10

# #cityIO grid data
table_name_map={'Boston':"mocho",
     'Hamburg':"grasbrook",
     'Detroit': "corktown"}
host='https://cityio.media.mit.edu/'
cityIO_grid_url=host+'api/table/'+table_name_map[city]
UPDATE_FREQ=1 # seconds
CITYIO_SAMPLE_PATH='scripts/cities/'+city+'/clean/sample_cityio_data.json' #cityIO backup data

# destination for output files
CITYIO_OUTPUT_PATH=host+'api/table/update/'+table_name_map[city]+'/'

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
for ind, row in sim_net_floyd_df.iterrows():
    nodes_to_link_attributes['{}_{}'.format(row['aNodes'], row['bNodes'])]={
        'distance': row['distance'],
        'from_coord': [float(row['aNodeLon']), float(row['aNodeLat'])],
        'to_coord': [float(row['bNodeLon']), float(row['bNodeLat'])]}

sim_net_map_node_lls=json.load(open(INT_NET_COORDINATES_PATH))
sim_node_ids=[node for node in sim_net_map_node_lls]
sim_node_lls=[sim_net_map_node_lls[node] for node in sim_node_ids]
int_nodes_kdtree=spatial.KDTree(np.array(sim_node_lls))

# add centroids and closest sim network nodes to portals
for p in portals['features']:
    centroid=approx_shape_centroid(p['geometry'])
    p['properties']['centroid']=centroid
    p['properties']['closest_nodes']=[sim_node_ids[n_ind] for n_ind in 
      int_nodes_kdtree.query(centroid, 3)[1]]

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
# Get the grid data
# Interactive grid parameters
try:
    with urllib.request.urlopen(cityIO_grid_url+'/header/spatial') as url:
    #get the latest grid data
        cityIO_spatial_data=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))
    cityIO_spatial_data=cityIO_data['header']['spatial']
    
# Full meta grid geojson      
try:
    with urllib.request.urlopen(cityIO_grid_url+'/meta_grid') as url:
    #get the latest grid data
        meta_grid=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    meta_grid=json.load(open(META_GRID_SAMPLE_PATH))
    
# create a lookup from interactive grid to meta_grid
# and a dict of static land uses to their locations in the meta_grid
int_to_meta_grid={}
static_land_uses={}
for fi, f in enumerate(meta_grid['features']):
    if f['properties']['interactive']:
        int_to_meta_grid[int(f['properties']['interactive_id'])]=fi
    else:
        this_land_use_input=f['properties']['land_use']
        this_land_use_standard=get_standard_lu_from_base(this_land_use_input)
        if this_land_use_standard in static_land_uses:
            static_land_uses[this_land_use_standard].append(fi)
        else:
            static_land_uses[this_land_use_standard]=[fi]

# add centroids and closest nodes in sim network to meta_grid_cells
for cell in meta_grid['features']:
    centroid=approx_shape_centroid(cell['geometry'])
    cell['properties']['centroid']=centroid
    cell['properties']['closest_nodes']=[sim_node_ids[n_ind] for n_ind in 
      int_nodes_kdtree.query(centroid, 3)[1]]
    
meta_grid_ll=[meta_grid['features'][i][
        'geometry']['coordinates'][0][0
        ] for i in range(len(meta_grid['features']))]

grid_points_ll=[meta_grid_ll[int_to_meta_grid[int_grid_cell]]
                 for int_grid_cell in int_to_meta_grid]

# create a lookup from interactive grid to puma
puma_shape=json.load(open(PUMA_SHAPE_PATH))
puma_order=[f['properties']['PUMACE10'] for f in puma_shape['features']]
puma_included=json.load(open(PUMAS_INCLUDED_PATH)) 
puma_path_dict = {feature["properties"]["GEOID10"][2:]: mplPath.Path(feature["geometry"]["coordinates"][0][0]) 
                   for feature in puma_shape['features'] if feature["properties"]["GEOID10"][2:] in puma_included}
int_grid_to_puma = {'g'+str(grid_id): None for grid_id in range(len(grid_points_ll))}
for grid_id, grid_point_ll in enumerate(grid_points_ll):
    for puma_id, puma_path in puma_path_dict.items():
        if puma_path.contains_point((grid_point_ll[0], grid_point_ll[1])):
            int_grid_to_puma['g'+str(grid_id)] = puma_id
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



#graphs=createGridGraphs(grid_points_ll, graphs, cityIO_spatial_data['nrows'], 
#                        cityIO_spatial_data['ncols'], cityIO_spatial_data['cellSize'])

sim_area_zone_list+=['g'+str(i) for i in range(len(grid_points_ll))]
                   
# =============================================================================
# Population
# =============================================================================

# load sim_persons
base_sim_persons=json.load(open(SIM_POP_PATH))
# load floaters
base_floating_persons=json.load(open(FLOATING_PATH))
# load vacant houses
base_vacant_houses=json.load(open(VACANT_PATH))
for h in base_vacant_houses:
    h['centroid']=all_geoid_centroids[h['home_geoid']]

if base_sim_persons: 
    get_simulation_locations(base_sim_persons)
    get_LLs(base_sim_persons, ['home', 'work'])
    get_route_costs(base_sim_persons)
    predict_modes(base_sim_persons)
    sample_activity_schedules(base_sim_persons)
    post_od_data(base_sim_persons, CITYIO_OUTPUT_PATH+'od')

if base_floating_persons:
    get_LLs(base_floating_persons, ['work'])
#    create_trips(base_sim_persons)
#    post_trips_data(base_sim_persons, CITYIO_OUTPUT_PATH+'trips')

# =============================================================================
# Handle Interactions
# =============================================================================

lastId=0
while True:
#check if grid data changed
    try:
        with urllib.request.urlopen(cityIO_grid_url+'/meta/hashes/grid') as url:
            hash_id=json.loads(url.read().decode())
    except:
        print('Cant access cityIO')
        hash_id=1
    if hash_id==lastId:
        sleep(1)
    else:
        try:
            with urllib.request.urlopen(cityIO_grid_url+'/grid') as url:
                cityIO_grid_data=json.loads(url.read().decode())
        except:
            print('Using static cityIO grid file')
            cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))  
            cityIO_grid_data=cityIO_data['grid']
        start_time=time.time()
        lastId=hash_id
## =============================================================================
##         FAKE DATA FOR SCENAIO EXPLORATION
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(3,5,len(int_to_meta_grid))] # all employment
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(1,3,len(int_to_meta_grid))] # all housing
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(1,5,len(int_to_meta_grid))] # random mix
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(2,4,len(int_to_meta_grid))] # affordable + employment
## =============================================================================
        new_houses=[]
        new_persons=[]
#        new_households=[]  
        int_grid_land_uses=[get_standard_lu_from_input_lu(g[0]) for g in cityIO_grid_data]
        for ht in housing_lus:
            ht_locs=[i for i in range(len(int_grid_land_uses)) if int_grid_land_uses[i]==ht]
            for htl in ht_locs:
                add_house=housing_lus[ht].copy()
                add_house['home_geoid']='g'+str(htl)
                add_house['centroid']=grid_points_ll[htl]
                new_houses.append(add_house)        
#        # for each office unit, add a (homeless) person
        random.seed(0)
        for et in employment_lus:
            et_locs=[i for i in range(len(int_grid_land_uses)) if int_grid_land_uses[i]==et]
            for etl in et_locs:
                for i in range(NEW_PERSONS_PER_BLD):
                    add_person=random.choice(base_floating_persons).copy()
                    add_person['work_geoid']='g'+str(etl)
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
                h['puma10'] = int_grid_to_puma[h['home_geoid']]
            h['puma10'] = str(h['puma10']).zfill(5)
        top_n_pumas = 5
        home_location_choices(vacant_houses, floating_persons)
        # new_sim_people = people living/working in simzone
        new_sim_persons=[p for p in floating_persons if
                         (p['home_geoid'] in sim_area_zone_list or
                          p['work_geoid'] in sim_area_zone_list)]
        pop_diversity=get_pop_diversity(base_sim_persons+ new_sim_persons)
        lu_diversity=get_lu_diversity(int_grid_land_uses)
        post_diversity_indicators(pop_diversity, lu_diversity, CITYIO_OUTPUT_PATH+'ind_diversity')
        get_LLs(new_sim_persons, ['home', 'work'])
        get_simulation_locations(new_sim_persons)
        get_route_costs(new_sim_persons)
        predict_modes(new_sim_persons)
        sample_activity_schedules(new_sim_persons)
        post_od_data(base_sim_persons+ new_sim_persons, CITYIO_OUTPUT_PATH+'od')
#        create_trips(new_sim_persons)
#        post_trips_data(base_sim_persons+ new_sim_persons, CITYIO_OUTPUT_PATH+'trips')
        finish_time=time.time()
        print('Response time: '+ str(finish_time-start_time))
        sleep(0.2)