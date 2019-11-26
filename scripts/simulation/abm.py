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
#from scipy import spatial
import requests
from time import sleep
import time
import sys

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
                relevant_land_use_codes=land_use_codes[place]
                possible_locations=[static_land_uses[rlu] for rlu in relevant_land_use_codes]
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
            
def approx_route_costs(start_coord, end_coord):
    approx_speeds_met_s={'driving':20/3.6,
        'cycling':10/3.6,
        'walking':3/3.6,
        'pt': 15/3.6 
        }
    straight_line_commute=get_haversine_distance(start_coord, end_coord)
    routes={}
    for mode in range(4):
        routes[mode]={'route': {'driving':0, 'walking':0, 'waiting':0,
                      'cycling':0, 'pt':0}, 'external_time':0}
    routes[0]['route']['driving']=(straight_line_commute/approx_speeds_met_s['driving'])/60
    routes[1]['route']['cycling']=(straight_line_commute/approx_speeds_met_s['cycling'])/60
    routes[2]['route']['walking']=(straight_line_commute/approx_speeds_met_s['walking'])/60
    routes[3]['route']['pt']=(straight_line_commute/approx_speeds_met_s['pt'])/60
    routes[3]['route']['walking']=(200/approx_speeds_met_s['walking'])/60
    return routes
    
        
def get_route_costs(persons):
    """ takes a list of person objects 
    and finds the travel time costs of travel by each mode
    modifies in place
    """
    for p in persons:
        p['routes']={}
        if ((p['home_sim']['type']=='meta_grid') and  (p['work_sim']['type']=='meta_grid')):
            p['type']=0 # lives and works on site
            home_coord=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
            work_coord=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
            p['routes']=approx_route_costs(home_coord, work_coord)
        elif p['work_sim']['type']=='meta_grid':
            p['type']=1 # commute_in
            work_coord=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
            for m in range(4):
                p['routes'][m]={}
                best_portal_route_time=float('inf')
                for portal in range(len(portals['features'])):
                    portal_coord=[nodes_xy[m]['p'+str(portal)]['x'], 
                                  nodes_xy[m]['p'+str(portal)]['y']]
                    internal_portal_route=approx_route_costs(portal_coord, work_coord)[m]
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
            home_coord=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
            for m in range(4):
                p['routes'][m]={}
                best_portal_route_time=float('inf')
                for portal in range(len(portals['features'])):
                    portal_coord=[nodes_xy[m]['p'+str(portal)]['x'], 
                                  nodes_xy[m]['p'+str(portal)]['y']]
                    internal_portal_route=approx_route_costs(home_coord, portal_coord)[m]
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
            p['home_sim']['ll']=[nodes_xy[0]['p'+str(p['home_sim']['ind'])]['x'],
                                 nodes_xy[0]['p'+str(p['home_sim']['ind'])]['y']]
            p['work_sim']['ll']=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
        elif p['work_sim']['type']=='portal': 
            p['work_sim']['ind']=p['routes'][chosen_mode]['portal']
            p['work_sim']['ll']=[nodes_xy[0]['p'+str(p['work_sim']['ind'])]['x'],
                     nodes_xy[0]['p'+str(p['work_sim']['ind'])]['y']]
            p['home_sim']['ll']=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
        else:
            p['home_sim']['ll']=meta_grid['features'][p['home_sim']['ind']]['properties']['centroid']
            p['work_sim']['ll']=meta_grid['features'][p['work_sim']['ind']]['properties']['centroid']
        p['external_time']=p['routes'][chosen_mode]['external_time']

def sample_activity_schedules(persons):
    for p in persons:
        matching_persons=activity_sched.loc[((activity_sched['income']==p['income'])&
                                (activity_sched['age']==p['age'])&
                                (activity_sched['children']==p['children']))]
        if len(matching_persons)>0:
            sampled_person=matching_persons.sample(1)
        else:
            # sampling activity schedule from all people
            sampled_person=activity_sched.sample(1)
        p['activities']=sampled_person.iloc[0]['activities'].split('_')
        if len(p['activities'])>1:
            start_times_str=sampled_person.iloc[0]['start_times'].split('_')
        else:
            start_times_str=[]
        p['start_times']=[int(st) for st in start_times_str]
                                

def post_od_data(persons, destination_address):
    od_str=json.dumps([{'home_ll': p['home_sim']['ll'],
                       'work_ll': p['work_sim']['ll'],
                       'home_sim': p['home_sim'],
                       'work_sim': p['work_sim'],
                       'type': p['type'],
                       'mode': p['mode'],
                       'activities': p['activities'],
                       'activity_start_times':p['start_times'],
                       'start_time': p['start_times'][0]+p['external_time']
                       } for p in persons if len(p['activities'])>1])
    try:
        r = requests.post(destination_address, data = od_str)
        print(r)
    except requests.exceptions.RequestException as e:
        print('Couldnt send to cityio')
        
def post_arc_data(persons, destination_address):
    persons_df=pd.DataFrame(persons)
#    if type==1, arc is from actual home (geoid) to sim_home (portal)
#    if type==2, arc is from sim_work (portal) to actual work (geoid)
    arcs=[]
    for ind, row in persons_df.iterrows():
        if row['type']==1:
            arcs.append({'start_latlon': row['home_ll'],
                         'end_latlon': row['home_node_ll'],
                         'mode': row['mode']})
        elif row['type']==2:
            arcs.append({'start_latlon': row['work_node_ll'],
                         'end_latlon': row['work_ll'],
                         'mode': row['mode']}) 
    try:
        r = requests.post(destination_address, data = json.dumps(arcs))
        print(r)
    except requests.exceptions.RequestException as e:
        print('Couldnt send to cityio')
          


def create_long_record(person, house, choice_id):
    """ takes a house object and a household object and 
    creates a row for the MNL long data frame 
    """
    beds=min(3, max(1, house['beds']))
    norm_rent=(house['rent']-rent_normalisation['mean'][str(int(beds))])/rent_normalisation['std'][str(int(beds))]
    return {'norm_rent': norm_rent,
            'work_dist': get_haversine_distance(p['work_ll'], h['centroid']),
            'puma_pop_per_sqmeter': house['puma_pop_per_sqmeter'],
            'income_disparity': np.abs(house['puma_med_income']-person['HINCP']),
            'built_since_jan2010': house['built_since_jan2010'],
            'custom_id': person['person_id'],
            'choice_id': choice_id,
            'actual_house_id':house['house_id']}  
#        
def home_location_choices(houses, persons):
    """ takes the house and person objects
    finds the vacant houses and homeless persons
    chooses a housing unit for each person
    modifies the house and person objects in place
    """
    long_data=[]
    # for each household, sample N potential housing choices
    # and add them to the long data frame
    for p in persons:
        #choose N houses
        h_alts=random.sample(houses, 9)
        for hi, h in enumerate(h_alts):
            long_record=create_long_record(p, h, hi+1)
            long_data.append(long_record)             
#             long_data.append(h.long_data_record(hh.hh_income, hh.household_id, hi+1, rent_normalisation))
    long_df=pd.DataFrame(long_data)
    # TODO: why do some houses have nan for norm_rent
    long_df.loc[long_df['norm_rent'].isnull(), 'norm_rent']=0
    long_df['predictions']=home_loc_logit.predict(long_df)
    for p_ind in set(long_df['custom_id']):
        # find maximum prob or sample from probs in subset of long_df
        house_id=np.random.choice(long_df.loc[long_df['custom_id']==p_ind, 'actual_house_id'], 
                                  p=long_df.loc[long_df['custom_id']==p_ind, 'predictions'])
        persons[p_ind]['house_id']=house_id
        persons[p_ind]['home_geoid']=houses[house_id]['home_geoid']
        # update characterictics of persons in these households

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
    all_lu_pop=[int(gd[0]) for gd in grid_data]
    housing_pop=[lu for lu in all_lu_pop if lu in housing_types]
    lu_diversity['housing']=shannon_equitability(housing_pop, set(housing_types))
    office_pop=[lu for lu in all_lu_pop if lu in employment_types]
    lu_diversity['office']=shannon_equitability(office_pop, set(employment_types))
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

ACTIVITY_SCHED_PATH='./scripts/cities/'+city+'/clean/person_sched.csv'

#Road network graph
PORTALS_PATH='./scripts/cities/'+city+'/clean/portals.geojson'
ROUTE_COSTS_PATH='./scripts/cities/'+city+'/clean/route_costs.json'
SIM_GRAPHS_PATH='./scripts/cities/'+city+'/clean/sim_area_nets.p'

META_GRID_SAMPLE_PATH='./scripts/cities/'+city+'/clean/meta_grid.geojson'
GRID_INT_SAMPLE_PATH='./scripts/cities/'+city+'/clean/grid_interactive.geojson'



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
housing_types={1:{'rent': 800, 'beds': 2, 'built_since_jan2010': True, 
                  'puma_pop_per_sqmeter': 0.000292, 'puma_med_income': 60000,
                  'pop_per_sqmile': 5000, 'tenure': 'rented'},
               2:{'rent': 1500, 'beds': 2, 'built_since_jan2010': True, 
                  'puma_pop_per_sqmeter': 0.000292, 'puma_med_income': 60000,
                  'pop_per_sqmile': 5000, 'tenure': 'rented'}}
# TODO: number of each employment sector for each building type
employment_types= {3:{},4:{}}

land_use_codes={'home': ['R'], 'work': ['M', 'B']}

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
#load the network graphs
#graphs=pickle.load(open(SIM_GRAPHS_PATH, 'rb'))
# load the external route costs
ext_route_costs=json.load(open(ROUTE_COSTS_PATH))
activity_sched=pd.read_csv(ACTIVITY_SCHED_PATH)

#for graph in graphs:
#    graphs[graph]['kdtree']=spatial.KDTree(
#            np.array(graphs[graph]['nodes'][['x', 'y']]))

#nodes=pd.read_csv(NODES_PATH)
# load the zones geojson
all_zones=json.load(open(ALL_ZONES_PATH))
sim_zones=json.load(open(SIM_ZONES_PATH))
portals=json.load(open(PORTALS_PATH))

# add centroids to portals
for p in portals['features']:
    p['properties']['centroid']=approx_shape_centroid(p['geometry'])


if city=='Hamburg':
    geoid_order_all=[f['properties']['GEO_ID'] for f in all_zones['features']]
    sim_area_zone_list=sim_zones
else:
    geoid_order_all=[f['properties']['GEO_ID'].split('US')[1] for f in all_zones['features']]
    sim_area_zone_list=[z.split('US')[1] for z in sim_zones]

all_geoid_centroids={}
for ind, geo_id in enumerate(geoid_order_all):
#    centroid=shape(all_zones['features'][ind]['geometry']).centroid
    centroid=approx_shape_centroid(all_zones['features'][ind]['geometry'])
#    all_geoid_centroids[geo_id]=[centroid.x, centroid.y]
    all_geoid_centroids[geo_id]=list(centroid)

# =============================================================================
# Processing of spatial grid data
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
n_cells=cityIO_spatial_data['ncols']*cityIO_spatial_data['nrows']

# Interactive grid geojson    
try:
    with urllib.request.urlopen(cityIO_grid_url+'/grid_interactive_area') as url:
    #get the latest grid data
        grid_interactive=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    grid_interactive=json.load(open(GRID_INT_SAMPLE_PATH))
    
# Full meta grid geojson      
try:
    with urllib.request.urlopen(cityIO_grid_url+'/meta_grid') as url:
    #get the latest grid data
        meta_grid=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    meta_grid=json.load(open(META_GRID_SAMPLE_PATH))
    
# create a lookup from interactive grid to meta_grid
# and a dict of statuc land uses to their locations in the meta_grid
int_to_meta_grid={}
static_land_uses={}
for fi, f in enumerate(meta_grid['features']):
    if f['properties']['interactive']:
        int_to_meta_grid[int(f['properties']['interactive_id'])]=fi
    else:
        this_land_use=f['properties']['land_use']
        if not this_land_use:
            this_land_use='None'
        this_land_use_simple=this_land_use[0]
        if this_land_use_simple in static_land_uses:
            static_land_uses[this_land_use_simple].append(fi)
        else:
            static_land_uses[this_land_use_simple]=[fi]

# add centroids to meta_grid_cells
for cell in meta_grid['features']:
    cell['properties']['centroid']=approx_shape_centroid(cell['geometry'])
    

grid_points_ll=[meta_grid['features'][int_to_meta_grid[int_grid_cell]][
        'geometry']['coordinates'][0][0
        ] for int_grid_cell in range(n_cells)]


sim_area_zone_list+=['g'+str(i) for i in range(len(grid_points_ll))]
#
# =============================================================================
# Node Locations
# =============================================================================

# create a list of nodes with their coords for each mode
nodes_xy={}
for mode in mode_graphs:
    nodes_xy[mode]={}
#    for i in range(len(graphs[mode_graphs[mode]]['nodes'])):
#        nodes_xy[mode][i]={'x':graphs[mode_graphs[mode]]['nodes'].iloc[i]['x'],
#             'y':graphs[mode_graphs[mode]]['nodes'].iloc[i]['y']}
#    for i in range(len(grid_points_ll)):
#        nodes_xy[mode]['g'+str(i)]={'x':grid_points_ll[i][0],
#             'y':grid_points_ll[i][1]}
    for p in range(len(portals['features'])):
#        p_centroid=shape(portals['features'][p]['geometry']).centroid
        p_centroid=approx_shape_centroid(portals['features'][p]['geometry'])
#        nodes_xy[mode]['p'+str(p)]={'x':p_centroid.x,
#             'y':p_centroid.y}
        nodes_xy[mode]['p'+str(p)]={'x':p_centroid[0],
             'y':p_centroid[1]}
                   
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
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(3,5,len(cityIO_grid_data))] # all employment
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(1,3,len(cityIO_grid_data))] # all housing
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(1,5,len(cityIO_grid_data))] # random mix
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(2,4,len(cityIO_grid_data))] # affordable + employment
## =============================================================================
        new_houses=[]
        new_persons=[]
#        new_households=[]        
        for ht in housing_types:
            ht_locs=[i for i in range(len(cityIO_grid_data)) if cityIO_grid_data[i][0]==ht]
            for htl in ht_locs:
                add_house=housing_types[ht].copy()
                add_house['home_geoid']='g'+str(htl)
                add_house['centroid']=grid_points_ll[htl]
                new_houses.append(add_house)        
#        # for each office unit, add a (homeless) person
        random.seed(0)
        for et in employment_types:
            et_locs=[i for i in range(len(cityIO_grid_data)) if cityIO_grid_data[i][0]==et]
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
        home_location_choices(vacant_houses, floating_persons)
        # new_sim_people = people living/working in simzone
        new_sim_persons=[p for p in floating_persons if
                         (p['home_geoid'] in sim_area_zone_list or
                          p['work_geoid'] in sim_area_zone_list)]
        pop_diversity=get_pop_diversity(base_sim_persons+ new_sim_persons)
        lu_diversity=get_lu_diversity(cityIO_grid_data)
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