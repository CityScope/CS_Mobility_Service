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
import pyproj
import math
import pandas as pd
from shapely.geometry import Point,shape
import numpy as np
import networkx as nx
from scipy import spatial
import requests
from time import sleep

# =============================================================================
# Functions
# =============================================================================
def createGrid(topLeft_lonLat, topEdge_lonLat, utm, wgs, cell_size, nrows, ncols, graphs):
    """
    takes the spatial information from cityIO and generates 
    grid_coords_ll: the coordinates of ach grid cell left to right, top to bottom
    G: a network of roads around the cells
    """
    topLeftXY=pyproj.transform(wgs, utm,topLeft_lonLat['lon'], topLeft_lonLat['lat'])
    topEdgeXY=pyproj.transform(wgs, utm,topEdge_lonLat['lon'], topEdge_lonLat['lat'])
    dydx=(topEdgeXY[1]-topLeftXY[1])/(topEdgeXY[0]-topLeftXY[0])
    theta=math.atan((dydx))
    cosTheta=math.cos(theta)
    sinTheta=math.sin(theta)
    x_unRot=[j*cell_size for i in range(nrows) for j in range(ncols)]
    y_unRot=[-i*cell_size for i in range(nrows) for j in range(ncols)]
    # use the rotation matrix to rotate around the origin
    x_rot=[x_unRot[i]*cosTheta -y_unRot[i]*sinTheta for i in range(len(x_unRot))]
    y_rot=[x_unRot[i]*sinTheta +y_unRot[i]*cosTheta for i in range(len(x_unRot))]
    x_rot_trans=[topLeftXY[0]+x_rot[i] for i in range(len(x_rot))]
    y_rot_trans=[topLeftXY[1]+y_rot[i] for i in range(len(x_rot))]
    lon_grid, lat_grid=pyproj.transform(utm,wgs,x_rot_trans, y_rot_trans)
    grid_coords_ll=[[lon_grid[i], lat_grid[i]] for i in range(len(lon_grid))]
    for mode in graphs:
#    create graph internal to the grid
        graphs[mode]['graph'].add_nodes_from('g'+str(n) for n in range(len(grid_coords_ll)))
        for c in range(ncols):
            for r in range(nrows):
                # if not at the end of a row, add h link
                if not c==ncols-1:
                    graphs[mode]['graph'].add_edge('g'+str(r*ncols+c), 'g'+str(r*ncols+c+1), 
                          attr_dict={'distance': cell_size, 'weight_minutes':(cell_size/1609)/(3*60)})
                    graphs[mode]['graph'].add_edge('g'+str(r*ncols+c+1), 'g'+str(r*ncols+c), 
                          attr_dict={'distance': cell_size, 'weight_minutes':(cell_size/1609)/(3*60)})
                # if not at the end of a column, add v link
                if not r==nrows-1:
                    graphs[mode]['graph'].add_edge('g'+str(r*ncols+c), 'g'+str((r+1)*ncols+c), 
                          attr_dict={'distance': cell_size, 'weight_minutes':(cell_size/1609)/(3*60)})
                    graphs[mode]['graph'].add_edge('g'+str((r+1)*ncols+c), 'g'+str(r*ncols+c), 
                          attr_dict={'distance': cell_size, 'weight_minutes':(cell_size/1609)/(3*60)})
        # create links between the 4 corners of the grid and the road network
        kd_tree_nodes=spatial.KDTree(np.array(graphs[mode]['nodes'][['x', 'y']]))
        for n in [0, ncols-1, (nrows-1)*ncols, (nrows*ncols)-1]: 
            closest=kd_tree_nodes.query(grid_coords_ll[n], k=1)[1]
            graphs[mode]['graph'].add_edge('g'+str(n), closest, attr_dict={'distance': cell_size, 
                       'weight_minutes':(cell_size/1609)/(3*60)})
            graphs[mode]['graph'].add_edge(closest, 'g'+str(n), attr_dict={'distance': cell_size, 
                       'weight_minutes':(cell_size/1609)/(3*60)})
    return grid_coords_ll, graphs 

def get_grid_geojson(grid_coords_ll, grid, ncols):
    delta_ll_across=[grid_coords_ll[1][0]-grid_coords_ll[0][0], grid_coords_ll[1][1]-grid_coords_ll[0][1]]
    delta_ll_down=[grid_coords_ll[ncols][0]-grid_coords_ll[0][0], grid_coords_ll[ncols][1]-grid_coords_ll[0][1]]
    features=[]
    for i, g in enumerate(grid_coords_ll):
        coords=[g, 
                [g[0]+delta_ll_down[0], g[1]+delta_ll_down[1]],
                [g[0]+delta_ll_across[0]+delta_ll_down[0], g[1]+delta_ll_across[1]+delta_ll_down[1]],
                [g[0]+delta_ll_across[0],g[1]+delta_ll_across[1]], 
                g]
        features.append({'type': 'Feature',
                         'geometry':{'type': 'Polygon', 'coordinates': [coords]},
                         'properties': {'usage': grid[i][0]}})
    geojson_object={'type': 'FeatureCollection',
                    'features': features}
    return geojson_object

def random_points_within(poly, num_points):
    """ takes a polygon such as an admin boundary or building and selects 
    a random point inside using rejection sampling
    """
    min_x, min_y, max_x, max_y = poly.bounds
    points = []
    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)
    return points

def get_LLs(persons):
    """ takes a list of person objects and 
    finds home and work coordinates for them
    modifies in place
    """
    for p in persons:  
        for place in ['home', 'work']:
            geoid=p[place+'_geoid']
            if 'g' in str(geoid):
                ll=grid_points_ll[int(geoid[1:])]
            else:
                poly=shape(all_zones['features'][geoid_order_all.index(geoid)]['geometry'])
                ll=random_points_within(poly,1)[0]
            p[place+'_ll']=[ll.x, ll.y]

def find_route_multi(start_nodes, end_nodes, graph, weight):
    """
    tries to find paths between lists of possible start and end nodes
    Once a path is successfully found it is returned. Otherwise returns None
    """
    for sn in start_nodes:
        for en in end_nodes:
            try:
                node_path=nx.shortest_path(graph,sn,en, weight=weight)
                return node_path
            except:
                pass
    return None

def get_route_costs(start_nodes, end_nodes, graph, weight):
    node_route=find_route_multi(start_nodes, end_nodes, 
                                            graph, weight)
    if node_route:
        weights=[graph[node_route[i]][node_route[i+1]]['attr_dict'][weight] 
            for i in range(len(node_route)-1)]
        types=[graph[node_route[i]][node_route[i+1]]['attr_dict']['type'] 
            for i in range(len(node_route)-1)]
        route={'node_route': node_route}
        for c in ['driving', 'walking', 'waiting',
                  'cycling', 'PT']:
            route[c]=sum([weights[i] for i in range(len(weights)
            ) if types[i]==c])
    else:
        route={'node_route': [start_nodes[0], end_nodes[0]]}
        for c in ['driving', 'walking', 'waiting',
                  'cycling', 'PT']:
            route[c]=1000
    return route
        
def get_routes(persons):
    """ takes a list of person objects 
    and finds the travel time costs of travel by each mode
    modifies in place
    """
    for p in persons:
        p['routes']={}
        start_time=7*60*60+random.choice(range(0,3*60*60))
        if p['home_geoid'] in sim_area_zone_list and p['work_geoid'] in sim_area_zone_list:
            for m in range(4):
                if 'g' in str(p['home_geoid']):
                    home_node_list=[p['home_geoid']]
                else:
                    home_node_list=graphs[mode_graphs[m]]['kdtree'].query(np.array(p['home_ll']), 5)[1]
                if 'g' in str(p['work_geoid']):
                    work_node_list=[p['work_geoid']]
                else:
                    work_node_list=graphs[mode_graphs[m]]['kdtree'].query(np.array(p['work_ll']), 5)[1]
                p['routes'][m]=get_route_costs(home_node_list, work_node_list, 
                                            graphs[mode_graphs[m]]['graph'], 'weight_minutes')
                p['routes'][m]['sim_start_time']=start_time
        elif p['work_geoid'] in sim_area_zone_list:
            for m in range(4):
                portal_routes={}
                best_portal_route_time=float('inf')
                if 'g' in str(p['work_geoid']):
                    work_node_list=[p['work_geoid']]
                else:
                    work_node_list=graphs[mode_graphs[m]]['kdtree'].query(np.array(p['work_ll']), 5)[1]
                for portal in range(len(ext_route_costs[mode_graphs[m]][str(p['home_geoid'])])):
                    # get route from home zone to portal by this mode
                    route_to_portal=ext_route_costs[mode_graphs[m]][str(p['home_geoid'])][str(portal)] 
                    portal_routes[portal]=get_route_costs(['p'+str(portal)], work_node_list, 
                                            graphs[mode_graphs[m]]['graph'], 'weight_minutes')
                    for c in ['driving', 'walking', 'waiting',
                              'cycling', 'PT']:
                        portal_routes[portal][c]+=route_to_portal[c]
                    total_time=sum([portal_routes[portal][c] for c in [
                            'driving', 'walking', 'cycling', 'PT']])
                    if total_time<best_portal_route_time:
                        best_portal=portal
                        best_portal_route_time=total_time
                p['routes'][m]=portal_routes[best_portal]
                p['routes'][m]['sim_start_time']=int(start_time+best_portal_route_time*60)
        elif p['home_geoid'] in sim_area_zone_list:
            for m in range(4):
                portal_routes={}
                best_portal_route_time=float('inf')
                if 'g' in str(p['home_geoid']):
                    home_node_list=[p['home_geoid']]
                else:
                    home_node_list=graphs[mode_graphs[m]]['kdtree'].query(np.array(p['home_ll']), 5)[1]
                for portal in range(len(ext_route_costs[mode_graphs[m]][str(p['work_geoid'])])):
                    # get route from home zone to portal by this mode
                    route_from_portal=ext_route_costs[mode_graphs[m]][str(p['work_geoid'])][str(portal)] 
                    portal_routes[portal]=get_route_costs( home_node_list, ['p'+str(portal)],
                                            graphs[mode_graphs[m]]['graph'], 'weight_minutes')
                    for c in ['driving', 'walking', 'waiting',
                              'cycling', 'PT']:
                        portal_routes[portal][c]+=route_from_portal[c]
                    total_time=sum([portal_routes[portal][c] for c in [
                            'driving', 'walking', 'cycling', 'PT']])
                    if total_time<best_portal_route_time:
                        best_portal=portal
                        best_portal_route_time=total_time
                p['routes'][m]=portal_routes[best_portal]
                p['routes'][m]['sim_start_time']=start_time   
def predict_modes(persons):
    """ takes list of person objects and 
    predicts transport modes for each person's commute
    modifies in place
    """
    feature_df=pd.DataFrame(persons)  
#    feature_df['bach_degree_yes']=feature_df['SCHL']>20
#    feature_df['bach_degree_no']=~feature_df['bach_degree_yes']
    for feat in ['income', 'age', 'children', 'workers', 'tenure', 'sex', 'bach_degree']:
        new_dummys=pd.get_dummies(feature_df[feat], prefix=feat)
        feature_df=pd.concat([feature_df, new_dummys],  axis=1)
    # TODO: better method of predicting travel times
    # routing engine or feedback from simulation
    feature_df['drive_time_minutes']=  feature_df.apply(lambda row: row['routes'][0]['driving'], axis=1)     
    feature_df['cycle_time_minutes']=  feature_df.apply(lambda row: row['routes'][1]['cycling'], axis=1)     
    feature_df['walk_time_minutes']=  feature_df.apply(lambda row: row['routes'][2]['walking'], axis=1)     
    feature_df['PT_time_minutes']=  feature_df.apply(lambda row: row['routes'][3]['PT'], axis=1)
    feature_df['walk_time_PT_minutes']=feature_df.apply(lambda row: row['routes'][3]['walking'], axis=1)  
    feature_df['drive_time_PT_minutes']=0 
    # TODO: below should come directly from the path-finding
    feature_df['network_dist_km']=feature_df.apply(lambda row: row['drive_time_minutes']*30/60, axis=1) 
    # TODO: change below if modelling housing sales as well
    feature_df['tenure_owned']=False
    feature_df['tenure_other']=False
    feature_df['purpose_HBW']=1
    feature_df['purpose_NHB']=0
    feature_df['purpose_HBO']=0
    assert all([rff in feature_df.columns for rff in rf_features]
    ),"Features in table dont match features in RF model"
    feature_df=feature_df[rf_features]#reorder columns to match rf model
    mode_probs=mode_rf.predict_proba(feature_df)
    for i,p in enumerate(persons): 
        chosen_mode=int(np.random.choice(range(4), size=1, replace=False, p=mode_probs[i])[0])
        p['mode']=chosen_mode
        home_node=p['routes'][chosen_mode]['node_route'][0]
        work_node=p['routes'][chosen_mode]['node_route'][-1]
        p['home_node_ll']=[nodes_xy[chosen_mode][home_node]['x'], 
                           nodes_xy[chosen_mode][home_node]['y']]
        p['work_node_ll']=[nodes_xy[chosen_mode][work_node]['x'], 
                           nodes_xy[chosen_mode][work_node]['y']]
        p['sim_start_time']=p['routes'][chosen_mode]['sim_start_time']

def post_od_data(persons, destination_address):
    od_str=json.dumps([{'home_ll': p['home_node_ll'],
                       'work_ll': p['work_node_ll'],
                       'mode': p['mode'],
                       'start_time': p['sim_start_time']} for p in persons])
    try:
        r = requests.post(destination_address, data = od_str)
        print(r)
    except requests.exceptions.RequestException as e:
        print('Couldnt send to cityio')
    
#        
#def create_trips(persons):
#    """ returns a trip objects for each person
#    each  trip object contains a list of [lon, lat, timestamp] coordinates
#    this is the format required by the deckGL trips layer
#    modifies in place
#    """
#    for p in persons:
#        speed_met_s=SPEEDS_MET_S[p['mode']]
#        p['kgCO2']=2* kgCO2PerMet[p['mode']]* (p['network_dist_km']/1000)
#        route_coords=[node_coords[p['node_route'][n]] for n in range(len(p['node_route']))]
#        p['trip']=[[int(1e5*route_coords[n][0])/1e5, # reduce precision
#                    int(1e5*route_coords[n][1])/1e5,
#                    int(p['start_time']+p['cum_dist_m'][n]/speed_met_s)] for n in range(len(p['node_route']))]
# 
#def post_trips_data(persons, destination_address):
#    """ posts trip data json to cityIO
#    """
#    trips_str=json.dumps([{'mode': p['mode'], 'segments': p['trip']} for p in persons]) 
#    try:
#        r = requests.post(destination_address, data = trips_str)
#        print(r)
#    except requests.exceptions.RequestException as e:
#        print('Couldnt send to cityio')
#        
#def post_grid_geojson(grid_geo, destination_address):
#    """ posts grid geojson to cityIO
#    """
#    try:
#        r = requests.post(destination_address, data = json.dumps(grid_geo))
#        print(r)
#    except requests.exceptions.RequestException as e:
#        print('Couldnt send grid geojson to cityio')
#        
#def create_long_record(household, house, choice_id):
#    """ takes a house object and a household object and 
#    creates a row for the MNL long data frame 
#    """
#    beds=min(3, max(1, house['beds']))
#    norm_rent=(house['rent']-rent_normalisation['mean'][str(int(beds))])/rent_normalisation['std'][str(int(beds))]
#    return {'norm_rent': norm_rent,
#            'puma_pop_per_sqmeter': house['puma_pop_per_sqmeter'],
#            'income_disparity': np.abs(house['puma_med_income']-household['HINCP']),
#            'built_since_jan2010': house['built_since_jan2010'],
#            'custom_id': household['household_id'],
#            'choice_id': choice_id,
#            'actual_house_id':house['house_id']}  
#        
#def home_location_choices(houses, households, persons):
#    """ takes the house, household and person objects
#    finds the vacant houses and homeless households
#    chooses a housing unit for each household
#    modifies the house, household and person objects in place
#    """
#    # identify vacant and homeless
#    global moved_persons
#    vacant_housing = [h for h in houses if h['household_id']==None]
#    homeless_hhs= [hh for hh in households if hh['house_id']==None]
#    # build long dataframe
#    long_data=[]
#    # for each household, sample N potential housing choices
#    # and add them to the long data frame
#    for hh in homeless_hhs:
#        #choose N houses
#        h_alts=random.sample(vacant_housing, 9)
#        for hi, h in enumerate(h_alts):
#            long_record=create_long_record(hh, h, hi+1)
#            long_data.append(long_record)             
##             long_data.append(h.long_data_record(hh.hh_income, hh.household_id, hi+1, rent_normalisation))
#    long_df=pd.DataFrame(long_data)
#    long_df['predictions']=home_loc_logit.predict(long_df)
#    for hh_ind in set(long_df['custom_id']):
#        # find maximum prob or sample from probs in subset of long_df
#        house_id=np.random.choice(long_df.loc[long_df['custom_id']==hh_ind, 'actual_house_id'], p=long_df.loc[long_df['custom_id']==hh_ind, 'predictions'])
##        house_id=int(long_df.loc[long_df[long_df['custom_id']==hh_ind]['predictions'].idxmax(), 'actual_house_id'])
#        households[hh_ind]['house_id']=house_id 
#        houses[house_id]['household_id']=hh_ind
#        # update characterictics of persons in these households
#    for p in persons: 
##        print(p['household_id'])
#        if p['household_id'] in set(long_df['custom_id']):
##            print('yes')
#            moved_persons+=[p['person_id']]
#            for col in person_cols_hh:
#                p[col]=households[p['household_id']][col]
# =============================================================================
# Parameters
# =============================================================================
city='Hamburg'
send_to_cityIO=True

# =============================================================================
# Constants
# =============================================================================

ALL_ZONES_PATH='../cities/'+city+'/clean/model_area.geojson'
SIM_ZONES_PATH='../cities/'+city+'/clean/sim_area.geojson'
# Synthpop results
SIM_POP_PATH='../cities/'+city+'/clean/sim_pop.json'
VACANT_PATH='../cities/'+city+'/clean/vacant.json'
FLOATING_PATH='../cities/'+city+'/clean/floating.json'
# Mode choice model
FITTED_MODE_MODEL_PATH='../models/trip_mode_rf.p'
RF_FEATURES_LIST_PATH='../models/rf_features.json'
# Home location choice model
FITTED_HOME_LOC_MODEL_PATH='../models/home_loc_logit.p'
RENT_NORM_PATH='../models/rent_norm.json'

#Road network graph
PORTALS_PATH='../cities/'+city+'/clean/portals.geojson'
ROUTE_COSTS_PATH='../cities/'+city+'/clean/route_costs.json'
SIM_GRAPHS_PATH='../cities/'+city+'/clean/sim_area_nets.p'



# the graph used by each mode
mode_graphs={0:'driving',
             1:'cycling',
             2:'walking',
             3:'pt'}

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

# Projection systems
UTM_MAP={'Boston':pyproj.Proj("+init=EPSG:32619"),
     'Hamburg':pyproj.Proj("+init=EPSG:32632")}
utm=UTM_MAP[city]
wgs=pyproj.Proj("+init=EPSG:4326")


# #cityIO grid data
table_name_map={'Boston':"mocho",
     'Hamburg':"grasbrook"}
host='https://cityio.media.mit.edu/'
cityIO_grid_url=host+'api/table/'+table_name_map[city]
UPDATE_FREQ=1 # seconds
CITYIO_SAMPLE_PATH='../'+city+'/clean/sample_cityio_data.json' #cityIO backup data

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
graphs=pickle.load(open(SIM_GRAPHS_PATH, 'rb'))
# load the external route costs
ext_route_costs=json.load(open(ROUTE_COSTS_PATH))

for graph in graphs:
    graphs[graph]['kdtree']=spatial.KDTree(
            np.array(graphs[graph]['nodes'][['x', 'y']]))

#nodes=pd.read_csv(NODES_PATH)
# load the zones geojson
all_zones=json.load(open(ALL_ZONES_PATH))
sim_zones=json.load(open(SIM_ZONES_PATH))
portals=json.load(open(PORTALS_PATH))

if city=='Hamburg':
    geoid_order_all=[f['properties']['GEO_ID'] for f in all_zones['features']]
    geoid_order_sim=[f['properties']['GEO_ID'] for f in sim_zones['features']]
else:
    geoid_order_all=[f['properties']['GEO_ID'].split('US')[1] for f in all_zones['features']]
    geoid_order_sim=[f['properties']['GEO_ID'].split('US')[1] for f in sim_zones['features']]


# =============================================================================
# Processing of spatial grid data
# =============================================================================
try:
    with urllib.request.urlopen(cityIO_grid_url+'/header/spatial') as url:
    #get the latest grid data
        cityIO_spatial_data=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))
    cityIO_spatial_data=cityIO_data['header']['spatial']

# TODO calculate this from cityIO spatial 
position={'Hamburg':{'topleft':{'lat':53.533681, 'lon':10.011585},
                    'topedge':{'lat':53.533433, 'lon':10.012213}},
        'Boston': {'topleft':{'lat':42.365980,    'lon': -71.085560},
                  'topedge': {'lat':42.3649,   'lon':  -71.082947}}}
topLeft_lonLat=position[city]['topleft']
topEdge_lonLat=position[city]['topedge']

grid_points_ll, graphs=createGrid(topLeft_lonLat, topEdge_lonLat, utm, wgs, cityIO_spatial_data['cellSize'], cityIO_spatial_data['nrows'], cityIO_spatial_data['ncols'], graphs)

sim_area_zone_list=geoid_order_sim.copy()+['g'+str(i) for i in range(len(grid_points_ll))]
# =============================================================================
# Locations
# =============================================================================

# create a list of nodes with their coords for each mode
nodes_xy={}
for mode in mode_graphs:
    nodes_xy[mode]={}
    for i in range(len(graphs[mode_graphs[mode]]['nodes'])):
        nodes_xy[mode][i]={'x':graphs[mode_graphs[mode]]['nodes'].iloc[i]['x'],
             'y':graphs[mode_graphs[mode]]['nodes'].iloc[i]['y']}
    for i in range(len(grid_points_ll)):
        nodes_xy[mode]['g'+str(i)]={'x':grid_points_ll[i][0],
             'y':grid_points_ll[i][1]}
    for p in range(len(portals['features'])):
        p_centroid=shape(portals['features'][p]['geometry']).centroid
        nodes_xy[mode]['p'+str(p)]={'x':p_centroid.x,
             'y':p_centroid.y}
## create a list of locations with a list of the associated nodes for each mode
#locations={}
#for i in range(len(geoid_order_sim)):
#    locations[geoid_order_sim[i]]={}
#    zone_shape=shape(sim_zones['features'][i]['geometry'])
#    for mode in mode_graphs:
#        locations[geoid_order_sim[i]][mode]={'nodes':[]}
#        for ni in range(len(graphs[mode_graphs[mode]]['nodes'])):
#            if zone_shape.contains(Point(
#                    [graphs[mode_graphs[mode]]['nodes'].iloc[ni]['x'], 
#                     graphs[mode_graphs[mode]]['nodes'].iloc[ni]['y']])):
#                locations[geoid_order_sim[i]][mode]['nodes'].extend([ni])
#for i in range(len(grid_points_ll)):
#    locations['g'+str(i)]={}
#    for mode in mode_graphs:
#        locations['g'+str(i)][mode]={'nodes':['g'+str(i)]}
                   
# =============================================================================
# Population
# =============================================================================

# load sim_persons
base_sim_persons=json.load(open(SIM_POP_PATH))
# load floaters
floating_persons=json.load(open(FLOATING_PATH))
# load vacant houses
vacant_houses=json.load(open(VACANT_PATH))
    
get_LLs(base_sim_persons)
get_routes(base_sim_persons)
predict_modes(base_sim_persons)
post_od_data(base_sim_persons, CITYIO_OUTPUT_PATH+'od')


    
# =============================================================================
# Handle Interactions
# =============================================================================



#house_cols=['puma10','beds', 'rent', 'tenure','built_since_jan2010', 'home_geoid']
#household_cols=['HINCP', 'cars','NP',
#       'workers', 'tenure', 'children', 'income', 'serialno']
#person_cols=['COW', 'bach_degree', 'age', 'sex']
#person_cols_hh=['income', 'children', 'workers', 'tenure', 'household_id', 'pop_per_sqmile_home']
#
#lastId=0
#while True:
##check if grid data changed
#    try:
#        with urllib.request.urlopen(cityIO_grid_url+'/meta/hashes/grid') as url:
#            hash_id=json.loads(url.read().decode())
#    except:
#        print('Cant access cityIO')
#        hash_id=1
#    if hash_id==lastId:
#        sleep(1)
#    else:
#        try:
#            with urllib.request.urlopen(cityIO_grid_url+'/grid') as url:
#                cityIO_grid_data=json.loads(url.read().decode())
#        except:
#            print('Using static cityIO grid file')
#            cityIO_data=json.load(open(CITYIO_SAMPLE_PATH))  
#            cityIO_grid_data=cityIO_data['grid']
#        for hh_id in mobile_hh_ids:
#            #    set houses of mobile HHs to vacant
#            #    set mobile household to homeless
#            base_houses[hh_id]['household_id']=None
#            base_households[hh_id]['house_id']=None
#        lastId=hash_id
## =============================================================================
##         FAKE DATA FOR SCENAIO EXPLORATION
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(3,5,len(cityIO_grid_data))] # all employment
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(1,3,len(cityIO_grid_data))] # all housing
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(1,5,len(cityIO_grid_data))] # random mix
##        cityIO_grid_data=[[int(i)] for i in np.random.randint(2,4,len(cityIO_grid_data))] # affordable + employment
## =============================================================================
#        grid_geo=get_grid_geojson(grid_points_ll, cityIO_grid_data, cityIO_spatial_data['ncols'])
#        new_houses=[]
#        new_persons=[]
#        new_households=[]        
#        for ht in housing_types:
#            ht_locs=[i for i in range(len(cityIO_grid_data)) if cityIO_grid_data[i][0]==ht]
#            for htl in ht_locs:
#                add_house=housing_types[ht].copy()
#                add_house['house_id']=len(base_houses)+len(new_houses)
#                add_house['household_id']=None
#                add_house['home_geoid']='g'+str(htl)
#                new_houses.append(add_house)        
#        # for each office unit, add a (homeless) household to new_hhs
#        random.seed(0)
#        for et in employment_types:
#            et_locs=[i for i in range(len(cityIO_grid_data)) if cityIO_grid_data[i][0]==et]
#            for etl in et_locs:
#                sample_hh_row=random.choice(base_households).copy()
#                household_id=len(base_households)+len(new_households)
#                # TODO: get pop_per_sqmile from zone data
#                pop_per_sq_mile=5000
#                add_household={col: sample_hh_row[col] for col in household_cols}
#                copied_from_hh_id=sample_hh_row['household_id']
#                add_household['house_id']=None
#                add_household['household_id']=household_id
#                add_household['pop_per_sqmile_home']=pop_per_sq_mile
#                new_households.append(add_household)
#                person_copies=[p.copy() for p in base_persons if p['household_id']==copied_from_hh_id]
##                person_rows=synth_persons_df[synth_persons_df['serialno']==add_household['serialno']].sample(n=add_household['NP'])
#                for pc in person_copies:
#                    add_person={col: pc[col] for col in person_cols}
#                    for hh_col in person_cols_hh:
#                        add_person[hh_col]=add_household[hh_col]
#                    add_person['person_id']=len(base_persons)+len(new_persons)
#                    # TODO, use O-D matrix for work locations
#                    add_person['work_geoid']='g'+str(etl)
#                    new_persons.append(add_person)
#        houses=base_houses+new_houses
#        households=base_households+new_households
#        persons=base_persons+new_persons
#        moved_persons=[]
#        random.seed(0)
#        home_location_choices(houses, households, persons)
#        viz_persons=[persons[mp] for mp in moved_persons]
#        get_LLs(viz_persons)
#        get_routes(viz_persons)
#        predict_modes(viz_persons)
#        create_trips(viz_persons)
#        post_trips_data(viz_persons, CITYIO_OUTPUT_PATH+'trips')
#        post_grid_geojson(grid_geo, CITYIO_OUTPUT_PATH+'site_geojson')
#        for m in range(4):
#            print(100*sum([1 for p in viz_persons if p['mode']==m])/len(viz_persons))
#        print(np.mean([p['kgCO2']for p in viz_persons]))
#    sleep(0.2)
##    print('Done sleeping')
        

    


