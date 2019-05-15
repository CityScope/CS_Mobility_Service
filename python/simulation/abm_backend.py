#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:32:34 2019

@author: doorleyr
"""
import random
import requests
import json
import pyproj
import time
import pandas as pd
import pickle
import numpy as np
import math
import networkx as nx
from scipy import spatial
import urllib
from Agents import Person, Location, House


def createGrid(topLeft_lonLat, topEdge_lonLat, utm, wgs, cell_size, nrows, ncols):
    #retuns the top left coordinate of each grid cell from left to right, top to bottom
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
    G=nx.DiGraph()
    G.add_nodes_from(range(len(grid_coords_ll)))
    for c in range(ncols):
        for r in range(nrows):
            # if not at the end of a row, add h link
            if not c==ncols-1:
                G.add_edge(r*ncols+c, r*ncols+c+1, weight=cell_size)
                G.add_edge(r*ncols+c+1, r*ncols+c, weight=cell_size)
            # if not at the end of a column, add v link
            if not r==nrows-1:
                G.add_edge(r*ncols+c, (r+1)*ncols+c, weight=cell_size)
                G.add_edge((r+1)*ncols+c, r*ncols+c, weight=cell_size)
    return grid_coords_ll,  G
    

def update_and_send_points():
    global ts
    features=[]
    for ag in agents:
        if ((not ag.finished) and ts>ag.start_time):
            ag.update_position(TIMESTEP_SEC)
        else:
            ag.position=[ag.position[0]+np.random.normal(0,0.000001), ag.position[1]+np.random.normal(0,0.000001)]
#            ll=pyproj.transform( utm, wgs, ag.position[0], ag.position[1])
#            ll=[int(ll[0]*1e5)/1e5, int(ll[1]*1e5)/1e5] # reduce precision for sending data
        if (ag.position[0]>region_lon_bounds[0] and ag.position[0]<region_lon_bounds[1] and 
            ag.position[1]>region_lat_bounds[0] and ag.position[1]<region_lat_bounds[1]):
            geometry={"type": "Point",
                     "coordinates": [int(ag.position[0]*1e5)/1e5, int(ag.position[1]*1e5)/1e5]
                    }
            feature={"type": "Feature",
                     "geometry":geometry,
                     'properties':
                             {'mode':ag.mode}
                     }
            features.append(feature)        
    geojson_object={
      "type": "FeatureCollection",
      "features": features    
    }        
#    cityio_json['objects']={"points": geojson_object}    
    try:
        r = requests.post(sim_api_root+sim_api_end, data = json.dumps(geojson_object))
    except:
        print('Couldnt send to cityio')
        time.sleep(5)
        r='Failed to get response'
    ts+=1
    time.sleep(0.05)
    print(r)
    
def send_routes():
    features=[]
    for ag in agents:
        geometry={"type": "Point",
                 "coordinates": [ag.position[0], ag.position[1]]
                }
        feature={"type": "Feature",
                 "geometry":geometry,
                 'properties':{'mode':ag.mode, 'age':ag.age, 'hh_income':ag.hh_income, 'id': ag.person_id,
                               'route': ag.route, 'speed': ag.speed, 'position': ag.position, 
                               'next_node_index': ag.next_node_index, 'next_node_ll': ag.next_node_ll, 
                               'prop_of_link_left': ag.prop_of_link_left ,'finished': ag.finished}
                 }
        features.append(feature)        
    geojson_object={
      "type": "FeatureCollection",
      "features": features    
    }   
    r = requests.post(sim_api_root+sim_api_end, data = json.dumps(geojson_object))
    print(r)
        
#def check_grid_data(p):
#    global agents, base_agents, new_agents, lastId
#    with urllib.request.urlopen(cityIO_grid_url) as url:
#    #get the latest json data
#        cityIO_grid_data=json.loads(url.read().decode())
#    hash_id=cityIO_grid_data['meta']['id']
#    if hash_id==lastId:
#        pass
#    else:
#        print('Update agents')
#        lu={}
#        lu['RM']=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==1]
#        lu['RL']=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==2]
#        lu['OM']=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==3]
#        lu['OL']=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==4]        
##        lu['live_1']=[1,3,5,7]
##        lu['live_2']=[14, 15, 16]
##        lu['work_1']=[246, 247, 248, 249]
##        lu['work_2']=[241, 242, 243, 244, 245]
#        for lu_type in lu:
#            lu[lu_type]*=PERSONS_PER_BLD
#            random.shuffle(lu[lu_type])
#        new_agents=[]
#        # TODO: use pot_pop and discrete choice for location choices.
#        for level in ['L', 'M']:# for each type of housing (assuming people working in Work_1 live in Live_1)
#            n_residents, n_workers=len(lu['R'+level]), len(lu['O'+level])
#            for i in range(min(n_residents, n_workers)):
#                new_agents.append(Person(25, True, 5, grid_locations[lu['R'+level][i]], 
#                             grid_locations[lu['O'+level][i]], True, 'HWH', 8000, len(agents), routes, node_coords))
#            if n_residents>n_workers: # more res than off for this type
#                # add the new agents with outside work locs
#                for i in range(n_workers, n_residents):
#                    work_zone=random.choice(range(len(zone_locations))) 
#                    new_agents.append(Person(25, True, 5, grid_locations[lu['R'+level][i]], 
#                             zone_locations[work_zone], True, 'HWH', 8000, len(agents), routes, node_coords))
#            else:
#                # add the new agents with outside home locs
#                for i in range(n_residents, n_workers):
#                    home_zone=random.choice(range(len(zone_locations))) 
#                    new_agents.append(Person(25, True, 5, zone_locations[home_zone], 
#                             grid_locations[lu['O'+level][i]], True, 'HWH', 8000, len(agents), routes, node_coords))
#        agents=base_agents+new_agents
#        for ag in new_agents: ag.init_period(period, TIMESTEP_SEC)
#        predict_modes(new_agents)
#    lastId=hash_id
    
    
def check_grid_data(p):
    global agents, base_agents, new_agents, lastId, base_housing, new_housing, housing
    with urllib.request.urlopen(cityIO_grid_url) as url:
    #get the latest json data
        cityIO_grid_data=json.loads(url.read().decode())
    hash_id=cityIO_grid_data['meta']['id']
    if hash_id==lastId:
        pass
    else:
        #create new houses
        new_housing=[]
        for ht in housing_types:
            ht_locs=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==ht]
            for htl in ht_locs:
                new_housing.append(House(housing_types[ht]['rent'], 150000, 60000, grid_locations[htl], len(base_housing)+len(new_housing)))
        housing=base_housing+new_housing
        #create new persons
        new_agents=[]
        for et in employment_types:
            et_locs=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==et]
            for etl in et_locs:
                new_agents.append(Person(25, True, 5, None, grid_locations[etl], True, 
                 'HWH', 8000, len(base_agents)+len(new_agents), routes, node_coords))
        agents=base_agents+new_agents
        #each new person chooses a house
        long_data=[]
        for ag in new_agents:
            #choose N houses
            h_alts=random.sample(housing, 6)
            for hi, h in enumerate(h_alts):
                 long_data.append(h.long_data_record(10000*ag.hh_income, ag.person_id, hi+1))
        long_df=pd.DataFrame(long_data)
        long_df['predictions']=home_loc_logit.predict(long_df)
        for ag_ind in set(long_df['custom_id']):
            # find maximum prob or sample from probs in subset of long_df
            house_id=long_df.loc[long_df[long_df['custom_id']==ag_ind]['predictions'].idxmax(), 'actual_house_id']
            agents[ag_ind].home_loc=housing[house_id].location                
        #each new person chooses a mode
        for ag in new_agents:
            ag.init_routes(routes, node_coords)
            ag.init_period(period, TIMESTEP_SEC)
        predict_modes(new_agents)
        
    lastId=hash_id
    
def predict_modes(agent_list):
    # instead of using get attribute one at a time: create a class method to return the dict
    feature_df=pd.DataFrame([{f:getattr(a, f) for f in ['age', 'hh_income','male',
             'bachelor_degree', 'pop_per_sqmile_home',
             'network_dist_km']} for a in agent_list])    
    feature_df['drive_time_minutes']=  60*feature_df['network_dist_km']/speeds[0]     
    feature_df['cycle_time_minutes']=  60*feature_df['network_dist_km']/speeds[1]     
    feature_df['walk_time_minutes']=  60*feature_df['network_dist_km']/speeds[2]     
    feature_df['PT_time_minutes']=  60*feature_df['network_dist_km']/speeds[3]  
    feature_df['walk_time_PT_minutes']=5  
    feature_df['drive_time_PT_minutes']=5  
    feature_df['drive_cost']=0
    feature_df['cycle_cost']=0
    feature_df['walk_cost']=0
    feature_df['PT_cost']=0
    assert all([rff in feature_df.columns for rff in rf_features]),"Features in table dont match features in RF model"
    feature_df=feature_df[rf_features]#reorder columns to match rf model
    mode_probs=mode_rf.predict_proba(feature_df)
    for i,ag in enumerate(agent_list): ag.set_mode(int(np.random.choice(range(4), size=1, replace=False, p=mode_probs[i])[0]), speeds)

    
# =============================================================================
# Constants
# =============================================================================
city='Hamburg'

# bounding box for sending points
region_lon_bounds=[9.923536, 10.052368]
region_lat_bounds=[53.491466, 53.56]
#region_lon_bounds=[-90, 90]
#region_lat_bounds=[-90, 90]

#CITYIO_TEMPLATE_PATH='../'+city+'/clean/cityio_template.json'
SYNTHPOP_PATH='../'+city+'/clean/synth_pop.csv'
PICKLED_MODEL_PATH='../models/trip_mode_rf.p'
ZONE_NODE_ROUTES_PATH='../'+city+'/clean/route_nodes.json'
CONNECTION_NODE_ROUTES_PATH='../'+city+'/clean/connection_route_nodes.json'
NODES_PATH='../'+city+'/clean/nodes.csv'
CONNECTION_POINTS_PATH='../'+city+'/clean/connection_points.json'
RF_FEATURES_LIST_PATH='../models/rf_features.json'
FITTED_HOME_LOC_MODEL_PATH='../models/home_loc_logit.p'

PERSONS_PER_BLD=2
BASE_AGENTS=100
VACANT_HOUSES=1

SENDING_POINTS=True

speeds={0:40,
        1:15,
        2:5,
        3:25}

# TODO: put the below in a text file
# TODO: number of of people per housing cell should vary by type
housing_types={1:{'rent': 1000},
               2:{'rent': 2000} }
# TODO: number of each employment sector for each building type
employment_types= {3:{},4:{}}

UTM_MAP={'Boston':pyproj.Proj("+init=EPSG:32619"),
     'Hamburg':pyproj.Proj("+init=EPSG:32632")}
utm=UTM_MAP[city]
wgs=pyproj.Proj("+init=EPSG:4326")


# gFor getting grid data
cityIO_grid_url_map={'Boston':"mocho",
     'Hamburg':"grasbrook"}
lastId=0
host='https://cityio.media.mit.edu/'
#host='http://localhost:8080/' # local port running cityio
cityIO_grid_url=host+'api/table/'+cityIO_grid_url_map[city]

# For sending simulation data
TIMESTEP_SEC=30
sim_api_root='https://cityio.media.mit.edu/api/table/update/abm_service_'
sim_api_end='test'
#sim_api_end=city

# load the pre-calibrated mode choice model
mode_rf=pickle.load( open( PICKLED_MODEL_PATH, "rb" ) )
rf_features=json.load(open(RF_FEATURES_LIST_PATH, 'r'))
home_loc_logit=pickle.load( open( FITTED_HOME_LOC_MODEL_PATH, "rb" ) )

# load the pre-calibrated home location choice model

# load the connection points between real network and grid network
connection_points=json.load(open(CONNECTION_POINTS_PATH))
# load the routes from each zone to each connection
zone_connection_routes=json.load(open(CONNECTION_NODE_ROUTES_PATH))
# get the routes between each exisitng zone
zone_routes=json.load(open(ZONE_NODE_ROUTES_PATH))

nodes=pd.read_csv(NODES_PATH)

original_net_node_coords=nodes[['lon', 'lat']].values.tolist()

# create land use grid and road network from the grid

# =============================================================================
# Get the grid data and set up the interaction zone 
# =============================================================================
with urllib.request.urlopen(cityIO_grid_url) as url:
#get the latest json data
    cityIO_grid_data=json.loads(url.read().decode())
topLeft_lonLat={'lat':53.533681, 'lon':10.011585}
topEdge_lonLat={'lat':53.533433, 'lon':10.012213}
#topLeft_lonLat={'lat':cityIO_grid_data['header']['spatial']['latitude'], 
#                'lon':cityIO_grid_data['header']['spatial']['longitude']}

cell_size= cityIO_grid_data['header']['spatial']['cellSize']
nrows=cityIO_grid_data['header']['spatial']['nrows']
ncols=cityIO_grid_data['header']['spatial']['ncols']
grid_points_ll, net=createGrid(topLeft_lonLat, topEdge_lonLat, utm, wgs, cell_size, nrows, ncols)

# =============================================================================
# Get the routes within each sub-net and from each sub-net to the connection points 
# =============================================================================
# for each connection point, find the closest grid node
grid_tree = spatial.KDTree(np.array(grid_points_ll))
for cp in connection_points:
    cp['closest_grid_node']=grid_tree.query([cp['lon'], cp['lat']])[1]
# get all the internal routes and all the connection routes
grid_node_paths =  dict(nx.all_pairs_shortest_path(net))
grid_routes={str(i):{} for i in range(len(grid_points_ll))}
for o in range(len(grid_points_ll)):
    for d in range(len(grid_points_ll)):
        # TODO: tuple
        grid_routes[str(o)][str(d)]={'nodes':grid_node_paths[o][d],
                     'distances': [cell_size for l in range(len(grid_node_paths[o][d])-1)]}
grid_connection_routes=[{'to': [], 'from':[]} for g in range(len(grid_points_ll))]
for n in range(len(grid_points_ll)):
    for cp in range(len(connection_points)):
        # TODO: should already be tuples
        grid_connection_routes[n]['to'].append(grid_routes[str(n)][str(connection_points[cp]['closest_grid_node'])])
        grid_connection_routes[n]['from'].append(grid_routes[str(connection_points[cp]['closest_grid_node'])][str(n)])

routes= [zone_routes, grid_routes]
connection_routes= [zone_connection_routes, grid_connection_routes]
node_coords= [original_net_node_coords, grid_points_ll]
# =============================================================================

# for each zone, create a location object
zone_locations=[]
for z in range(len(zone_routes)):
    zone_locations.append(Location(0, z, connection_routes[0][z], 0.01))
    
grid_locations=[]
for n in range(len(grid_routes)):
    grid_locations.append(Location(1, n, connection_routes[1][n], 0))


# for each person in base pop, create Person
random.seed(0)
synth_pop=pd.read_csv(SYNTHPOP_PATH)
ag_ind=0
base_agents=[]
for ag_ind, row in synth_pop[:BASE_AGENTS].iterrows():
    base_agents.append(Person(row['age'], row['bachelor_degree'], row['hh_income'], zone_locations[row['home_geo_index']], 
                         zone_locations[row['work_geo_index']], row['male'], row['motif'], row['pop_per_sqmile_home'], len(base_agents), routes, node_coords))

agents=base_agents+[]
# create new vacant housing (representing turnover of rental market and newly built housing)
base_housing=[]
#TODO specify attributes of housing from the data
for i in range(VACANT_HOUSES):
    rent=random.choice([1000,1500,2000])
    puma_med_income=random.choice([30000, 60000, 100000])
    puma_pop=random.choice([100000, 150000, 200000])
    base_housing.append(House(rent, puma_pop, puma_med_income, random.choice(zone_locations), len(base_housing)))
    
housing=base_housing+[]
    
# =============================================================================
# Simulation Loop
# =============================================================================   
period=0
for ag in agents: 
    ag.init_routes(routes, node_coords)
    ag.init_period(period, TIMESTEP_SEC)
predict_modes(agents)
# TODO: separate function to predict modes
#predict modes of all agents
#do all at one in a pandas df because its faster than 


prop=0
ts=1

if SENDING_POINTS:
    while True:
        if ts%100==1:
            try: check_grid_data(period)
            except: print('Problem updating from grid')
        if prop>0.5:
            period+=1
            ts=0
            for ag in agents: 
                ag.init_routes(routes, node_coords)
                ag.init_period(period, TIMESTEP_SEC)
            predict_modes(agents)
        update_and_send_points()
        prop=sum([ag.finished for ag in agents])/len(agents)
        print(ts)
else:
    try: check_grid_data(period)
    except: print('Problem updating from grid')
    send_routes()

            
