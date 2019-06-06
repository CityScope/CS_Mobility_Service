#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:50:11 2018

@author: doorleyr
"""

#!flask/bin/python
from flask import Flask, jsonify, make_response
import threading
import atexit
import pickle
import json
import random
import urllib
import pyproj
import math
import pandas as pd
from flask_cors import CORS
import numpy as np
import networkx as nx
from scipy import spatial
from Agents import Person, Location, House, Household


# =============================================================================
# Functions
# =============================================================================
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

def predict_modes(agent_list):
    # instead of using get attribute one at a time: create a class method to return the dict
    feature_df=pd.DataFrame([{f:getattr(a, f) for f in ['pop_per_sqmile_home',
             'network_dist_km']} for a in agent_list])  
    feature_df['bach_degree_yes']=[a.school_level >=4 for a in agent_list]
    feature_df['bach_degree_no']=~feature_df['bach_degree_yes']
    for feat in ['income', 'age', 'children', 'workers', 'tenure', 'sex']:
        new_dummys=pd.get_dummies([getattr(a, feat) for a in agent_list], prefix=feat)
        feature_df=pd.concat([feature_df, new_dummys],  axis=1)
    feature_df['drive_time_minutes']=  60*feature_df['network_dist_km']/speeds[0]     
    feature_df['cycle_time_minutes']=  60*feature_df['network_dist_km']/speeds[1]     
    feature_df['walk_time_minutes']=  60*feature_df['network_dist_km']/speeds[2]     
    feature_df['PT_time_minutes']=  60*feature_df['network_dist_km']/speeds[3]  
    feature_df['walk_time_PT_minutes']=5  
    feature_df['drive_time_PT_minutes']=5  
    feature_df['tenure_owned']=False
    feature_df['tenure_other']=False

    assert all([rff in feature_df.columns for rff in rf_features]),"Features in table dont match features in RF model"
    feature_df=feature_df[rf_features]#reorder columns to match rf model
    mode_probs=mode_rf.predict_proba(feature_df)
    for i,ag in enumerate(agent_list): ag.set_mode(int(np.random.choice(range(4), size=1, replace=False, p=mode_probs[i])[0]), speeds)

def create_geojson(persons):
    features=[]
    for per in persons:
        if (houses[households[per.household_id].house_id].location.graph_id==1 or per.work_loc.graph_id==1):
            geometry={"type": "Point",
                     "coordinates": [per.position[0], per.position[1]]
                    }
            feature={"type": "Feature",
                     "geometry":geometry,
                     'properties':{'mode':per.mode, 'id': per.person_id,
                                   'route': per.route, 'speed': per.speed, 'position': per.position, 
                                   'next_node_index': per.next_node_index, 'next_node_ll': per.next_node_ll, 
                                   'prop_of_link_left': per.prop_of_link_left ,'finished': per.finished}
                     }
            features.append(feature)        
    geojson_object={
      "type": "FeatureCollection",
      "features": features    
    }
    return geojson_object

def get_trip_data(persons):
    trips=[]
    for per in persons:
        if (houses[households[per.household_id].house_id].location.graph_id==1 or per.work_loc.graph_id==1): 
            segments=[[int(1e5*per.route['coordinates'][n][0])/1e5, int(1e5*per.route['coordinates'][n][1])/1e5, per.route['timestamps'][n]] for n in range(len(per.route['timestamps']))]
            trips.append({'mode': per.mode, 'segments': segments})
    return trips
# =============================================================================
# Constants
# =============================================================================
city='Hamburg'

# bounding box for sending points
region_lon_bounds=[9.923536, 10.052368]
region_lat_bounds=[53.491466, 53.56]
#region_lon_bounds=[-90, 90]
#region_lat_bounds=[-90, 90]

#SYNTHPOP_PATH='../'+city+'/clean/synth_pop.csv'
SYNTH_HH_PATH='../'+city+'/clean/synth_households.csv'
SYNTH_PERSONS_PATH='../'+city+'/clean/synth_persons.csv'
PICKLED_MODEL_PATH='../models/trip_mode_rf.p'
ZONE_NODE_ROUTES_PATH='../'+city+'/clean/route_nodes.json'
CONNECTION_NODE_ROUTES_PATH='../'+city+'/clean/connection_route_nodes.json'
NODES_PATH='../'+city+'/clean/nodes.csv'
CONNECTION_POINTS_PATH='../'+city+'/clean/connection_points.json'
RF_FEATURES_LIST_PATH='../models/rf_features.json'
FITTED_HOME_LOC_MODEL_PATH='../models/home_loc_logit.p'
CITYIO_SAMPLE_PATH='../'+city+'/clean/sample_cityio_data.json'
RENT_NORM_PATH='../models/rent_norm.json'

NUM_HOUSEHOLDS=1000
NUM_MOVERS=100

POOL_TIME=1 # seconds
TIMESTEP_SEC=1

speeds={0:40,
        1:15,
        2:5,
        3:25}

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
#cityIO_grid_url=host+'api/table/'+'virtual_table'

# =============================================================================
# Load Data
# =============================================================================
# load the pre-calibrated mode choice model
mode_rf=pickle.load( open( PICKLED_MODEL_PATH, "rb" ) )
rf_features=json.load(open(RF_FEATURES_LIST_PATH, 'r'))
# load the pre-calibrated home location choice model
home_loc_logit=pickle.load( open( FITTED_HOME_LOC_MODEL_PATH, "rb" ) )
rent_normalisation=json.load(open(RENT_NORM_PATH))

# load the synthestic pops of households and persons
synth_hh_df=pd.read_csv(SYNTH_HH_PATH)
# sample ranted households only
# TODO: separate models for rented and owned housing
synth_hh_df=synth_hh_df.loc[~synth_hh_df['RNTP'].isnull()].sample(n=NUM_HOUSEHOLDS)
synth_persons_df=pd.read_csv(SYNTH_PERSONS_PATH).sample(n=10000)

# load the connection points between real network and grid network
connection_points=json.load(open(CONNECTION_POINTS_PATH))
# load the routes from each zone to each connection
zone_connection_routes=json.load(open(CONNECTION_NODE_ROUTES_PATH))
# get the routes between each exisitng zone
zone_routes=json.load(open(ZONE_NODE_ROUTES_PATH))

nodes=pd.read_csv(NODES_PATH)

original_net_node_coords=nodes[['lon', 'lat']].values.tolist()
# =============================================================================
# Preliminary Processing
# =============================================================================
try:
    with urllib.request.urlopen(cityIO_grid_url) as url:
    #get the latest json data
    #    print('Getting initial grid data')
        cityIO_grid_data=json.loads(url.read().decode())
except:
    print('Using static cityIO grid file')
    cityIO_grid_data=json.load(open(CITYIO_SAMPLE_PATH))
    
topLeft_lonLat={'lat':53.533681, 'lon':10.011585}
topEdge_lonLat={'lat':53.533433, 'lon':10.012213}
#topLeft_lonLat={'lat':cityIO_grid_data['header']['spatial']['latitude'], 
#                'lon':cityIO_grid_data['header']['spatial']['longitude']}

cell_size= cityIO_grid_data['header']['spatial']['cellSize']
nrows=cityIO_grid_data['header']['spatial']['nrows']
ncols=cityIO_grid_data['header']['spatial']['ncols']
grid_points_ll, net=createGrid(topLeft_lonLat, topEdge_lonLat, utm, wgs, cell_size, nrows, ncols)

# TODO: put the below in a text file
# TODO: number of of people per housing cell should vary by type
housing_types={1:{'rent': 0.8*rent_normalisation['mean']['2'], 'beds': 2},
               2:{'rent': rent_normalisation['mean']['2'], 'beds': 2 }}
# TODO: number of each employment sector for each building type
employment_types= {3:{},4:{}}

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
        grid_connection_routes[n]['to'].append(grid_routes[str(n)][str(connection_points[cp]['closest_grid_node'])])
        grid_connection_routes[n]['from'].append(grid_routes[str(connection_points[cp]['closest_grid_node'])][str(n)])

routes= [zone_routes, grid_routes]
connection_routes= [zone_connection_routes, grid_connection_routes]
node_coords= [original_net_node_coords, grid_points_ll]


zone_locations=[]
for z in range(len(zone_routes)):
    first_node=connection_routes[0][z]['to'][0]['nodes'][0]
    zone_locations.append(Location(0, z , connection_routes[0][z], node_coords[0][first_node], len(zone_locations)))
    
grid_locations=[]
for n in range(len(grid_routes)):
    grid_locations.append(Location(1, n, connection_routes[1][n], node_coords[1][n], len(zone_locations)+len(grid_locations)))

locations=zone_locations+grid_locations
# =============================================================================
# Create the HH, person and housing agents 
# =============================================================================

base_persons, base_households, base_houses=[],[],[]
# for each row in hh pop, create household and a house
# TODO: add the PUMA attributes
# TODO: only rented houses
for ind, row in synth_hh_df.iterrows():
    house_id=len(base_houses)
    household_id=len(base_households)
    # TODO zone locations should correspong to the actual zones
    # wont work for Hamburg but will for USA
    base_households.append(Household(row['VEH'], row['workers'], row['children'], 
                           row['HINCP'], row['income'], row['tenure'], house_id, household_id))
    base_houses.append(House( row['RNTP'], 0.000292, 50000, row['BDSP'], row['YBL'], random.choice(zone_locations), 
                       house_id, household_id))
# for each hh , spawn num_workers people
for hh in base_households:
    # TODO: use the num people in the household
    for i in range(2):
        base_persons.append(hh.spawn_person(synth_persons_df, 
                                            len(base_persons), routes, node_coords,random.choice(zone_locations)))
persons, households, houses=base_persons+[], base_households+[], base_houses+[] 

for p in persons:p.init_routes(routes, node_coords, houses[households[p.household_id].house_id].location)

# random select ids of hoseholds to enter housing market in each experiment
drifters=random.sample(range(len(households)), NUM_MOVERS)

# =============================================================================
# Create the Flask app
# =============================================================================

dataLock = threading.Lock()
# thread handler
yourThread = threading.Thread()

def create_app():
    app = Flask(__name__)

    def interrupt():
        global yourThread
        yourThread.cancel()

    def background():
        global lastId
        global persons, base_persons, new_persons  
        global houses, base_houses, new_houses
        global households, base_households, new_households
        with dataLock:
#            print('Getting grid update')
            try:
                with urllib.request.urlopen(cityIO_grid_url) as url:
                    cityIO_grid_data=json.loads(url.read().decode())
            except:
                print('Using static cityIO grid file')
                cityIO_grid_data=json.load(open(CITYIO_SAMPLE_PATH))
            hash_id=cityIO_grid_data['meta']['id']
            if hash_id==lastId:
                pass
            else:
#                print('Updating agents')
                # free floating people become homeless, their homes become vacant
                for d in drifters:
                    houses[households[d].house_id].household_id=None
                    households[d].house_id=None
                #create new houses
                new_houses=[]
                new_persons=[]
                new_households=[]
                
                for ht in housing_types:
                    ht_locs=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==ht]
                    for htl in ht_locs:
                        new_houses.append(House(housing_types[ht]['rent'], 
                                                 0.000292, 60000, housing_types[ht]['beds'],
                                                 15, grid_locations[htl], 
                                                 len(base_houses)+len(new_houses), None))
                
                # for each office unit, add a (homeless) household to new_hhs
                for et in employment_types:
#                    et_locs=list(range(50))
                    et_locs=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==et]
                    for etl in et_locs:
                        sample_hh_row=synth_hh_df.sample(n=1).squeeze()
                        new_hh=Household(sample_hh_row['VEH'], sample_hh_row['workers'], sample_hh_row['children'], 
                           sample_hh_row['HINCP'], sample_hh_row['income'], sample_hh_row['tenure'], 
                           None, len(base_households)+len(new_households))
                        new_households.append(new_hh)
                        # TODO: use the num people in the household
                        for i in range(2):
                            new_persons.append(new_hh.spawn_person(synth_persons_df, 
                                                                len(base_persons)+len(new_persons), routes, node_coords, grid_locations[etl]))
                # for each homeless hh, choose a house
                houses=base_houses+new_houses
                persons=base_persons+new_persons
                households=base_households+new_households
                
                #each homeless household chooses a house
                vacant_housing = [h for h in houses if h.household_id==None]
                homeless_hhs= [hh for hh in households if hh.house_id==None]
                long_data=[]
                for hh in homeless_hhs:
                    #choose N houses
                    h_alts=random.sample(vacant_housing, 9)
                    for hi, h in enumerate(h_alts):
                         long_data.append(h.long_data_record(hh.hh_income, hh.household_id, hi+1, rent_normalisation))
                long_df=pd.DataFrame(long_data)
                long_df['predictions']=home_loc_logit.predict(long_df)
                for hh_ind in set(long_df['custom_id']):
                    # find maximum prob or sample from probs in subset of long_df
                    house_id=np.random.choice(long_df.loc[long_df['custom_id']==hh_ind, 'actual_house_id'], p=long_df.loc[long_df['custom_id']==hh_ind, 'predictions'])
                    house_id=int(long_df.loc[long_df[long_df['custom_id']==hh_ind]['predictions'].idxmax(), 'actual_house_id'])
                    households[hh_ind].house_id=house_id 
                    houses[house_id].household_id=hh_ind

                for p in persons:
                    p.set_home_loc(houses[households[p.household_id].house_id].location)
                    p.init_routes(routes, node_coords, houses[households[p.household_id].house_id].location)      
            lastId=hash_id
        yourThread = threading.Timer(POOL_TIME, background, args=())
        yourThread.start()        

    def initialise():
        # Create the initial background thread
        yourThread = threading.Timer(POOL_TIME, background, args=())
        yourThread.start()
    # Initiate
    initialise()
    # When you kill Flask (SIGTERM), clear the trigger for the next thread
    atexit.register(interrupt)
    return app

app = create_app()
CORS(app)

@app.route('/abm_service/Hamburg/routes/<int:period>', methods=['GET'])
def return_routes(period):
#    print('Data requested')
    for p in persons: p.init_period(period, TIMESTEP_SEC)
    predict_modes(persons)
#    person_data= {p.person_id: 
#        {'mode':p.mode,
##         'route':[[int(c[0]*1e5)/1e5,int(c[1]*1e5)/1e5]  for c in ag.route['coordinates']],
#         'route':p.route['coordinates'],
#         'distances':p.route['distances']
#        } for p in persons}
    return json.dumps(create_geojson(persons))

@app.route('/abm_service/Hamburg/trips/<int:period>', methods=['GET'])
def return_trips(period):
    for p in persons: p.init_period(period, TIMESTEP_SEC)
    predict_modes(persons)
    return json.dumps(get_trip_data(persons))  

@app.route('/abm_service/Hamburg/arcs/<int:period>', methods=['GET'])
def return_arcs(period):
    for p in persons: p.init_period(period, TIMESTEP_SEC)
    predict_modes(persons)
    arcs=[ {'centroid': locations[d].centroid,'flows':{o: {0:0, 1:0, 2:0, 3:0} for o in range(len(zone_locations)+len(grid_locations))}}
        for d in range(len(zone_locations)+len(grid_locations))]
    for p in persons:
        arcs[p.work_loc.zone_id]['flows'][houses[households[p.household_id].house_id].location.zone_id][p.mode]+=1
    return json.dumps(arcs)


@app.errorhandler(404)
# standard error is html message- we need to ensure that the response is always json
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(port=7777, debug=False, use_reloader=False, threaded=True)
    # if reloader is True, it starts the background thread twice

