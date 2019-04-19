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
import datetime
import pandas as pd
import pickle
import numpy as np
import math
import networkx as nx
from scipy import spatial
import threading
import urllib
import atexit


class Location:
    def __init__(self, graph_id, term_node, cp_routes):
        self.graph_id=graph_id
        self.term_node=term_node
        self.cp_routes=cp_routes
        
class Person:
    def __init__(self,age, bachelor_degree, hh_income, home_loc, work_loc, male, motif, pop_per_sqmile_home, id):
        self.id=id
        self.age=age
        self.bachelor_degree=bachelor_degree
        self.hh_income=hh_income
        self.home_loc=home_loc
        self.work_loc=work_loc
        self.male=male
        self.motif=motif
        self.pop_per_sqmile_home=pop_per_sqmile_home
        if home_loc.graph_id==work_loc.graph_id:            
            self.all_routes=[routes[home_loc.graph_id][str(home_loc.term_node)][str(work_loc.term_node)].copy(),
                             routes[home_loc.graph_id][str(work_loc.term_node)][str(home_loc.term_node)].copy()]
            for r in self.all_routes:
                r['coordinates']=[node_coords[home_loc.graph_id][n].copy() for n in r['nodes']]
        else:
            self.all_routes=[{'nodes': home_loc.cp_routes['to'][0]['nodes']+work_loc.cp_routes['from'][0]['nodes'].copy(),
                            'distances':home_loc.cp_routes['to'][0]['distances']+[1]+work_loc.cp_routes['from'][0]['distances'].copy(),
                            'coordinates': [node_coords[home_loc.graph_id][n].copy() for n in home_loc.cp_routes['to'][0]['nodes']]+
                            [node_coords[work_loc.graph_id][n].copy() for n in work_loc.cp_routes['from'][0]['nodes']]},
                            {'nodes': work_loc.cp_routes['to'][0]['nodes']+home_loc.cp_routes['from'][0]['nodes'].copy(),
                            'distances':work_loc.cp_routes['to'][0]['distances']+[1]+home_loc.cp_routes['from'][0]['distances'].copy(),
                            'coordinates': [node_coords[work_loc.graph_id][n].copy() for n in work_loc.cp_routes['to'][0]['nodes']]+
                            [node_coords[home_loc.graph_id][n].copy() for n in home_loc.cp_routes['from'][0]['nodes']]}]
        
            
    def init_period(self, p):
        self.route=self.all_routes[p%len(self.all_routes)]
#        get the travel time and cost for each mode
#        TODO: get travel times of each mode beforehand
        route_distance=sum(self.route['distances'])     
        # all times should be in minutes
        [drive_time, cycle_time, walk_time, PT_time]=[(route_distance/speeds[i])*(1000/60) for i in range(4)]
        walk_time_PT, drive_time_PT=600, 600 # minutes
        drive_cost, cycle_cost, walk_cost, PT_cost=0,0,0,0
        self.mode=int(mode_rf.predict(np.array([drive_time, cycle_time, walk_time, PT_time, 
                                   walk_time_PT, drive_time_PT,
                                   drive_cost, cycle_cost, walk_cost, PT_cost,
                                   self.age, self.hh_income, self.male, 
                                   self.bachelor_degree , self.pop_per_sqmile_home]).reshape(1,-1))[0])
        speed_mode=speeds[self.mode] 
        self.speed=random.triangular(0.7*speed_mode, 1.3*speed_mode, speed_mode)
        self.position=self.route['coordinates'][0].copy()
        self.next_node_index=1
        if len(self.route['coordinates'])>1: 
            self.next_node_ll=self.route['coordinates'][1].copy()
            self.finished=False
            self.prop_of_link_left=1
        else: 
            self.next_node_ll=self.route['coordinates'][0].copy()
            self.finished=True
            self.prop_of_link_left=0        
    def update_position(self, seconds):
        # update an agent's position along a predefined route based on their 
        # speed and the time elapsed
        dist_to_move_m=self.speed*seconds/3.6
        finished_move=False
        while finished_move==False and self.finished==False:
            d_to_next_node=self.prop_of_link_left*self.route['distances'][self.next_node_index-1]
            move_ratio=dist_to_move_m/d_to_next_node
            if move_ratio<1:
                # just move the agent along this segment. move finished.
                self.position[0]=self.position[0]+move_ratio*(self.next_node_ll[0]-self.position[0])
                self.position[1]=self.position[1]+move_ratio*(self.next_node_ll[1]-self.position[1])
                self.prop_of_link_left=self.prop_of_link_left*(1-move_ratio)
                finished_move=True
            else:
                #agent moves to start of next segment and then continues the move
                self.position[0]=self.next_node_ll[0]
                self.position[1]=self.next_node_ll[1]
                self.next_node_index+=1
                if self.next_node_index==len(self.route['coordinates']):
                    self.finished=True
                else:
                    self.next_node_ll=self.route['coordinates'][self.next_node_index]
                    self.prop_of_link_left=1
                    dist_to_move_m-=d_to_next_node 
                    

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
    

def update_and_send():
    features=[]
    for ag in agents:
        if not ag.finished:
            ag.update_position(TIMESTEP_SEC)
        else:
            ag.position=[ag.position[0]+np.random.normal(0,0.00002), ag.position[1]+np.random.normal(0,0.00002)]
#            ll=pyproj.transform( utm, wgs, ag.position[0], ag.position[1])
#            ll=[int(ll[0]*1e5)/1e5, int(ll[1]*1e5)/1e5] # reduce precision for sending data
        geometry={"type": "Point",
                 "coordinates": [int(ag.position[0]*1e5)/1e5, int(ag.position[1]*1e5)/1e5]
                }
        feature={"type": "Feature",
                 "geometry":geometry,
                 'properties':
#                         {'mode':ag.mode, 'age':ag.age, 'hh_income':ag.hh_income, 'id': ag.id}
                     {}
                 }
        features.append(feature)        
    geojson_object={
      "type": "FeatureCollection",
      "features": features    
    }        
    cityio_json['objects']={"points": geojson_object}    
    r = requests.post('https://cityio.media.mit.edu/api/table/update/abm_service_'+city, data = json.dumps(geojson_object))
    time.sleep(0.3)
    print(r)
        
def check_grid_data(p):
    global agents, base_agents, new_agents, lastId
    with urllib.request.urlopen(cityIO_grid_url) as url:
    #get the latest json data
        cityIO_grid_data=json.loads(url.read().decode())
    hash_id=cityIO_grid_data['meta']['id']
    if hash_id==lastId:
        pass
    else:
        print('Update agents')
        live_1=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==0]
        live_2=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==1]
        work_1=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==2]
        work_2=[i for i in range(len(cityIO_grid_data['grid'])) if cityIO_grid_data['grid'][i][0]==3]
#        live_1= [1,2,3,4]
#        work_1= [55,56,57,58]
        random.shuffle(live_1)
        random.shuffle(work_1)
        new_agents=[]
        for i in range(min(len(live_1), len(work_1))):
            new_agents.append(Person(25, True, 5, grid_locations[live_1[i]], 
                         grid_locations[work_1[i]], True, 'HWH', 8000, len(agents))) 
        agents=base_agents+new_agents
        for ag in new_agents: ag.init_period(period)
    lastId=hash_id
# =============================================================================
# Constants
# =============================================================================
city='Hamburg'

CITYIO_TEMPLATE_PATH='./'+city+'/clean/cityio_template.json'
SYNTHPOP_PATH='./'+city+'/clean/synth_pop.csv'
PICKLED_MODEL_PATH='./models/trip_mode_rf.p'
ZONE_NODE_ROUTES_PATH='./'+city+'/clean/route_nodes.json'
CONNECTION_NODE_ROUTES_PATH='./'+city+'/clean/connection_route_nodes.json'
NODES_PATH='./'+city+'/clean/nodes.csv'
CONNECTION_POINTS_PATH='./'+city+'/clean/connection_points.json'


speeds={0:40,
        1:15,
        2:5,
        3:25}

UTM_MAP={'Boston':pyproj.Proj("+init=EPSG:32619"),
     'Hamburg':pyproj.Proj("+init=EPSG:32632")}

utm=UTM_MAP[city]
wgs=pyproj.Proj("+init=EPSG:4326")

TIMESTEP_SEC=10
counter=0

# getting grid data
lastId=0
host='https://cityio.media.mit.edu/'
#host='http://localhost:8080/' # local port running cityio
cityIO_grid_url='{}api/table/mocho'.format(host)



# load the pre-calibrated choice model
mode_rf=pickle.load( open( PICKLED_MODEL_PATH, "rb" ) )
# load the cityio template
cityio_json=json.load(open(CITYIO_TEMPLATE_PATH))
# load the connection points between real network and grid network
connection_points=json.load(open(CONNECTION_POINTS_PATH))
# load the routes from each zone to each connection
zone_connection_routes=json.load(open(CONNECTION_NODE_ROUTES_PATH))
# get the routes between each exisitng zone
zone_routes=json.load(open(ZONE_NODE_ROUTES_PATH))

nodes=pd.read_csv(NODES_PATH)

original_net_node_coords=nodes[['lon', 'lat']].values.tolist()

# create land use grid and road network from the grid
# TODO the grid information should come from cityIO grid data

# =============================================================================
# Happens with each change 
# =============================================================================
with urllib.request.urlopen(cityIO_grid_url) as url:
#get the latest json data
    cityIO_grid_data=json.loads(url.read().decode())
topLeft_lonLat={'lat':53.533192, 'lon':10.014198}
topEdge_lonLat={'lat':53.531324, 'lon':10.019037}

cell_size= cityIO_grid_data['header']['spatial']['cellSize']
nrows=cityIO_grid_data['header']['spatial']['nrows']
ncols=cityIO_grid_data['header']['spatial']['ncols']
grid_points_ll, net=createGrid(topLeft_lonLat, topEdge_lonLat, utm, wgs, cell_size, nrows, ncols)
# for each connection point, find the closest grid node
grid_tree = spatial.KDTree(np.array(grid_points_ll))
for cp in connection_points:
    cp['closest_grid_node']=grid_tree.query([cp['lon'], cp['lat']])[1]
# get all the internal routes and all the connection routes
grid_node_paths =  dict(nx.all_pairs_shortest_path(net))
grid_routes={str(i):{} for i in range(len(grid_points_ll))}
for o in range(len(grid_points_ll)):
    for d in range(len(grid_points_ll)):
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
# =============================================================================

# for each zone, create a location object
zone_locations=[]
for z in range(len(zone_routes)):
    zone_locations.append(Location(0, z, connection_routes[0][z]))
    
grid_locations=[]
for n in range(len(grid_routes)):
    grid_locations.append(Location(1, n, connection_routes[1][n]))


# for each person in base pop, create Person
random.seed(0)
synth_pop=pd.read_csv(SYNTHPOP_PATH)
ag_ind=0
num_base=100
num_internal=20
num_commute_in=50
base_agents=[]
for ag_ind, row in synth_pop[:num_base].iterrows():
    base_agents.append(Person(row['age'], row['bachelor_degree'], row['hh_income'], zone_locations[row['home_geo_index']], 
                         zone_locations[row['work_geo_index']], row['male'], row['motif'], row['pop_per_sqmile_home'], ag_ind))
    ag_ind+=1
new_agents=[]

agents= base_agents+ new_agents

# create some people internal to grid
# new agents
#for j in range(num_internal):
#    home_grid_cell=random.choice(range(200))
#    work_grid_cell=random.choice(range(200))
#    agents.append(Person(25, True, 5, grid_locations[home_grid_cell], 
#                         grid_locations[work_grid_cell], True, 'HWH', 8000, ag_ind))
#    ag_ind+=1
## create some people commuting into grid
#for j in range(num_commute_in):    
#    home_zone=random.choice(range(len(zone_locations)))
#    work_grid_cell=random.choice(range(200))
#    agents.append(Person(25, True, 5, zone_locations[home_zone], 
#                         grid_locations[work_grid_cell], True, 'HWH', 8000, ag_ind))
#    ag_ind+=1
    
# =============================================================================
# Simulation Loop
# =============================================================================   
period=0
for ag in agents: ag.init_period(period)
prop=0
count=1
while True:
    if count%20==0:
        check_grid_data(period)
    if prop>0.5:
        period+=1
        for ag in agents: ag.init_period(period)
    update_and_send()
    prop=sum([ag.finished for ag in agents])/len(agents)
    count+=1
    print(count)
    

            