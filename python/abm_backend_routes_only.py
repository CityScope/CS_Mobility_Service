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

class Person:
    def __init__(self,row, id):
        self.id=id
        self.age=row['age']
        self.bachelor_degree=row['bachelor_degree']
        self.hh_income=row['hh_income']
        self.home_geo_index=row['home_geo_index']
        self.work_geo_index=row['work_geo_index']
        self.male=row['male']
        self.motif=row['motif']
        self.pop_per_sqmile_home=row['pop_per_sqmile_home']        
#        root_route_work_xy=routes[str(self.home_geo_index)][str(self.work_geo_index)]['xy']
#        root_route_home_xy=routes[str(self.work_geo_index)][str(self.home_geo_index)]['xy']
#        home_loc=[[root_route_work_xy[0][0]+random.gauss(0,50), root_route_work_xy[0][1]+random.gauss(0,50)]]
#        work_loc=[[root_route_work_xy[-1][0]+random.gauss(0,50), root_route_work_xy[-1][1]+random.gauss(0,50)]]
        self.all_routes=[routes[str(self.home_geo_index)][str(self.work_geo_index)].copy(),
                         routes[str(self.work_geo_index)][str(self.home_geo_index)].copy()]

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
                    
                    
class new_person(Person):
    def __init__(self, work_grid_cell, home_zone, id):
        self.id=id
        self.age=20
        self.bachelor_degree=True
        self.hh_income=5
        self.home_geo_index=home_zone
        self.work_geo_index=-1
        self.male=True
        self.motif='HWH'
        self.pop_per_sqmile_home=15000
        self.work_grid_cell=work_grid_cell
        self.all_routes=[full_routes_city_to_grid[home_zone][work_grid_cell].copy(),
                         full_routes_grid_to_city[work_grid_cell][home_zone].copy()] 
class new_person_live_work(Person):
    def __init__(self, work_grid_cell, home_grid_cell, id):
        self.id=id
        self.age=20
        self.bachelor_degree=True
        self.hh_income=5
        self.home_geo_index=-1
        self.work_geo_index=-1
        self.male=True
        self.motif='HWH'
        self.pop_per_sqmile_home=15000
        self.work_grid_cell=work_grid_cell
        self.home_grid_cell=home_grid_cell
        self.all_routes=[routes_grid_to_grid[home_grid_cell][work_grid_cell].copy(),
                         routes_grid_to_grid[home_grid_cell][work_grid_cell].copy()] 

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

def sim_period():
    prop_finished=0
    while prop_finished<0.95: 
        print(prop_finished)
    #    start_time=datetime.datetime.now()
        features=[]
        for ag in agents:
            if not ag.finished:
                ag.update_position(TIMESTEP_SEC)
#            ll=pyproj.transform( utm, wgs, ag.position[0], ag.position[1])
#            ll=[int(ll[0]*1e4)/1e4, int(ll[1]*1e4)/1e4] # reduce precision for sending data
            geometry={"type": "Point",
                     "coordinates": [ag.position[0], ag.position[1]]
                    }
            feature={"type": "Feature",
                     "geometry":geometry,
                     'properties':{'mode':ag.mode, 'age':ag.age, 'hh_income':ag.hh_income, 'id': ag.id}
                     }
            features.append(feature)        
        geojson_object={
          "type": "FeatureCollection",
          "features": features    
        }
        
        cityio_json['objects']={"points": geojson_object}    
        r = requests.post('https://cityio.media.mit.edu/api/table/update/abm_service_'+city, data = json.dumps(cityio_json))
    #    end_time=datetime.datetime.now()
    #    print((end_time-start_time).microseconds)
        prop_finished=sum([a.finished for a in agents])/len(agents)
        time.sleep(0.05)

# =============================================================================
# Constants
# =============================================================================
city='Hamburg'

ZONE_ROUTES_PATH='./'+city+'/clean/routes.json'
CITYIO_TEMPLATE_PATH='./'+city+'/clean/cityio_template.json'
SYNTHPOP_PATH='./Hamburg/clean/synth_pop.csv'
PICKLED_MODEL_PATH='./models/trip_mode_rf.p'
CONNECTION_ROUTES_PATH='./'+city+'/clean/connections.json'
CONNECTION_POINTS_PATH='./'+city+'/clean/connection_points.json'


speeds={0:40,
        1:15,
        2:5,
        3:25}

UTM_MAP={'Boston':pyproj.Proj("+init=EPSG:32619"),
     'Hamburg':pyproj.Proj("+init=EPSG:32632")}

utm=UTM_MAP[city]
wgs=pyproj.Proj("+init=EPSG:4326")

# load the pre-calibrated choice model
mode_rf=pickle.load( open( PICKLED_MODEL_PATH, "rb" ) )
# load the cityio template
cityio_json=json.load(open(CITYIO_TEMPLATE_PATH))
# load the connection points between real network and grid network
connection_points=json.load(open(CONNECTION_POINTS_PATH))
# load the routes from each zone to each connection
connection_routes=json.load(open(CONNECTION_ROUTES_PATH))
# get the routes between each exisitng zone
routes=json.load(open(ZONE_ROUTES_PATH))

# =============================================================================
#  Compute all the routes
# =============================================================================

# prepare the routes between each existing zones
for o in routes:
    for d in routes[o]:
        xy=[list(pyproj.transform(wgs, utm, r[0], r[1])) for r in routes[o][d]['coordinates']]
        routes[o][d]['xy']=xy
        routes[o][d]['distances']=[((xy[i][0]-xy[i+1][0])**2+(xy[i][1]-xy[i+1][1])**2)**(1/2) for i in range(len(xy)-1)]

for o in connection_routes['route_from_grid']:
    for d in connection_routes['route_from_grid'][o]:
        xy=[list(pyproj.transform(wgs, utm, r[0], r[1])) for r in connection_routes['route_from_grid'][o][d]['coordinates']]
        connection_routes['route_from_grid'][o][d]['xy']=xy
        connection_routes['route_from_grid'][o][d]['distances']=[((xy[i][0]-xy[i+1][0])**2+(xy[i][1]-xy[i+1][1])**2)**(1/2) for i in range(len(xy)-1)]
for o in connection_routes['route_to_grid']:
    for d in connection_routes['route_to_grid'][o]:
        xy=[list(pyproj.transform(wgs, utm, r[0], r[1])) for r in connection_routes['route_to_grid'][o][d]['coordinates']]
        connection_routes['route_to_grid'][o][d]['xy']=xy
        connection_routes['route_to_grid'][o][d]['distances']=[((xy[i][0]-xy[i+1][0])**2+(xy[i][1]-xy[i+1][1])**2)**(1/2) for i in range(len(xy)-1)]

# create land use grid and road network from the grid
# TODO the grid information should come from cityIO grid data
topLeft_lonLat={'lat':53.533192, 'lon':10.014198}
topEdge_lonLat={'lat':53.531324, 'lon':10.019037}
cell_size, nrows, ncols= 20, 10, 20
grid_points_ll, net=createGrid(topLeft_lonLat, topEdge_lonLat, utm, wgs, cell_size, nrows, ncols)
# find the closest grid point to each connection point
grid_tree = spatial.KDTree(np.array(grid_points_ll))
for cp in connection_points:
    cp['closest_grid_node']=grid_tree.query([cp['lon'], cp['lat']])[1]
# find the closest connection point to each zone
cp_per_zone={}
cp_tree = spatial.KDTree(np.array([[cp['lon'], cp['lat']] for cp in connection_points]))
for o in routes:
    ll=routes[o][o]['coordinates'][0]
    cp_per_zone[o]=cp_tree.query(ll)[1]    
# find the route from each connection point to each grid cell
grid_paths =  dict(nx.all_pairs_shortest_path(net))


#construct routs within grid
routes_grid_to_grid={o:{d:{} for d in range(len(grid_points_ll))} for o in range(len(grid_points_ll))}
for o in range(len(grid_points_ll)):
    for d in range(len(grid_points_ll)):
        grid_node_path=grid_paths[o][d]
        grid_path_coords=[grid_points_ll[n].copy() for n in grid_node_path]
        routes_grid_to_grid[o][d]={'coordinates':grid_path_coords.copy(), 'distances': [cell_size*3 for n in range(len(grid_path_coords)-1)]}

#construct the full routes in and out of interaction_zone
full_routes_grid_to_city={n:{d:{} for d in range(len(routes))} for n in range(len(grid_points_ll))}
full_routes_city_to_grid={d:{n:{} for n in range(len(grid_points_ll))} for d in range(len(routes))}
for node in range(len(grid_points_ll)):
    for d in range(len(routes)):
        cp=cp_per_zone[str(d)]
        grid_cp=connection_points[cp]['closest_grid_node']
        grid_node_path_to_city=grid_paths[node][grid_cp]
        grid_node_path_from_city=grid_paths[grid_cp][node]
        grid_path_coords_to_city=[grid_points_ll[n].copy() for n in grid_node_path_to_city]
        grid_path_coords_from_city=[grid_points_ll[n].copy() for n in grid_node_path_from_city]
        # TODO: distance of link from grid node to connection point
#        full_routes_grid_to_city[node][d]={'coordinates':grid_path_coords_to_city + connection_routes['route_from_grid'][str(cp)][str(d)]['coordinates'].copy()
#                                , 'distances': [cell_size for n in range(len(grid_path_coords_to_city)-1)]+[1]+connection_routes['route_from_grid'][str(cp)][str(d)]['distances'].copy()}
        full_routes_grid_to_city[node][d]={'coordinates':grid_path_coords_to_city.copy(), 'distances': [cell_size for n in range(len(grid_path_coords_to_city)-1)]}
        full_routes_city_to_grid[d][node]={'coordinates':grid_path_coords_from_city.copy(), 'distances': [cell_size for n in range(len(grid_path_coords_from_city)-1)]}

## grid to city
#for node in range(len(grid_points_ll)):
#    full_routes_grid_to_city[node]={}
#    for d in range(len(routes)):
#        cp=cp_per_zone[str(d)]
#        grid_cp=connection_points[cp]['closest_grid_node']
#        grid_route=paths[node][grid_cp]
#        grid_route_xy= [grid_points_xy[n] for n in grid_route]
#        full_route=(grid_route_xy+[[connection_points[cp]['x'], connection_points[cp]['y']]] 
#        +connection_routes['route_from_grid'][str(cp)][str(d)]['xy'])
#        full_routes_grid_to_city[node][d]={'xy':full_route}
## city to grid 
#for d in range(len(routes)):
#    full_routes_city_to_grid[d]={}
#    for node in range(len(grid_points_ll)):
#        cp=cp_per_zone[str(d)]
#        grid_cp=connection_points[cp]['closest_grid_node']
#        grid_route=paths[node][grid_cp]
#        grid_route_xy= [grid_points_xy[n] for n in grid_route]
#        full_route=(connection_routes['route_to_grid'][str(d)][str(cp)]['xy']
#        +[[connection_points[cp]['x'], connection_points[cp]['y']]]+grid_route_xy)
#        full_routes_city_to_grid[d][node]={'xy':full_route}
# =============================================================================
# create the agents  
# =============================================================================
random.seed(0)
num_base=300
num_new_commute=0
num_new_live_work=100

synth_pop=pd.read_csv(SYNTHPOP_PATH)
agents=[]
for ag_ind, row in synth_pop[:num_base].iterrows():
    agents.append(Person(row, ag_ind))

# new agents live outside, work inside
for j in range(num_base, num_base+num_new_commute):
    grid_cell=random.choice(range(len(grid_points_ll)))
    home_zone=random.choice(range(len(routes)))
    agents.extend([new_person(grid_cell, home_zone, j)])
    
for j in range(num_base+num_new_commute, num_base+num_new_commute+num_new_live_work):
    home_grid_cell=random.choice(range(len(grid_points_ll)))
    work_grid_cell=random.choice(range(len(grid_points_ll)))
    agents.extend([new_person_live_work(work_grid_cell, home_grid_cell, j)])

period=0
for ag in agents: ag.init_period(period)
# =============================================================================
# Send to cityIO
# =============================================================================
features=[]
for ag in agents:
    geometry={"type": "Point",
             "coordinates": [ag.position[0], ag.position[1]]
            }
    feature={"type": "Feature",
             "geometry":geometry,
             'properties':{'mode':ag.mode, 'age':ag.age, 'hh_income':ag.hh_income, 'id': ag.id,
                           'route': ag.route, 'speed': ag.speed, 'position': ag.position, 
                           'next_node_index': ag.next_node_index, 'next_node_ll': ag.next_node_ll, 
                           'prop_of_link_left': ag.prop_of_link_left ,'finished': ag.finished}
             }
    features.append(feature)        
geojson_object={
  "type": "FeatureCollection",
  "features": features    
}

cityio_json['objects']={"points": geojson_object}    
r = requests.post('https://cityio.media.mit.edu/api/table/update/abm_service_'+city, data = json.dumps(cityio_json))
print(r)