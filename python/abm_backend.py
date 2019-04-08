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

city='Hamburg'

ZONE_ROUTES_PATH='./'+city+'/clean/routes.json'
CITYIO_TEMPLATE_PATH='./'+city+'/clean/cityio_template.json'
SYNTHPOP_PATH='./Hamburg/clean/synth_pop.csv'
PICKLED_MODEL_PATH='./models/trip_mode_rf.p'


speeds={0:40,
        1:15,
        2:5,
        3:25}

UTM_MAP={'Boston':pyproj.Proj("+init=EPSG:32619"),
     'Hamburg':pyproj.Proj("+init=EPSG:32632")}

utm=UTM_MAP[city]
wgs=pyproj.Proj("+init=EPSG:4326")

TIMESTEP_SEC=10


class Person:
    def __init__(self,row):
        self.age=row['age']
        self.bachelor_degree=row['bachelor_degree']
        self.hh_income=row['hh_income']
        self.home_geo_index=row['home_geo_index']
        self.work_geo_index=row['work_geo_index']
        self.male=row['male']
        self.motif=row['motif']
        self.pop_per_sqmile_home=row['pop_per_sqmile_home']        
        root_route_work_xy=routes[str(self.home_geo_index)][str(self.work_geo_index)]['xy']
        root_route_home_xy=routes[str(self.work_geo_index)][str(self.home_geo_index)]['xy']
        home_loc=[[root_route_work_xy[0][0]+random.gauss(0,50), root_route_work_xy[0][1]+random.gauss(0,50)]]
        work_loc=[[root_route_work_xy[-1][0]+random.gauss(0,50), root_route_work_xy[-1][1]+random.gauss(0,50)]]
        self.all_routes=[home_loc+root_route_work_xy+work_loc,
                         work_loc+root_route_home_xy+home_loc]

    def init_period(self, p):
        self.route=self.all_routes[p%len(self.all_routes)]
#        get the travel time and cost for each mode
#        TODO: get travel times of each mode beforehand
        route_distance=sum([((self.route[i][0]-self.route[i+1][0])**2+(self.route[i][1]-self.route[i+1][1])**2)**(1/2) for i in range(len(self.route)-1)])      
        # all times should be in minutes
        [drive_time, cycle_time, walk_time, PT_time]=[(route_distance/speeds[i])*(1000/60) for i in range(4)]
        walk_time_PT, drive_time_PT=5, 5 # minutes
        drive_cost, cycle_cost, walk_cost, PT_cost=0,0,0,0
        self.mode=int(mode_rf.predict(np.array([drive_time, cycle_time, walk_time, PT_time, 
                                   walk_time_PT, drive_time_PT,
                                   drive_cost, cycle_cost, walk_cost, PT_cost,
                                   self.age, self.hh_income, self.male, 
                                   self.bachelor_degree , self.pop_per_sqmile_home]).reshape(1,-1))[0])
        self.speed=speeds[self.mode]
        self.position=self.route[0]
        self.next_node_index=1
        self.next_node_xy=self.route[1]
        self.finished=False
        
    def update_position(self, seconds):
        # update an agent's position along a predefined route based on their 
        # speed and the time elapsed
        dist_to_move_m=self.speed*seconds/3.6
        finished_move=False
        while finished_move==False and self.finished==False:
            d_to_next_node=((self.next_node_xy[0]-self.position[0])**2+
                             (self.next_node_xy[1]-self.position[1])**2)**(1/2)
            move_ratio=dist_to_move_m/d_to_next_node
            if move_ratio<1:
                # just move the agent along this segment. move finished.
                self.position[0]=self.position[0]+move_ratio*(self.next_node_xy[0]-self.position[0])
                self.position[1]=self.position[1]+move_ratio*(self.next_node_xy[1]-self.position[1])
                finished_move=True
            else:
                #agent moves to start of next segment and then continues the move
                self.position[0]=self.next_node_xy[0]
                self.position[1]=self.next_node_xy[1]
                self.next_node_index+=1
                if self.next_node_index==len(self.route):
                    self.finished=True
                else:
                    self.next_node_xy=self.route[self.next_node_index]
                    dist_to_move_m-=d_to_next_node       
def sim_period():
    while sum([a.finished for a in agents])/len(agents)<0.9: 
    #    start_time=datetime.datetime.now()
        features=[]
        for ag in agents:
            if not ag.finished:
                ag.update_position(TIMESTEP_SEC)
            ll=pyproj.transform( utm, wgs, ag.position[0], ag.position[1])
            ll=[int(ll[0]*1e4)/1e4, int(ll[1]*1e4)/1e4] # reduce precision for sending data
            geometry={"type": "Point",
                     "coordinates": ll
                    }
            feature={"type": "Feature",
                     "geometry":geometry,
                     'properties':{'mode':ag.mode, 'age':ag.age, 'hh_income':ag.hh_income}
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
        time.sleep(0.05)

                
# prepare the routes
routes=json.load(open(ZONE_ROUTES_PATH))
for o in routes:
    for d in routes[o]:
        routes[o][d]['xy']=[list(pyproj.transform(wgs, utm, r[0], r[1])) for r in routes[o][d]['coordinates']]

# load the pre-calibrated choice model
mode_rf=pickle.load( open( PICKLED_MODEL_PATH, "rb" ) )

# open the cityio template
cityio_json=json.load(open(CITYIO_TEMPLATE_PATH))

# create the agents  
synth_pop=pd.read_csv(SYNTHPOP_PATH)
agents=[]
for ag_ind, row in synth_pop.iterrows():
    agents.append(Person(row))

# =============================================================================
# Simulation Loop
# =============================================================================
period=0
while True:
    then=datetime.datetime.now()
    for ag in agents: ag.init_period(period)
    now=datetime.datetime.now()
    print((now-then).seconds)
    sim_period()
    period+=1
            