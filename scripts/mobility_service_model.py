#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:47:42 2020

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
#import time
import copy
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
from transport_network import Transport_Network, get_haversine_distance, approx_shape_centroid, Polygon_Location

from mode_logit import long_form_data, asclogit_pred



class World():
    def __init__(self, table_name, city_folder):
        self.table_name=table_name
        self.ALL_ZONES_PATH='./cities/'+city_folder+'/clean/model_area.geojson'
        self.SIM_ZONES_PATH='./cities/'+city_folder+'/clean/sim_zones.json'
        # Synthpop results
        self.SIM_POP_PATH='./cities/'+city_folder+'/clean/sim_pop.json'
        self.VACANT_PATH='./cities/'+city_folder+'/clean/vacant.json'
        self.FLOATING_PATH='./cities/'+city_folder+'/clean/floating.json'
        # Mode choice model
        self.FITTED_RF_MODE_MODEL_PATH='./cities/'+city_folder+'/models/trip_mode_rf.p'
        self.RF_FEATURES_LIST_PATH='./cities/'+city_folder+'/models/rf_features.json'
        self.FITTED_LOGIT_MODE_MODEL_PATH='./cities/'+city_folder+'/models/trip_mode_logit.p'
        self.LOGIT_FEATURES_LIST_PATH='./cities/'+city_folder+'/models/logit_features.json'
        # Home location choice model
        self.FITTED_HOME_LOC_MODEL_PATH='./cities/'+city_folder+'/models/home_loc_logit.p'
        self.RENT_NORM_PATH='./cities/'+city_folder+'/models/rent_norm.json'        
        # activity schedule results
        self.MOTIF_SAMPLE_PATH='./cities/'+city_folder+'/clean/motif_samples.csv'        
        # portals
#        self.PORTALS_PATH='./cities/'+city_folder+'/clean/portals.geojson'
        # external route costs
        self.ROUTE_COSTS_PATH='./cities/'+city_folder+'/clean/route_costs.json'
        # internal network and route costs
        self.FLOYD_PREDECESSOR_PATH='./cities/'+city_folder+'/clean/fw_result.json'
        self.INT_NET_DF_FLOYD_PATH='./cities/'+city_folder+'/clean/sim_net_df_floyd.csv'
        self.INT_NET_COORDINATES_PATH='./cities/'+city_folder+'/clean/sim_net_node_coords.json'
        # PUMAs        
        self.PUMA_SHAPE_PATH='./cities/'+city_folder+'/raw/PUMS/pumas.geojson'
        self.PUMAS_INCLUDED_PATH='./cities/'+city_folder+'/raw/PUMS/pumas_included.json'
        self.PUMA_ATTR_PATH = './cities/'+city_folder+'/models/puma_attr.json'
        self.EXTERNAL_LU_PATH = './cities/'+city_folder+'/clean/external_lu.json'
        # activity-LU mappings
        self.MAPPINGS_PATH = './cities/'+city_folder+'/mappings'
        # city_folderIO
        host='https://cityio.media.mit.edu/'
        self.CITYIO_GET_URL=host+'api/table/'+table_name
        self.CITYIO_POST_URL=host+'api/table/update/'+table_name+'/'
        self.UPDATE_FREQ=1
        
        self.build_model()
    
        
    
    def build_model(self):
        self.build_transport_networks()
        self.build_geography()
        self.build_synth_pop()
     
    def build_geography(self):
        print('Building geography')
        self.build_zones()
        self.build_geogrid()
#        self.build_portals(world=self)
        # geogrid
        
    def build_zones(self):
        zones_geo=json.load(open(self.ALL_ZONES_PATH))
        sim_zones=json.load(open(self.SIM_ZONES_PATH))
        all_zones=[]
        self.zone_geoid_index=[]
        for feature in zones_geo['features']:
            geoid=feature['properties']['GEO_ID'].split('US')[1]  
            is_sim_zone=geoid in sim_zones
            all_zones.append(Polygon_Location(geometry=feature['geometry'], 
                     area_type='zone',
                     in_sim_area=is_sim_zone,
                    geoid=geoid))
            self.zone_geoid_index.append(geoid)
        self.zones=all_zones  
        
    def build_geogrid(self):
        with urllib.request.urlopen(self.CITYIO_GET_URL+'/GEOGRID') as url:
            geogrid_geojson=json.loads(url.read().decode())
        self.geogrid=GeoGrid(geogrid_geojson, transport_network=self.tn)
        
#    def build_portals(self, world):
#        portals_geojson=json.load(open(world.external_routes.PORTALS_PATH))
#        self.portals=[]
#        for feature in portals_geojson['features']:
#            self.portals.append(Portal(feature['geometry']))

        
        
    def build_transport_networks(self):
        # internal network
        self.tn=Transport_Network('corktown', 'Detroit')

        
    def build_synth_pop(self):
        print('Building synthetic population')
        base_sim_persons=json.load(open(self.SIM_POP_PATH))
        base_floating_persons=json.load(open(self.FLOATING_PATH))
        base_vacant_housing=json.load(open(self.VACANT_PATH))
        self.pop=Population(base_sim_persons, base_floating_persons, base_vacant_housing, world=self)
        
    def add_mode_choice_model(self, mc):
        self.mode_choice_model=mc
    
    def add_home_location_choice_model(self, hlc):
        self.home_location_choice_modell=hlc
    
    def add_activity_location_choice_model(self, alc):
        self.activity_location_choice_model=alc
        
    def get_geogrid_data(self):
        pass
    
    def listen(self):
        self.update_simulation()
        self.post_results()
        
    
    def update_simulation():
        pass
#        self.get_activities()
#        self.get_activity_locations()
#        self.get_mode_choice_sets()
#        self.predict_mode_choices()
        
    def post_results():
        pass
        
        

class Population():
    def __init__(self, base_sim_persons, base_floating_persons, base_vacant_housing, world):
        self.base_sim=[]
        self.base_floating=[]
        self.base_vacant=[]
        for person_record in base_sim_persons:
            self.base_sim.append(Person(person_record))
        for person_record in base_floating_persons:
            self.base_floating.append(Person(person_record))
        for housing_record in base_vacant_housing:
            self.base_vacant.append(Housing_Unit(housing_record, world=world))
        
        
class Person():
    def __init__(self, attributes):
        for key in attributes:
            setattr(self, key, attributes[key])  
        
#class Mode():
#    def __init__(self, graph_ind, name, speed_met_s, kg_co2_per_met)  :
        
class Housing_Unit():
    def __init__(self, attributes, world):       
        for key in attributes:
            setattr(self, key, attributes[key]) 
        self.centroid=world.zones[world.zone_geoid_index.index(self.home_geoid)].centroid
                
class Grid_Cell(Polygon_Location):
    def set_initial_land_use(self, land_use):
        self.initial_land_use=land_use
        
class GeoGrid():
    def __init__(self, grid_geojson, transport_network):
        self.cells=[]
        for feature in grid_geojson['features']:
            new_grid_cell=Grid_Cell(feature['geometry'], 
                                    area_type='grid',
                                    in_sim_area=True)
            new_grid_cell.get_close_nodes(transport_network=transport_network)
            self.cells.append(new_grid_cell)
            
this_world=World('corktown', 'Detroit')
            
#def main():
#    this_world=World('corktown', 'Detroit')
#
#
#if __name__ == '__main__':
#	main()  