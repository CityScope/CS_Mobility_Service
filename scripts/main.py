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

from mode_logit import long_form_data, asclogit_pred

def approx_shape_centroid(geometry):
    if geometry['type']=='Polygon':
        centroid=list(np.mean(geometry['coordinates'][0], axis=0))
        return centroid
    elif geometry['type']=='MultiPolygon':
        centroid=list(np.mean(geometry['coordinates'][0][0], axis=0))
        return centroid
    else:
        print('Unknown geometry type')

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
        self.PORTALS_PATH='./cities/'+city_folder+'/clean/portals.geojson'
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
        self.build_transport_network()
        self.build_geography()
        self.build_synth_pop()
     
    def build_geography(self):
        print('Building geography')
        self.build_zones()
        self.build_geogrid()
        self.build_portals()
        # geogrid
        
    def build_zones(self):
        zones_geo=json.load(open(self.ALL_ZONES_PATH))
        sim_zones=json.load(open(self.SIM_ZONES_PATH))
        all_zones={}
        for feature in zones_geo['features']:
            geoid=feature['properties']['GEO_ID'].split('US')[1]  
            is_sim_zone=geoid in sim_zones
            all_zones[geoid]=Zone(feature['geometry'], is_sim_zone)
        self.zones=all_zones  
        
    def build_geogrid(self):
        with urllib.request.urlopen(self.CITYIO_GET_URL+'/GEOGRID') as url:
            geogrid_geojson=json.loads(url.read().decode())
        self.geogrid=GeoGrid(geogrid_geojson)
        
    def build_portals(self):
        portals_geojson=json.load(open(self.PORTALS_PATH))
        self.portals=[]
        for feature in portals_geojson['features']:
            self.portals.append(Portal(feature['geometry']))

        
        
    def build_transport_network(self):
        # internal network
        self.internal_net_fw_results=json.load(open(self.FLOYD_PREDECESSOR_PATH))
        self.internal_network_df=pd.read_csv(self.INT_NET_DF_FLOYD_PATH)
        # external costs
        self.external_route_costs=json.load(open(self.ROUTE_COSTS_PATH))
        # TODO: build KD Tree of internal node locations

        
    def build_synth_pop(self):
        print('Building synthetic population')
        base_sim_persons=json.load(open(self.SIM_POP_PATH))
        base_floating_persons=json.load(open(self.FLOATING_PATH))
        base_vacant_housing=json.load(open(self.VACANT_PATH))
        self.pop=Population(base_sim_persons, base_floating_persons, base_vacant_housing, world=self)
        
        

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
        self.location=world.zones[self.home_geoid].centroid
        
class Zone():
    def __init__(self, geometry, is_sim_zone):
        self.geometry=geometry
        self.centroid=approx_shape_centroid(geometry)
        self.is_sim_zone=is_sim_zone
        
class Portal():
    def __init__(self, geometry):
        self.geometry=geometry
        self.centroid=approx_shape_centroid(geometry)
        # get close nodes

# TODO: should zones, portals and grid cells all be extending a location class?        
class Grid_Cell():
    def __init__(self, geometry, initial_land_use):
        self.centroid=approx_shape_centroid(geometry)
        self.dynamic_type=None
        self.initial_land_use=initial_land_use
        
class GeoGrid():
    def __init__(self, grid_geojson):
        self.cells=[]
        for feature in grid_geojson['features']:
            self.cells.append(Grid_Cell(feature['geometry'], feature['properties']['land_use']))
            # TODO: add closest nodes in network
        
        
this_world=World('corktown', 'Detroit')
            
#def main():
#    this_world=World('corktown', 'Detroit')
#
#
#if __name__ == '__main__':
#	main()  