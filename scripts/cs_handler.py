#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:13:11 2020

@author: doorleyr
"""
from time import sleep
import urllib
import json
import random

from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_logit_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC

class CS_Handler():
    def __init__(self, table_name, city_folder, host='https://cityio.media.mit.edu/',sleep_time=1):
        self.table_name=table_name
        self.city_folder=city_folder
        self.CITYIO_GET_URL=host+'api/table/'+table_name
        self.sleep_time=sleep_time # seconds
        self.grid_hash_id=-1
        self.types=self.get_geogrid_type_defs()
        
    def add_model(self, mobility_model):
        self.model= mobility_model
        
    def initialise_model(self):
        self.model.init_simulation()
        
    def listen_city_IO(self):
        while True:
            sleep(self.sleep_time)
            grid_hash_id=self.get_grid_hash_id()
            if not grid_hash_id==self.grid_hash_id:
                self.perform_city_io_update(grid_hash_id)
                
    def perform_city_io_update(self, grid_hash_id):
        geogrid_data=self.get_geogrid_data()
        if geogrid_data is not None:
            self.model.update_simulation(geogrid_data)
            self.grid_hash_id=grid_hash_id
        
    def get_grid_hash_id(self):
        try:
            with urllib.request.urlopen(self.CITYIO_GET_URL+'/meta/hashes/GEOGRIDDATA') as url:
                hash_id=json.loads(url.read().decode())
            return hash_id
        except:
            print('Cant access cityIO for GEOGRID hash')
            return self.grid_hash_id
        
    def get_geogrid_data(self):
        try:
            with urllib.request.urlopen(self.CITYIO_GET_URL+'/GEOGRIDDATA') as url:
                geogrid_data=json.loads(url.read().decode())
            return geogrid_data
        except:
            print('Cant access cityIO for GEOGRIDDATA')
            return None
    def get_geogrid_type_defs(self):
        with urllib.request.urlopen(self.CITYIO_GET_URL+'/GEOGRID/properties/types') as url:
            self.types=json.loads(url.read().decode())
    
    def get_local_geogrid_data(self):
        pass
    
    def random_geogrid_data(self):
        pass
        
    def generate_training_example(self):
        pass
        # simulate geogriddata
        # save X
        # update simulation
        # save y
    
    def generate_training_data(self):
        """ In order to train a ML model to approximate the results of the simulation
        in deployments where indicators must be available in real-time
        """
        pass
        
        
def main():
    this_model=MobilityModel('corktown', 'Detroit')
    
    this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))
    
    this_model.assign_mode_choice_model(NhtsModeLogit(table_name='corktown', city_folder='Detroit'))
    
    this_model.assign_home_location_choice_model(
            TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                             geogrid=this_model.geogrid, base_vacant_houses=this_model.pop.base_vacant))
    
    handler=CS_Handler(this_model)
    handler.listen_city_IO()
    

if __name__ == '__main__':
	main()  
    