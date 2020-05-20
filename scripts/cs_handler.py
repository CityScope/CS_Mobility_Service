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
    def __init__(self, mobility_model,  host='https://cityio.media.mit.edu/',sleep_time=1):
        self.model=mobility_model
        self.table_name=mobility_model.table_name
        self.CITYIO_GET_URL=mobility_model.CITYIO_GET_URL
        self.sleep_time=sleep_time # seconds
        self.grid_hash_id=-1
        self.initialise_model()

        
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
            self.model.post_trips_layer()
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
    
    def random_geogrid_data(self):
        geogrid_data=[]
        for cell in self.model.geogrid.cells:
            if cell.updatable:                
                cell_type=random.choice([
                    type_name for type_name in self.model.geogrid.type_defs])
                cell_height=random.randint(1,10)
            else:
                cell_type=cell.base_land_use
                cell_height=cell.base_height
            geogrid_data.append({'name': cell_type, 'height': cell_height})
        return geogrid_data
    
    def get_outputs(self):
        all_persons=self.model.pop.base_sim+self.model.pop.new
        avg_co2=self.model.get_avg_co2(all_persons)
        live_work_prop=self.model.get_live_work_prop(all_persons)
        mode_split=self.model.get_mode_split(all_persons)
        delta_f_physical_activity_pp=self.model.health_impacts_pp(all_persons)
        return {'avg_co2': avg_co2, 'live_work_prop': live_work_prop,
                'mode_split': mode_split, 
                'delta_f_physical_activity_pp': delta_f_physical_activity_pp}
                
    def generate_training_example_co2_lwp(self):
        geogrid_data=self.random_geogrid_data()
        x={cs_type:0 for cs_type in self.model.geogrid.type_defs}
        for g in geogrid_data:
            x[g['name']]+=g['height']
        self.model.update_simulation(geogrid_data)
        y=self.get_outputs()
        return x, y
    
    def post_trips_data(self):
        all_persons=self.model.pop.base_sim+self.model.pop.new
        self.model.post_trips_layer(all_persons)

        
    
    def generate_training_data(self, iterations):
        """ In order to train a ML model to approximate the results of the simulation
        in deployments where indicators must be available in real-time
        """
        X=[]
        Y=[]
        for it in range(iterations):
            x, y = self.generate_training_example_co2_lwp()
            X.append(x)
            Y.append(y)
        return X, Y
            
        
        
def main():
    this_model=MobilityModel('corktown', 'Detroit')
    
    this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))
    
    this_model.assign_mode_choice_model(NhtsModeLogit(table_name='corktown', city_folder='Detroit'))
    
    this_model.assign_home_location_choice_model(
            TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                             geogrid=this_model.geogrid, base_vacant_houses=this_model.pop.base_vacant))
    
    handler=CS_Handler(this_model)
    X, Y = handler.generate_training_data(iterations=3)

if __name__ == '__main__':
	main()  
    