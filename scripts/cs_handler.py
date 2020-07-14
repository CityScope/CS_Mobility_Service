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
import requests

from mobility_service_model import MobilityModel
from activity_scheduler import ActivityScheduler
from mode_choice_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC

class CS_Handler():
    def __init__(self, mobility_model,  host_mode='remote',sleep_time=1, new_logit_params={}):
        self.model=mobility_model
        self.new_logit_params=new_logit_params
        self.table_name=mobility_model.table_name
        self.CITYIO_GET_URL=mobility_model.CITYIO_GET_URL
        self.sleep_time=sleep_time # seconds
        self.grid_hash_id=-1
        self.initialise_model()
        self.host_mode=host_mode

        
    def initialise_model(self):
        self.model.init_simulation(new_logit_params=self.new_logit_params)
        
    def listen_city_IO(self):
        while True:
            sleep(self.sleep_time)
            grid_hash_id=self.get_grid_hash_id()
            if not grid_hash_id==self.grid_hash_id:
                self.perform_city_io_update(grid_hash_id)
                
    def perform_city_io_update(self, grid_hash_id):
        print('Starting update')
        geogrid_data=self.get_geogrid_data()
        if geogrid_data is not None:
            self.model.update_simulation(geogrid_data, new_logit_params=self.new_logit_params)
            self.ind_get_update_post()
            self.model.post_trips_layer()
            self.grid_hash_id=grid_hash_id
        print('Done with update')
        
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
    
    def random_geogrid_data(self, ref_geogrid):
        geogrid_data=[]
        for i_c, cell in enumerate(self.model.geogrid.cells):
            if cell.interactive:  
                if random.randint(1,5)==3:
                    # only change 20% of cells
                    cell_type=random.choice([
                        type_name for type_name in self.model.geogrid.int_type_defs])
                    cell_height=random.randint(1,10)
                else:
                    cell_type=ref_geogrid[i_c]['name']
                    cell_height=ref_geogrid[i_c]['height']
                    if isinstance(cell_height, list):
                        cell_height=cell_height[-1]
            else:
                cell_type=cell.base_land_use
                cell_height=cell.base_height
            geogrid_data.append({'name': cell_type, 'height': cell_height})
        return geogrid_data
    
    def normalise_ind(self, value, min_value, max_value):
        return (value-min_value)/(max_value-min_value)
    
    def get_outputs(self):
        avg_co2=self.model.get_avg_co2()
        avg_co2_norm=self.normalise_ind(avg_co2, min_value=12, max_value=5)
        live_work_prop=self.model.get_live_work_prop()
        mode_split=self.model.get_mode_split()
        delta_f_physical_activity_pp=self.model.health_impacts_pp()
        delta_f_norm=self.normalise_ind(delta_f_physical_activity_pp, min_value=0, max_value=0.004)
        output= {'CO2 Performance raw kg/day': avg_co2, 
                 'Mobility Health Impacts raw mortality/year':delta_f_physical_activity_pp,
                 'CO2 Performance norm': avg_co2_norm,
                 'Mobility Health Impacts norm': delta_f_norm
                 }
        for mode in mode_split:
            output[mode]=100*mode_split[mode]
        return output
                
    def generate_training_example_co2_lwp(self, ref_geogrid):
        geogrid_data=self.random_geogrid_data(ref_geogrid)
        x={cs_type:0 for cs_type in self.model.geogrid.type_defs}
        for g in geogrid_data:
            x[g['name']]+=g['height']
        self.model.update_simulation(geogrid_data, new_logit_params=self.new_logit_params)
        y=self.get_outputs()
        return x, y
    
    def post_trips_data(self):
        self.model.post_trips_layer()
        
    def calculate_indicators(self):
        avg_co2=self.model.get_avg_co2()
        avg_co2_norm=self.normalise_ind(avg_co2, min_value=12, max_value=5)
        delta_f_physical_activity_pp=self.model.health_impacts_pp()
        delta_f_norm=self.normalise_ind(delta_f_physical_activity_pp, min_value=0, max_value=0.004)
        sustainable_mobility=(avg_co2_norm+ delta_f_norm)/2
        indicators=[{'name': 'Mobility CO2 Performance', 'value': avg_co2_norm, 
                 'raw_value':avg_co2,'viz_type': 'bar', 'units': 'kg/day'},
                {'name': 'Mobility Health Impacts', 'value': delta_f_norm, 
                 'raw_value':delta_f_physical_activity_pp, 'viz_type': 'bar', 'units': 'mortality/year'},
                 {'name': 'Sustainable Mobility', 'value': sustainable_mobility, 
                  'units': None, 'raw_value': None, "viz_type": "radar",
                  'ref_value': 0.42}]
        return indicators

        
#    def post_ind(self):
#        new_indicators=self.calculate_indicators()
#        post_url=self.model.CITYIO_POST_URL+'indicators/update'
#        try:            
#            r = requests.post(post_url, data = json.dumps(new_indicators))
#            print('Mobility Indicators: {}'.format(r))
#        except requests.exceptions.RequestException as e:
#            print('Couldnt send to cityio') 
            
    def ind_get_update_post(self):
        new_indicators=self.calculate_indicators()
        post_url=self.model.CITYIO_POST_URL+'indicators'
        with urllib.request.urlopen(self.CITYIO_GET_URL+'/indicators') as url:
            existing_indicators=json.loads(url.read().decode())
        existing_indicator_names=[ind['name'] for ind in existing_indicators]
        for indicator_update in new_indicators:
            try:
                loc=existing_indicator_names.index(indicator_update['name'])
                existing_indicators[loc]['value']=indicator_update['value']
                existing_indicators[loc]['raw_value']=indicator_update['raw_value']
            except:
                existing_indicators.append(indicator_update)
        try:            
            r = requests.post(post_url, data = json.dumps(existing_indicators))
            print('Mobility Indicators: {}'.format(r))
        except requests.exceptions.RequestException as e:
            print('Couldnt send to cityio') 
        

        
        
        
        
           
    def generate_training_data(self, iterations, ref_geogrid):
        """ In order to train a ML model to approximate the results of the simulation
        in deployments where indicators must be available in real-time
        """
        X=[]
        Y=[]
        for it in range(iterations):
            x, y = self.generate_training_example_co2_lwp(ref_geogrid)
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
    