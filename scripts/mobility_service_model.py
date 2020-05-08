#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:47:42 2020

@author: doorleyr
"""


import json
import random
import urllib
import numpy as np
import requests
import pandas as pd
import datetime

from transport_network import Transport_Network,  Polygon_Location
from activity_scheduler import ActivityScheduler
from mode_logit_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC



class MobilityModel():
    def __init__(self, table_name, city_folder):
        # TODO: new housing attributes directly from cityIO data
        self.new_house_attributes={'rent': 1500, 'beds': 2, 'built_since_jan2010': True, 
                  'puma_pop_per_sqmeter': 0.000292, 'puma_med_income': 60000,
                  'pop_per_sqmile': 5000, 'tenure': 'rented'}
        self.NEW_PERSONS_PER_BLD=1
        self.table_name=table_name
        self.city_folder=city_folder
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
        
    def build_zones(self):
        zones_geo=json.load(open(self.ALL_ZONES_PATH))
        sim_zones=json.load(open(self.SIM_ZONES_PATH))
        all_zones=[]
        self.zone_geoid_index=[]
        for feature in zones_geo['features']:
            geoid=feature['properties']['GEO_ID'] 
            is_sim_zone=geoid in sim_zones
            geoid_short=geoid.split('US')[1] 
            new_zone=Polygon_Location(geometry=feature['geometry'], 
                     area_type='zone',
                     in_sim_area=is_sim_zone,
                    geoid=geoid_short)
            new_zone.get_close_nodes(transport_network=self.tn)
            all_zones.append(new_zone)
            self.zone_geoid_index.append(geoid_short)
        self.zones=all_zones  
        


        
    def build_geogrid(self):
        with urllib.request.urlopen(self.CITYIO_GET_URL+'/GEOGRID') as url:
            geogrid_geojson=json.loads(url.read().decode())
        self.geogrid=GeoGrid(geogrid_geojson, transport_network=self.tn, 
                             city_folder=self.city_folder)
        
        
    def build_transport_networks(self):
        print('Building transport network')
        # internal network
        self.tn=Transport_Network('corktown', 'Detroit')

        
    def build_synth_pop(self):
        print('Building synthetic population')
        base_sim_persons=json.load(open(self.SIM_POP_PATH))
        base_floating_persons=json.load(open(self.FLOATING_PATH))
        base_vacant_housing=json.load(open(self.VACANT_PATH))
        self.pop=Population(base_sim_persons, base_floating_persons, base_vacant_housing, model=self)

    def assign_activity_scheduler (self, act_sch): 
        self.activity_scheduler=act_sch

    
    def assign_mode_choice_model(self, mc):
        self.mode_choice_model=mc
    
    def assign_home_location_choice_model(self, hlc):
        self.hlc=hlc
        
    def get_geogrid_data(self):
        pass
    
    def init_simulation(self):
        print('Initialising Simulation')
        print('\t Activity Scheduling')
        for person in self.pop.base_sim:
            self.activity_scheduler.sample_activity_schedules(person, model=self)
        self.create_trips(self.pop.base_sim)
        self.predict_trip_modes(self.pop.base_sim)
#        self.predict_mode_choices()
            
    def create_trips(self, persons):
        print('\t Creating Trips')
        for person in persons:
            trips=[]
            for ind_act in range(len(person.activities)-1):
                origin=person.activities[ind_act]
                destination=person.activities[ind_act+1]
                mode_choice_set=self.tn.get_routes(origin.location, destination.location)
                enters_sim=((origin.location.area_type=='grid') or (destination.location.area_type=='grid'))
                trips.append(Trip(mode_choice_set, enters_sim=enters_sim,
                                  from_activity=origin, to_activity=destination))
            person.assign_trips(trips)

            
    def predict_trip_modes(self, persons):
        print('\t Predicting Trip modes')
        all_trips=[]
        for person_id, p in enumerate(persons):
            all_trips.extend(p.trips_to_list(person_id=person_id))
        self.mode_choice_model.generate_feature_df(all_trips)
        self.mode_choice_model.predict_modes()
        for i, trip_record in enumerate(all_trips):
            person_id=trip_record['person_id']
            trip_id=trip_record['trip_id']
            predicted_mode=self.mode_choice_model.predicted_modes[i]
            trip_utility=self.mode_choice_model.predicted_v[i][predicted_mode]
            persons[person_id].trips[trip_id].set_mode(predicted_mode, 
                   mode_list=self.tn.base_modes, utility=trip_utility)
            
    def post_outputs(self, persons):
        trips_layer_data=[]
        for person in persons:
            for trip in person.trips:
                if trip.enters_sim:
                    trips_layer_data.append(trip.to_deckgl_trip_format())
        post_url=self.CITYIO_POST_URL+'ABM'
        trips_str = json.dumps(trips_layer_data)
        try:
            r = requests.post(post_url, data = trips_str)
            print('Trips Layer: {}'.format(r))
        except requests.exceptions.RequestException as e:
            print('Couldnt send to cityio')        
            
    def update_simulation(self, geogrid_data):
        then=datetime.datetime.now()
        print('Updating Simulation')
        self.geogrid_data=geogrid_data
        self.update_grid()
        self.create_new_agents()
        self.hlc.home_location_choices(self.pop)
        new_sim_pop=self.pop.new + self.pop.base_floating
        print('\t Activity Scheduling')
        for person in new_sim_pop:
            self.activity_scheduler.sample_activity_schedules(person, model=self)
        self.create_trips(new_sim_pop)
        self.predict_trip_modes(new_sim_pop)  
        now=datetime.datetime.now()
        print('Update took {} seconds'.format((now-then).total_seconds()))

    def update_grid(self):
        for grid_index, ui_land_use_index in enumerate(self.geogrid_data):
            self.geogrid.cells[grid_index].set_new_land_use(ui_land_use_index)
        
    def create_new_agents(self):
        print('\t Creating new agents')
        new_persons=[]
        new_houses=[]
        for cell in self.geogrid.cells:
            if cell.new_land_use:
                if 'Office' in cell.new_land_use:
                    for i in range(self.NEW_PERSONS_PER_BLD):
                        add_person_record=random.choice(self.pop.base_floating_person_records).copy()
                        add_person=Person(add_person_record, p_id='n'+str(len(new_persons)))
                        add_person.assign_work_location(cell)
                        new_persons.append(add_person)
                if 'Residential' in cell.new_land_use:
                    add_house_record=self.new_house_attributes.copy()
                    add_house_id=len(self.pop.base_vacant)+len(new_houses)
                    add_house=Housing_Unit(add_house_record, house_id=add_house_id)
                    add_house.set_location(cell)
                    new_houses.append(add_house)
        self.pop.add_new_pop(new_persons)
        self.pop.add_new_housing_units(new_houses)
        
    def get_avg_co2(self, persons):
        total_co2_kg=0
        total_dist=0
        count=0
        for p in persons:
            count+=1
            for trip in p.trips:
                if trip.total_distance < 1000000:
                    # hack used here because when a route cant be found by some mode
                    # an arbitrality large distance is assumed for purpose of mode choice model
                    mode=trip.mode
                    total_co2_kg+=trip.total_distance*mode.co2_emissions_kg_met
                    total_dist+=trip.total_distance
        return total_co2_kg/count, total_dist/count
    
    def get_mode_split(self, persons):
        count=0
        split={m.name: 0 for m in self.tn.base_modes}
        for p in persons:
            for trip in p.trips:
                count+=1
                split[trip.mode.name]+=1
        prop_split={mode_name: split[mode_name]/count for mode_name in split}
        return prop_split
    
    def get_live_work_prop(self, persons):
        live_work=0
        for p in persons:
            if p.home_loc is not None:
                if (p.home_loc.area_type=='grid') and (p.work_loc.area_type=='grid'):
                    live_work+=1
        return live_work/len(self.pop.base_sim+self.pop.base_floating+self.pop.new)
    
    def get_avg_utility(self, persons):
        total_v=0
        count=0
        for p in persons:
            for trip in p.trips:
                if trip.total_distance < 1000000:
                    count+=1
                    # hack used here because when a route cant be found by some mode
                    # an arbitrality large distance is assumed for purpose of mode choice model
                    total_v+=trip.utility
        return total_v/count
        

        

                    
        
class Population():
    def __init__(self, base_sim_persons, base_floating_persons, base_vacant_housing, model):
        self.base_floating_person_records=base_floating_persons
        self.base_sim=[]
        self.base_floating=[]
        self.base_vacant=[]
        self.new=[]

        for ind_p, person_record in enumerate(base_sim_persons):
            new_person=Person(person_record, p_id='b' + str(ind_p))
            new_person.get_home_location_from_zone(model=model)
            new_person.get_work_location_from_zone(model=model)
            self.base_sim.append(new_person)
        for ind_p, person_record in enumerate(base_floating_persons):
            new_person=Person(person_record , p_id='f' + str(ind_p))
            new_person.get_work_location_from_zone(model=model)
            new_person.assign_home_location(None)
            self.base_floating.append(new_person)
        for ind_h, housing_record in enumerate(base_vacant_housing):
            new_housing_unit=Housing_Unit(housing_record, house_id=ind_h)
            new_housing_unit.set_location(model.zones[model.zone_geoid_index.index(new_housing_unit.home_geoid)])
            self.base_vacant.append(new_housing_unit)
    def add_new_pop(self, new_pop):
        self.new=new_pop
    def add_new_housing_units(self, new_housing):
        self.new_vacant=new_housing
                
class Person():
    def __init__(self, attributes, p_id):
        for key in attributes:
            setattr(self, key, attributes[key]) 
        self.person_id=p_id
        self.trips=[]
        
    def get_home_location_from_zone(self, model):
        zone_index=model.zone_geoid_index.index(self.home_geoid)
        if model.zones[zone_index].in_sim_area:
            # select a geogrid cell to be the home location
            self.home_loc=random.choice(model.geogrid.find_locations_for_activity('Home'))
        else:
            # use the zone itself as the location
            self.home_loc=model.zones[zone_index]
    def get_work_location_from_zone(self, model):
        zone_index=model.zone_geoid_index.index(self.work_geoid)
        if model.zones[zone_index].in_sim_area:
            # select a geogrid cell to be the home location
            self.work_loc=random.choice(model.geogrid.find_locations_for_activity('Work'))
        else:
            # use the zone itself as the location
            self.work_loc=model.zones[zone_index]
            
    def assign_work_location(self, loc):
        self.work_loc=loc
        
    def assign_home_location(self, loc):
        self.home_loc=loc
                        
    def assign_activities(self, activities):
        self.activities=activities
        
    def assign_trips(self, trips):
        self.trips=trips
        
    def trips_to_list(self, person_id):
        trips_list=[]
        for i_t,t in enumerate(self.trips):
            this_trip_record={
                    'person_id': person_id,
                     'trip_id': i_t,
                     'income': self.income,
                     'age': self.age,
                     'children': self.children,
                     'workers': self.workers,
                     'tenure': self.tenure,
                     'sex': self.sex,
                     'bach_degree': self.bach_degree,
                     'race': self.race,
                     'cars': self.cars,
                     'pop_per_sqmile_home': self.pop_per_sqmile_home,
                     'purpose': t.purpose}
            if t.mode_choice_set:
                for m in t.mode_choice_set:
                    this_trip_record[m+'_route']=t.mode_choice_set[m].costs
                trips_list.append(this_trip_record)
        return trips_list
        


#mode_choice_set
        
#class Mode():
#    def __init__(self, graph_ind, name, speed_met_s, kg_co2_per_met)  :
        
class Housing_Unit():
    def __init__(self, attributes, house_id):       
        for key in attributes:
            setattr(self, key, attributes[key]) 
        self.house_id=house_id
#        if self.home_geoid is not None:
#            self.loc=model.zones[model.zone_geoid_index.index(self.home_geoid)]
            
    def set_location(self, loc):
        self.loc=loc
        
                
class GridCell(Polygon_Location):
    def attach_geogrid(self, geogrid, geogrid_id):
        self.geogrid=geogrid
        self.geogrid_id=geogrid_id
    def set_initial_land_use(self, base_land_use):
        self.new_land_use=None
        if base_land_use is None:
            self.initial_land_use= None
            self.land_use=None
        elif base_land_use=='None':
            self.initial_land_use= None
            self.land_use=None
        else:
            standardised_lu=self.geogrid.base_lu_to_lu[base_land_use]
            self.initial_land_use=standardised_lu
            self.land_use=standardised_lu
    def set_new_land_use(self, ui_land_use_index):
        try: 
            ui_land_use_name=self.geogrid.lu_inputs[str(ui_land_use_index)]
        except:
            return None
        possible_standard_lus=self.geogrid.lu_input_to_lu_standard[ui_land_use_name]
        new_land_use= np.random.choice([k for k in possible_standard_lus],
                            1, p=[v for v in possible_standard_lus.values()])[0]
        self.new_land_use=new_land_use
        self.land_use=new_land_use
    def set_interactivity(self, interactive):
        self.interactive=interactive
        
class GeoGrid():
    def __init__(self, grid_geojson, transport_network, city_folder):
        MAPPINGS_PATH = './cities/'+city_folder+'/mappings'
        self.lu_inputs=json.load(open(MAPPINGS_PATH+'/lu_inputs.json'))
        self.lu_input_to_lu_standard=json.load(open(MAPPINGS_PATH+'/lu_input_to_lu_standard.json'))
        self.activities_to_lu=json.load(open(MAPPINGS_PATH+'/activities_to_lu_2.json'))
        self.base_lu_to_lu=json.load(open(MAPPINGS_PATH+'/base_lu_to_lu.json'))
        self.cells=[]
        for ind_f, feature in enumerate(grid_geojson['features']):
            new_grid_cell=GridCell(feature['geometry'], 
                                    area_type='grid',
                                    in_sim_area=True)
            new_grid_cell.set_interactivity(True)
            new_grid_cell.attach_geogrid(geogrid=self, geogrid_id=ind_f)
            new_grid_cell.get_close_nodes(transport_network=transport_network)
#            new_grid_cell.set_initial_land_use(feature['properties']['land_use'])
            new_grid_cell.set_initial_land_use(feature['properties']['name'])
            self.cells.append(new_grid_cell)
    def find_locations_for_activity(self, activity):
        possible_lus=self.activities_to_lu[activity]
        possible_cells=[c for c in self.cells if c.land_use in possible_lus]
#        if len(possible_cells)==0:
#            print('No suitable cells found for {}'.format(activity))
        return possible_cells
    

        
            
class Trip():
    def __init__(self, mode_choice_set, enters_sim, from_activity, to_activity):
        self.enters_sim=enters_sim
        self.activity_start=from_activity.start_time
        self.mode_choice_set=mode_choice_set
        self.mode=None
        if from_activity.activity_id + to_activity.activity_id in ['HW', 'WH']:
            self.purpose='HBW'
        elif 'H' in [from_activity.activity_id , to_activity.activity_id]:
            self.purpose= 'HBO'
        else:
            self.purpose='NHB'
    def set_mode(self, mode, mode_list, utility=None):
        mode_name=mode_list[mode].name
        self.mode=mode_list[mode]
        self.internal_route=self.mode_choice_set[mode_name].internal_route['internal_route']
        self.pre_time=self.mode_choice_set[mode_name].pre_time
        self.post_time=self.mode_choice_set[mode_name].post_time
        self.total_distance=self.internal_route['total_distance'
                                ]+((self.pre_time)+(self.post_time))*60*self.mode.speed_met_s
        if utility is not None:
            self.utility=utility
    def to_deckgl_trip_format(self):
        cum_dist=np.cumsum(self.internal_route['distances'])
        internal_trip_start_time=self.activity_start+self.pre_time
        timestamps=[int(internal_trip_start_time)] + [
                int(internal_trip_start_time+ (cd/self.mode.speed_met_s)) for cd in cum_dist]
        trips_object={'mode': [self.mode.id, 0],
                       'path': self.internal_route['coords'],
                       'timestamps': timestamps}
        return trips_object


            
def main():
    this_model=MobilityModel('corktown', 'Detroit')
    
    this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))
    
    this_model.assign_mode_choice_model(NhtsModeLogit(table_name='corktown', city_folder='Detroit'))
    
    this_model.assign_home_location_choice_model(
            TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                             geogrid=this_model.geogrid, base_vacant_houses=this_model.pop.base_vacant))
    
    this_model.init_simulation()
    avg_co2=this_model.get_avg_co2(this_model.pop.base_sim)
    print(avg_co2)
    print(this_model.get_mode_split(this_model.pop.base_sim))
    print(this_model.get_live_work_prop(this_model.pop.base_sim))
    print(this_model.get_avg_utility(this_model.pop.base_sim))
    
    geogrid_data=[None for i in range(len(this_model.geogrid.cells))]
    for ig in range(len(geogrid_data)):
        if this_model.geogrid.cells[ig].interactive:
            geogrid_data[ig] = random.randint(0, 6)
#    for ig in range(len(geogrid_data)):
#        if this_model.geogrid.cells[ig].interactive:
#            geogrid_data[ig] = 1
    this_model.update_simulation(geogrid_data)
    # TODO: subset by people with live or work in sim area
    all_persons=this_model.pop.base_sim+this_model.pop.new
    avg_co2=this_model.get_avg_co2(all_persons)
    print(avg_co2)
    print(this_model.get_mode_split(all_persons))
    print(this_model.get_live_work_prop(all_persons))
    print(this_model.get_avg_utility(all_persons))

#
#if __name__ == '__main__':
#	main()  