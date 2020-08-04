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
from copy import deepcopy

from transport_network import Transport_Network,  Polygon_Location, Mode
from activity_scheduler import ActivityScheduler
from mode_choice_nhts import NhtsModeLogit
from two_stage_logit_hlc import TwoStageLogitHLC



class MobilityModel():
    def __init__(self, table_name, city_folder, seed=0, host='https://cityio.media.mit.edu/'):
        self.seed=seed
        self.new_house_attributes=[{'p': 0.2, 'rent': 950, 'beds': 3, 'built_since_jan2010': True, 
                  'puma_pop_per_sqmeter': 0.0016077275, 'puma_med_income': 24000,
                  'pop_per_sqmile': 0.0016077275/3.86102e-7, 'tenure': 'rented'},
                                    {'p': 0.8, 'rent': 2000, 'beds': 3, 'built_since_jan2010': True, 
                  'puma_pop_per_sqmeter': 0.0016077275, 'puma_med_income': 24000,
                  'pop_per_sqmile': 0.0016077275/3.86102e-7, 'tenure': 'rented'}]
        self.table_name=table_name
        trip_attrs_path='./cities/'+city_folder+'/clean/trip_attributes.json'
        self.trip_attrs=json.load(open(trip_attrs_path))
        self.city_folder=city_folder
        self.ALL_ZONES_PATH='./cities/'+city_folder+'/clean/model_area.geojson'
        self.SIM_ZONES_PATH='./cities/'+city_folder+'/clean/sim_zones.json'
        self.GEOGRID_PATH='./cities/'+city_folder+'/clean/geogrid.geojson'
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
        # external route costs
        self.ROUTE_COSTS_PATH='./cities/'+city_folder+'/clean/route_costs.json'
        # internal network and route costs
        self.FLOYD_PREDECESSOR_PATH='./cities/'+city_folder+'/clean/fw_result.json'
        self.INT_NET_DF_FLOYD_PATH='./cities/'+city_folder+'/clean/sim_net_df_floyd.csv'
        # activity-LU mappings
        self.MAPPINGS_PATH = './cities/'+city_folder+'/mappings'
        self.CITYIO_GET_URL=host+'api/table/'+table_name
        self.CITYIO_POST_URL=host+'api/table/update/'+table_name+'/'
        self.types=None
        # scale factor must match that from pop-synth
        self.scale_factor=20
        
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
        try:
            with urllib.request.urlopen(self.CITYIO_GET_URL+'/GEOGRID') as url:
                geogrid_geojson=json.loads(url.read().decode())
        except:
            print('Couldnt get GEOGRID. Using local copy')
            geogrid_geojson=json.load(open(self.GEOGRID_PATH))
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
    
    def init_simulation(self, new_logit_params={}):
        print('Initialising Simulation')
        print('\t Activity Scheduling')
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.activity_scheduler.assign_profiles(self.pop.base_sim)
        for person in self.pop.base_sim:
            self.activity_scheduler.sample_activity_schedules(person, model=self)
        self.create_trips(self.pop.base_sim)
        self.predict_trip_modes(self.pop.base_sim, logit_params=new_logit_params)
        self.pop.impact=self.pop.base_sim
#        self.predict_mode_choices()
            
    def create_trips(self, persons):
        print('\t Creating Trips')
        for person in persons:
            trips=[]
            for ind_act in range(len(person.activities)-1):
                origin=person.activities[ind_act]
                destination=person.activities[ind_act+1]
                if not origin==destination:
                    mode_choice_set=self.tn.get_routes(origin.location, destination.location)
                    enters_sim=((origin.location.area_type=='grid') or (destination.location.area_type=='grid'))
                    trips.append(Trip(mode_choice_set, enters_sim=enters_sim,
                                      from_activity=origin, to_activity=destination))
            person.assign_trips(trips)

            
    def predict_trip_modes(self, persons, method='random', seed=None, logit_params={}):
        print('\t Predicting Trip modes')
        temp_mode_choice_model=deepcopy(self.mode_choice_model)
        all_trips=[]
        for person_id, p in enumerate(persons):
            all_trips.extend(p.trips_to_list(person_id=person_id))
        temp_mode_choice_model.generate_feature_df(all_trips)
        if len(temp_mode_choice_model.new_alt_specs)>0:
            for new_spec in temp_mode_choice_model.new_alt_specs:
                temp_mode_choice_model.set_new_alt(new_spec)
        temp_mode_choice_model.set_logit_model_params(logit_params)
        print('\t \t predicting')
        temp_mode_choice_model.predict_modes(method=method, seed=seed)
        mode_list=self.tn.base_modes+ self.tn.new_modes
        print('\t \t applying predictions to trips')
        for i, trip_record in enumerate(all_trips):
            person_id=trip_record['person_id']
            trip_id=trip_record['trip_id']
            predicted_mode=temp_mode_choice_model.predicted_modes[i]
#            trip_utility=self.mode_choice_model.predicted_v[i][predicted_mode]
            persons[person_id].trips[trip_id].set_mode(predicted_mode, 
                   mode_list)
    
    def get_trips_layer(self, persons=None):
        if persons==None:
            persons=self.pop.impact
        trips_layer_data=[]
        for person in persons:
            for trip in person.trips:
                if trip.enters_sim:
                    new_trip=trip.to_deckgl_trip_format(person.motif)
                    if new_trip is not None:
                        trips_layer_data.extend(new_trip)
        mode_split=self.get_mode_split(persons, by_id=True)
        mode_attrs={}
        for mode_ind in self.trip_attrs['mode']:
            if int(mode_ind) in mode_split:
                splt_pct=str(int(1000*mode_split[int(mode_ind)])/10)+'%'# convert to percent with 1 decimal place
                mode_attrs[mode_ind]={'name':self.trip_attrs['mode'][mode_ind]['name']+': '+splt_pct,'color': self.trip_attrs['mode'][mode_ind]['color']}
        return {"trips": trips_layer_data, "attr": {'mode': mode_attrs, 'profile': self.trip_attrs['profile']}}
            
    def post_trips_layer(self, persons=None):
        if persons==None:
            persons=self.pop.impact
        trips_layer_data=self.get_trips_layer(persons)
        if len(trips_layer_data["trips"])>2000:
            trips_layer_data["trips"]=random.sample(trips_layer_data["trips"], 2000)
        post_url=self.CITYIO_POST_URL+'ABM2'
        trips_str = json.dumps(trips_layer_data)
        try:
            r = requests.post(post_url, data = trips_str)
            print('Trips Layer: {}'.format(r))
        except requests.exceptions.RequestException as e:
            print('Couldnt send to cityio')   
            
    def update_simulation(self, geogrid_data, new_logit_params={}):
        then=datetime.datetime.now()
        print('Updating Simulation')
        self.geogrid_data=geogrid_data
        self.update_grid()
        self.activity_scheduler.find_locations_for_activities(self)
        self.create_new_agents()
        self.hlc.home_location_choices(self.pop)
        new_sim_pop=[]
        for p in self.pop.new + self.pop.base_floating:
            if p.home_loc is not None:
                if ((p.home_loc.area_type=='grid') or (p.work_loc.area_type=='grid')):
                    new_sim_pop.append(p)
        self.pop.impact=self.pop.base_sim+new_sim_pop
        print('\t Activity Scheduling')
        self.activity_scheduler.assign_profiles(new_sim_pop)
        for person in new_sim_pop:
            self.activity_scheduler.sample_activity_schedules(person, model=self)
        self.create_trips(new_sim_pop)
        self.predict_trip_modes(new_sim_pop, logit_params=new_logit_params)  
        now=datetime.datetime.now()
        print('Update took {} seconds'.format((now-then).total_seconds()))

    def update_grid(self):
        for grid_index, grid_data in enumerate(self.geogrid_data):
            if self.geogrid.cells[grid_index].updatable:
                self.geogrid.cells[grid_index].set_properties(grid_data['name'],
                                  grid_data['height'], self.geogrid.type_defs)
        
    def create_new_agents(self):
        print('\t Creating new agents')
        new_persons=[]
        new_houses=[]
        for cell in self.geogrid.cells:
            if cell.updatable:
                if cell.naics is not None:
                    num_new_persons=sum([cell.naics[code] for code in cell.naics])/self.scale_factor
                    for i in range(int(num_new_persons)):
                        add_person_record=random.choice(self.pop.base_floating_person_records).copy()
                        # TODO: person attributes based on NAICS
                        add_person=Person(add_person_record, p_id='n'+str(len(new_persons)))
                        add_person.assign_work_location(cell)
                        new_persons.append(add_person)
                new_housing_capacity=0
                for code in cell.lbcs:
                    if code.startswith('11'):
                        new_housing_capacity+=cell.lbcs[code]
                new_housing_capacity=new_housing_capacity/self.scale_factor
                # TODO: number of new housing records to create
                if new_housing_capacity>0:
                    num_new_housing_units=max(1, int(new_housing_capacity/2))
                else:
                    num_new_housing_units=0
                for i in range(num_new_housing_units):
#                    chosen_house_type=random.choices(self.new_house_attributes, [h['p'] for h in self.new_house_attributes], k=1)[0]
                    chosen_house_type=np.random.choice(self.new_house_attributes, 1, [h['p'] for h in self.new_house_attributes])[0]
                    add_house_record=chosen_house_type.copy()
                    add_house_id=len(self.pop.base_vacant)+len(new_houses)
                    add_house=Housing_Unit(add_house_record, house_id=add_house_id)
                    add_house.set_location(cell)
                    new_houses.append(add_house)
        self.pop.add_new_pop(new_persons)
        self.pop.add_new_housing_units(new_houses)
        
    def get_avg_co2(self, persons=None):
        if persons==None:
            persons=self.pop.impact
        total_co2_kg=0
        total_dist=0
        count=0
        for p in persons:
            count+=1
            for trip in p.trips:
                if trip.mode is not None:
                    if trip.total_distance < 1000000:
                        # hack used here because when a route cant be found by some mode
                        # an arbitrality large distance is assumed for purpose of mode choice model
                        mode=trip.mode
                        total_co2_kg+=trip.total_distance*mode.co2_emissions_kg_met
                        total_dist+=trip.total_distance
        return total_co2_kg/count
    
    def get_num_trips(self, persons=None):
        if persons==None:
            persons=self.pop.impact
        count=0
        for p in persons:
            for trip in p.trips:
                if trip.mode is not None:
                    if trip.total_distance<1000000:
                        count+=1 
        return count*self.scale_factor
    
    def get_mode_split(self, persons=None, by_id=False):
        if persons==None:
            persons=self.pop.impact
        count=0
        split={m.name: 0 for m in self.tn.base_modes+self.tn.new_modes}
        for p in persons:
            for trip in p.trips:
                if trip.mode is not None:
                    if trip.total_distance<1000000:
                        split[trip.mode.name]+=1
                        count+=1
#                        split[trip.mode.name]+=trip.total_distance
#                        count+=trip.total_distance
        if by_id:
            prop_split={m.id: split[m.name]/count for m in self.tn.base_modes+self.tn.new_modes}
        else:
            prop_split={mode_name: split[mode_name]/count for mode_name in split}
#        if normalise==False:
#            for mode in prop_split:
#                prop_split[mode]=prop_split[mode]*count*self.scale_factor # scale up to actual trip numbers
        return prop_split
    
    def get_live_work_prop(self, persons=None):
        if persons==None:
            persons=self.pop.impact
        live_work=0
        for p in persons:
            if p.home_loc is not None:
                if ((p.home_loc.area_type=='grid') and (p.work_loc.area_type=='grid')):
                    live_work+=1
        return live_work/len(persons)
    
    def health_impacts(self, ref_rr, ref_quantity, actual_quantity, 
                      min_RR, N,  base_MR= 0.0090421):
        RR=1-ref_rr*(actual_quantity/ref_quantity)
        RR=max(RR, min_RR)
        deltaF=(1-RR)*N*base_MR
        return deltaF
    
    def health_impacts_pp(self, persons=None):
        if persons==None:
            persons=self.pop.impact
        delta_F, count=0, 0
        heat_params={'cycling': {'ref_RR': 0.9, 'ref_mins_week':100, 'min_RR':0.55},
                     'walking': {'ref_RR': 0.89, 'ref_mins_week':168, 'min_RR':0.7}}
        total_mins_per_week={'walking': 0, 'cycling': 0}
        for p in persons:
            count+=1
            for trip in p.trips:
                if trip.mode is not None:
                    if trip.total_distance < 1000000:
                        # hack used here because when a route cant be found by some mode
                        # an arbitrality large distance is assumed for purpose of mode choice model
                        mode_name=trip.mode.name
                        if mode_name not in trip.mode_choice_set: # base mode
                            mode_name=trip.mode.copy_route
                        total_mins_per_week['walking']+=5*trip.mode_choice_set[mode_name].costs['walking']
                        total_mins_per_week['cycling']+=5*trip.mode_choice_set[mode_name].costs['cycling']
        avg_mins_per_week= {mode_name: total_mins_per_week[mode_name]/count for mode_name in total_mins_per_week}
        for mode_name in avg_mins_per_week:
            delta_F+=self.health_impacts(heat_params[mode_name]['ref_RR'], 
                                   heat_params[mode_name]['ref_mins_week'], 
                                   avg_mins_per_week[mode_name], 
                                   heat_params[mode_name]['min_RR'],
                                   count)            
        return delta_F/count
        
        
    
    def get_avg_utility(self, persons=None):
        if persons==None:
            persons=self.pop.impact
        total_v=0
        count=0
        for p in persons:
            for trip in p.trips:
                if trip.mode is not None:
                    if trip.total_distance < 1000000:
                        count+=1
                        # hack used here because when a route cant be found by some mode
                        # an arbitrality large distance is assumed for purpose of mode choice model
                        total_v+=trip.utility
        return total_v/count
    
    def set_new_modes(self, new_alt_specs, nests_spec=None):
        self.mode_choice_model.add_new_alts(new_alt_specs)
        self.tn.set_new_modes(self.mode_choice_model.new_alt_specs)
        if nests_spec is not None:
            self.mode_choice_model.add_nests_spec(nests_spec)
    
    def set_prop_electric_cars(self, prop, co2_emissions_kg_met_ic,co2_emissions_kg_met_ev):
        self.tn.base_modes[0].co2_emissions_kg_met=(1-prop)*co2_emissions_kg_met_ic +prop*co2_emissions_kg_met_ev
        # assume electric cars are half as polluting
        
    
class Population():
    def __init__(self, base_sim_persons, base_floating_persons, base_vacant_housing, model):
        self.base_floating_person_records=base_floating_persons
        self.base_sim=[]
        self.base_floating=[]
        self.base_vacant=[]
        self.new=[]
        self.impact=[]

        for ind_p, person_record in enumerate(base_sim_persons):
            new_person=Person(person_record, p_id='b' + str(ind_p), 
                              worker=((person_record['COW']>0) and (person_record['COW']<9)))
            new_person.get_home_location_from_zone(model=model)
            new_person.get_work_location_from_zone(model=model)
            self.base_sim.append(new_person)
        for ind_p, person_record in enumerate(base_floating_persons):
            new_person=Person(person_record , p_id='f' + str(ind_p), 
                              worker=((person_record['COW']>0) and (person_record['COW']<9)))
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
    def __init__(self, attributes, p_id, worker=True):
        for key in attributes:
            setattr(self, key, attributes[key]) 
        self.person_id=p_id
        self.worker=worker
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
        
    def assign_motif(self, motif):
        self.motif=motif
        
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
                     # TODO: use actual 'pop_per_sqmile_home'
#                     'pop_per_sqmile_home': self.pop_per_sqmile_home,
                     'pop_per_sqmile_home': 1000,
                     'purpose': t.purpose}
            if t.mode_choice_set:
                for m in t.mode_choice_set:
                    this_trip_record[m+'_route']=t.mode_choice_set[m].costs
                    # TODO: dont hard code driving speed
                this_trip_record['network_dist_km']=this_trip_record['driving_route']['driving']*30/60
                this_trip_record['external_network_dist_mile'] = \
                    (t.mode_choice_set['walking'].pre_time+t.mode_choice_set['walking'].post_time)*0.041
                if this_trip_record['network_dist_km']>0:
                    trips_list.append(this_trip_record)
                else:
                    print([t.enters_sim, t.purpose])
        return trips_list
        
        
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
    def attach_geogrid(self, geogrid, geogrid_id, area):
        self.geogrid=geogrid
        self.geogrid_id=geogrid_id
        self.area=area
    def set_initial_state(self, base_land_use, base_height):
        self.base_land_use=base_land_use
        self.base_height=base_height

    def set_properties(self, cs_type, height, cs_type_defs):
        self.type=cs_type
        if ((cs_type is None) or (cs_type=='None')):
            self.naics={}
            self.lbcs={}
        else:
            this_type_def=cs_type_defs[cs_type]
            capacity_per_sqm=1/this_type_def['sqm_pperson']
            if isinstance(height, list):
                height=height[-1]
            capacity=capacity_per_sqm*self.area*height
            self.naics=self.flatten_cs_type_attribute(this_type_def, 'NAICS', capacity)
            self.lbcs=self.flatten_cs_type_attribute(this_type_def, 'LBCS', capacity)
                        
    def flatten_cs_type_attribute(self, type_def, attribute, capacity):
        output={}
        if type_def[attribute] is not None:
            for floor_group in type_def[attribute]:
                capacity_floor=capacity*floor_group['proportion']
                for code in floor_group['use']:
                    capacity_this_floor_this_attr=capacity_floor*floor_group['use'][code]
                    if code in output:
                        output[code]+=capacity_this_floor_this_attr
                    else:
                        output[code]=capacity_this_floor_this_attr
        return output
        
                        
    def set_updatable(self, updatable):
        self.updatable=updatable
        
    def set_interactive(self, interactive):
        self.interactive=interactive
        
class GeoGrid():
    def __init__(self, grid_geojson, transport_network, city_folder):
        MAPPINGS_PATH = './cities/'+city_folder+'/mappings'
        self.activities_to_lbcs=json.load(open(MAPPINGS_PATH+'/activities_to_lbcs.json'))
        self.cells=[]
        self.int_type_defs=grid_geojson['properties']['types'].copy()
        self.type_defs=grid_geojson['properties']['types']
        self.type_defs.update(grid_geojson['properties']['static_types'])
        side_len=grid_geojson['properties']['header']['cellSize']
        area=side_len*side_len
        for ind_f, feature in enumerate(grid_geojson['features']):
            new_grid_cell=GridCell(feature['geometry'], 
                                    area_type='grid',
                                    in_sim_area=True)
            new_grid_cell.set_interactive(feature['properties']['interactive'])
            new_grid_cell.set_updatable(feature['properties']['interactive'] or feature['properties']['static_new'])
            new_grid_cell.attach_geogrid(geogrid=self, geogrid_id=ind_f,area=area)
            new_grid_cell.get_close_nodes(transport_network=transport_network)
            new_grid_cell.set_initial_state(feature['properties']['type'],
                                            feature['properties']['height'])
            new_grid_cell.set_properties(feature['properties']['type'],
                                         feature['properties']['height'], 
                                         self.type_defs)
            self.cells.append(new_grid_cell)
    def find_locations_for_activity(self, activity):
        possible_lbcs=self.activities_to_lbcs[activity]
        possible_cells=[]
        for c in self.cells:
            suitable=False
            for lbcs_stem in possible_lbcs:
                if any(cell_lbcs.startswith(lbcs_stem) for cell_lbcs in c.lbcs):
                    suitable=True
            if suitable:
                possible_cells.append(c)
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
#        mode_name=mode_list[mode].name
        copy_route=mode_list[mode].copy_route
        self.mode=mode_list[mode]
        self.internal_route=self.mode_choice_set[copy_route].internal_route['internal_route']
        self.pre_time=self.mode_choice_set[copy_route].pre_time
        self.post_time=self.mode_choice_set[copy_route].post_time
        self.total_distance=self.internal_route['total_distance'
                                ]+((self.pre_time)+(self.post_time))*60*self.mode.speed_met_s
        if utility is not None:
            self.utility=utility

        
    def to_deckgl_trip_format(self, motif):
        if self.mode is not None:
#            cum_dist=np.cumsum(self.internal_route['distances'])
            route_time_s=[m*60 for m in self.internal_route['minutes']]
            cum_time=np.cumsum(route_time_s)
            internal_trip_start_time=self.activity_start+self.pre_time
            timestamps=[int(internal_trip_start_time)] + [int(internal_trip_start_time)+ int(ct) for ct in cum_time]
            if self.mode.name=='pt':
                trips_objects=self.multi_mode_deck_gl_trip(timestamps, {'pt':3, 'walking':2}, motif)
            else:    
                trips_objects=[{'mode': str(self.mode.id),
                                'profile': str(motif),
                               'path': self.internal_route['coords'],
                               'timestamps': timestamps}]
            
            return trips_objects
        else:
            return None
    
    def multi_mode_deck_gl_trip(self, timestamps, mode_ids, motif):
#        print('multi-modal')
        i=0
        trips_objects=[]
        activities=self.internal_route['activities']
        while i<len(activities):
            j=i
            while ((j+1)<len(activities) and (activities[j]==activities[j+1])):
                j+=1
            trip_part={'mode': str(mode_ids[activities[j]]),
                       'profile': str(motif),
                               'path': self.internal_route['coords'][i:j+2],
                               'timestamps': timestamps[i:j+2]}
            trips_objects.append(trip_part)
            i=j+1
        return trips_objects


            
def main():
    this_model=MobilityModel('corktown', 'Detroit')
    
    this_model.assign_activity_scheduler(ActivityScheduler(model=this_model))
    
    mode_choice_model=NhtsModeLogit(table_name='corktown', city_folder='Detroit')
    
            
    this_model.assign_mode_choice_model(mode_choice_model)
    
    
    this_model.assign_home_location_choice_model(
            TwoStageLogitHLC(table_name='corktown', city_folder='Detroit', 
                             geogrid=this_model.geogrid, base_vacant_houses=this_model.pop.base_vacant))
    
    this_model.init_simulation()

#
if __name__ == '__main__':
	main()  