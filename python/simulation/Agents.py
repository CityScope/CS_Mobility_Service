#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:11:12 2019

@author: doorleyr
"""
import random
import numpy as np

class Location:
    def __init__(self, graph_id, term_node, cp_routes, centroid, zone_id):
        self.centroid=centroid
        self.zone_id=zone_id
        self.graph_id=graph_id
        self.term_node=term_node
        self.cp_routes=cp_routes
        
class House:
    def __init__(self, rent, puma_pop_per_sqmeter, puma_med_income, beds, year_built, location, house_id, household_id):
        self.house_id=house_id
        self.beds=int(beds)
        self.built_since_jan2010=year_built>=14
        self.household_id=household_id
        self.rent=rent
        # TODO: puma_pop_per_sqmeter should come from property of Location
        self.puma_pop_per_sqmeter=puma_pop_per_sqmeter
        self.puma_med_income=puma_med_income
        self.location=location
    def long_data_record(self, hh_income, person_id, choice_id, rent_normalisation):
        beds=min(3, max(1, self.beds))
        norm_rent=(self.rent-rent_normalisation['mean'][str(beds)])/rent_normalisation['std'][str(beds)]
        return {'norm_rent': norm_rent,
                'puma_pop_per_sqmeter': self.puma_pop_per_sqmeter,
                'income_disparity': np.abs(self.puma_med_income-hh_income),
                'built_since_jan2010': self.built_since_jan2010,
                'custom_id': person_id,
                'choice_id': choice_id,
                'actual_house_id':self.house_id}
        
class Household:
    def __init__(self,  n_vehicles, n_workers, children, hh_income, hh_income_cat, tenure, house_id, household_id):
        self.n_vehicles=n_vehicles
        self.n_workers=n_workers
        self.children=children
        self.hh_income=hh_income
        self.hh_income_cat=hh_income_cat
        self.house_id=house_id
        self.household_id=household_id
        self.tenure=tenure
    def spawn_person(self, person_pop,  person_id, routes, node_coords, work_loc):
        # TODO: sample from the same zone
        sample_row=person_pop.sample(n=1).squeeze()
        return Person(sample_row['age'], sample_row['SCHL'], self.hh_income_cat, self.household_id, 
                      work_loc, sample_row['sex'], 
                     10000, person_id, routes, node_coords, 
                     self.children,  self.n_workers, self.tenure)
        
        
class Person:
    def __init__(self,age, school_level, hh_income_cat, household_id, work_loc, sex, 
                  pop_per_sqmile_home, person_id, routes, node_coords, 
                 children, workers, tenure):
        self.person_id=person_id
        self.age=age
        self.sex=sex
        self.workers=workers
        self.tenure=tenure
        self.school_level=school_level
        self.children=children
        self.income=hh_income_cat
        self.household_id=household_id
        self.work_loc=work_loc
        self.pop_per_sqmile_home=pop_per_sqmile_home
    def set_home_loc(self, loc):
        self.home_loc=loc
        
    def init_routes(self, routes, node_coords, home_loc):
        if home_loc.graph_id==self.work_loc.graph_id:            
            self.all_routes=[routes[home_loc.graph_id][str(home_loc.term_node)][str(self.work_loc.term_node)].copy(),
                             routes[home_loc.graph_id][str(self.work_loc.term_node)][str(home_loc.term_node)].copy()]
            for r in self.all_routes:
                r['coordinates']=[node_coords[home_loc.graph_id][n].copy() for n in r['nodes']]
        else:
#            TODO: connector links should not be 100 long
            self.all_routes=[{'nodes': home_loc.cp_routes['to'][0]['nodes']+self.work_loc.cp_routes['from'][0]['nodes'].copy(),
                            'distances':home_loc.cp_routes['to'][0]['distances']+[100]+self.work_loc.cp_routes['from'][0]['distances'].copy(),
                            'coordinates': [node_coords[home_loc.graph_id][n].copy() for n in home_loc.cp_routes['to'][0]['nodes']]+
                            [node_coords[self.work_loc.graph_id][n].copy() for n in self.work_loc.cp_routes['from'][0]['nodes']]},
                            {'nodes': self.work_loc.cp_routes['to'][0]['nodes']+home_loc.cp_routes['from'][0]['nodes'].copy(),
                            'distances':self.work_loc.cp_routes['to'][0]['distances']+[100]+home_loc.cp_routes['from'][0]['distances'].copy(),
                            'coordinates': [node_coords[self.work_loc.graph_id][n].copy() for n in self.work_loc.cp_routes['to'][0]['nodes']]+
                            [node_coords[home_loc.graph_id][n].copy() for n in home_loc.cp_routes['from'][0]['nodes']]}]
        for ri in range(len(self.all_routes)):
            self.all_routes[ri]['cumdist']=[0]+list(np.cumsum(self.all_routes[ri]['distances']))
                


    def init_period(self, p, TIMESTEP_SEC):
        self.route=self.all_routes[p%len(self.all_routes)]
        self.network_dist_km=sum(self.route['distances'])/1000
        self.mode=None
        self.speed=None
        self.position=self.route['coordinates'][0].copy()
        self.next_node_index=1
        self.start_time=random.choice(range(int(200/TIMESTEP_SEC)))
        if len(self.route['coordinates'])>1: 
            self.next_node_ll=self.route['coordinates'][1].copy()
            self.finished=False
            self.prop_of_link_left=1
        else: 
            self.next_node_ll=self.route['coordinates'][0].copy()
            self.finished=True
            self.prop_of_link_left=0  

    def set_mode(self, mode, speeds):
        self.mode=mode
        speed_mid=speeds[mode]
        self.speed=random.triangular(0.7*speed_mid, 1.3*speed_mid, speed_mid)
        start_time=random.randint(0,500)
        self.route['timestamps']=[start_time+int(100*cd/(self.speed/3.6))/100 for cd in self.route['cumdist']]
        
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