#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:11:12 2019

@author: doorleyr
"""
import random
import numpy as np

class Location:
    def __init__(self, graph_id, term_node, cp_routes, radius):
        self.radius=radius
        self.graph_id=graph_id
        self.term_node=term_node
        self.cp_routes=cp_routes
        
class House:
    def __init__(self, rent, puma_pop, puma_med_income, location, house_id):
        self.house_id=house_id
        self.rent=rent
        self.puma_pop=puma_pop
        self.puma_med_income=puma_med_income
        self.location=location
        self.vacant=True
        self.occupier=None
    def long_data_record(self, person_income, person_id, choice_id):
        return {'rent': self.rent,
                'population': self.puma_pop,
                'income_disparity': np.abs(self.puma_med_income-person_income),
                'custom_id': person_id,
                'choice_id': choice_id,
                'actual_house_id':self.house_id}
        
class Person:
    def __init__(self,age, bachelor_degree, hh_income, home_loc, work_loc, male, 
                 motif, pop_per_sqmile_home, person_id, routes, node_coords):
        self.person_id=person_id
        self.age=age
        self.bachelor_degree=bachelor_degree
        self.hh_income=hh_income
        self.home_loc=home_loc
        self.work_loc=work_loc
        self.male=male
        self.motif=motif
        self.pop_per_sqmile_home=pop_per_sqmile_home
        
    def init_routes(self, routes, node_coords):
        if self.home_loc.graph_id==self.work_loc.graph_id:            
            self.all_routes=[routes[self.home_loc.graph_id][str(self.home_loc.term_node)][str(self.work_loc.term_node)].copy(),
                             routes[self.home_loc.graph_id][str(self.work_loc.term_node)][str(self.home_loc.term_node)].copy()]
            for r in self.all_routes:
                r['coordinates']=[node_coords[self.home_loc.graph_id][n].copy() for n in r['nodes']]
        else:
#            TODO: connector links should not be 100 long
            self.all_routes=[{'nodes': self.home_loc.cp_routes['to'][0]['nodes']+self.work_loc.cp_routes['from'][0]['nodes'].copy(),
                            'distances':self.home_loc.cp_routes['to'][0]['distances']+[100]+self.work_loc.cp_routes['from'][0]['distances'].copy(),
                            'coordinates': [node_coords[self.home_loc.graph_id][n].copy() for n in self.home_loc.cp_routes['to'][0]['nodes']]+
                            [node_coords[self.work_loc.graph_id][n].copy() for n in self.work_loc.cp_routes['from'][0]['nodes']]},
                            {'nodes': self.work_loc.cp_routes['to'][0]['nodes']+self.home_loc.cp_routes['from'][0]['nodes'].copy(),
                            'distances':self.work_loc.cp_routes['to'][0]['distances']+[100]+self.home_loc.cp_routes['from'][0]['distances'].copy(),
                            'coordinates': [node_coords[self.work_loc.graph_id][n].copy() for n in self.work_loc.cp_routes['to'][0]['nodes']]+
                            [node_coords[self.home_loc.graph_id][n].copy() for n in self.home_loc.cp_routes['from'][0]['nodes']]}]


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