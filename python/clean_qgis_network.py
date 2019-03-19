#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:21:11 2019

@author: doorleyr
"""
import json
city='New York'
QGIS_NETWORK_PATH=city+'/raw/network_from_QGIS.geojson'
CLEAN_NETWORK_PATH='../ABM/includes/'+city+'/network.geojson'
ROAD_TYPES= ['motorway', 'trunk', 'primary', 'secondary'
             , 'tertiary', 'unclassified',
             'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 
             'tertiary_link'
            ]
network=json.load(open(QGIS_NETWORK_PATH))
network['features']=[network['features'][i] for i in range(len(network['features'])) 
                    if network['features'][i]['properties']['highway'] in ROAD_TYPES]
network['crs']['properties']['name']='EPSG:4326'

json.dump(network, open(CLEAN_NETWORK_PATH, 'w'))