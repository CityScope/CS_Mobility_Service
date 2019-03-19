#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 13:29:11 2018

@author: doorleyr
"""                   
import json  
import os                 
import pickle
import urllib.request
from shapely.geometry import shape
# =============================================================================
# Constants
# =============================================================================
city='New York'
CLEAN_SHP_PATH=city+'/clean/zones.geojson'
AMENITIES_PATH='../ABM/includes/'+city+'/amenities.geojson'
OSM_URL_ROOT='https://lz4.overpass-api.de/api/interpreter?data=[out:json][bbox];node[~"^(amenity|leisure|shop)$"~"."];out;&bbox='

tags={
      'food': ['amenity_restaurant', 'amenity_cafe' 'amenity_fast_food', 'amenity_pub'],
      'nightlife': ['amenity_bar' , 'amenity_pub' , 'amenity_nightclub', 'amenity_biergarten'],  #(according to OSM, pubs may provide food, bars dont)
      'groceries': ['shop_convenience', 'shop_grocer', 'shop_greengrocer', 'shop_food', 'shop_supermarket'], 
#      'education': ['amenity_school', 'amenity_university', 'amenity_college']
      }


# =============================================================================
# get all amenities in within bounding box of study area
# =============================================================================
tracts=json.load(open(CLEAN_SHP_PATH))
area=[shape(f['geometry']) for f in tracts['features']]
bounds=[shp.bounds for shp in area]
bounds_all=[min([b[0] for b in bounds]), #W
               min([b[1] for b in bounds]), #S
               max([b[2] for b in bounds]), #E
               max([b[3] for b in bounds])] #N
# To get all amenity data
str_bounds=str(bounds_all[0])+','+str(bounds_all[1])+','+str(bounds_all[2])+','+str(bounds_all[3])
osm_url_bbox=OSM_URL_ROOT+str_bounds
with urllib.request.urlopen(osm_url_bbox) as url:
    data=json.loads(url.read().decode())

# =============================================================================
# Create a geojson file including only the amenity types we're interested in
# include each possible tag as a property
# =============================================================================
features=[]
for a in range(len(data['elements'])):
    include=0
    for t in tags:
        data['elements'][a][t]=0
        for recordTag in list(data['elements'][a]['tags'].items()):
            if recordTag[0] +'_'+recordTag[1] in tags[t]:
                data['elements'][a][t]=1
                include=1
    if include==1:
        feature={"type": "Feature",
                 "geometry": {"type": "Point","coordinates": [data['elements'][a]['lon'], data['elements'][a]['lat']]},
                 "properties": {t: data['elements'][a][t] for t in tags}}
        feature["properties"]['osm_id']=data['elements'][a]['id']
        features.append(feature)
output={"crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
         "type": "FeatureCollection","features":features} 

json.dump(output, open(AMENITIES_PATH, "w"))