#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:21:11 2019

@author: doorleyr
"""
import urllib.request
import json
from shapely.geometry import shape


city='Hamburg'
CLEAN_SHP_PATH='../ABM/includes/'+city+'/zones.geojson'
SMALL_AREA_SHP_PATH='./'+city+'/raw/small_area.geojson'
OVERPASS_NODES_ROOT='https://lz4.overpass-api.de/api/interpreter?data=[out:json][bbox];node;out;&bbox='
OVERPASS_LINKS_ROOT='https://lz4.overpass-api.de/api/interpreter?data=[out:json][bbox];way[~"^(highway)$"~"."];out;&bbox='
CLEAN_NETWORK_PATH='../ABM/includes/'+city+'/network_osm.geojson'
# link types to include for the large area
ROAD_TYPES_LARGE= ['motorway', 'trunk', 'primary', 
             'motorway_link', 'trunk_link', 'primary_link',
             'secondary','secondary_link',
            ]
# link types to add for the small area
ADD_ROAD_TYPES_SMALL=[
              
             'tertiary', 'tertiary_link','unclassified' 
            ]

# =============================================================================
# get the json links and nodes
# =============================================================================
# open geojson of the large area
# TODO: avoid repetition in this block and next

zones_shp=json.load(open(CLEAN_SHP_PATH))
largeArea=[shape(f['geometry']) for f in zones_shp['features']]
bounds=[shp.bounds for shp in largeArea]
boundsAll=[min([b[0] for b in bounds]), #W
               min([b[1] for b in bounds]), #S
               max([b[2] for b in bounds]), #E
               max([b[3] for b in bounds])] #N
# get BB of the area as string
strBounds=str(boundsAll[0])+','+str(boundsAll[1])+','+str(boundsAll[2])+','+str(boundsAll[3])
# request data from Overpass API
node_url=OVERPASS_NODES_ROOT+strBounds
link_url=OVERPASS_LINKS_ROOT+strBounds
with urllib.request.urlopen(node_url) as url:
    node_data=json.loads(url.read().decode())
with urllib.request.urlopen(link_url) as url:
    link_data=json.loads(url.read().decode())

small_area_shp=json.load(open(SMALL_AREA_SHP_PATH))
smallArea=[shape(f['geometry']) for f in small_area_shp['features']]
bounds=[shp.bounds for shp in smallArea]
bounds_small=[min([b[0] for b in bounds]), #W
               min([b[1] for b in bounds]), #S
               max([b[2] for b in bounds]), #E
               max([b[3] for b in bounds])] #N
# get BB of the area as string
strBounds=str(bounds_small[0])+','+str(bounds_small[1])+','+str(bounds_small[2])+','+str(bounds_small[3])
# request data from Overpass API
link_url=OVERPASS_LINKS_ROOT+strBounds
with urllib.request.urlopen(link_url) as url:
    link_data_small=json.loads(url.read().decode())

# =============================================================================
# subset links by road types for each area with corresponding level of detail
# =============================================================================
print('Large Area')
node_id_order=[n['id'] for n in node_data['elements']]
features=[]
id_num=0
count=0
for l in link_data['elements']:
    if count%1000==0: print(count)
    count+=1
    road_type=l['tags']['highway']
    if (road_type in ROAD_TYPES_LARGE):            
        coordinates=[]
        for n in l['nodes']:
            try:
                node_index=node_id_order.index(n)
                ll=[node_data['elements'][node_index]['lon'] ,  node_data['elements'][node_index]['lat']] 
                coordinates.append(ll)
            except:
                print('couldnt find node '+str(n))
        if len(coordinates)>1:
            geometry={"type": "LineString",
                     "coordinates": coordinates
                    }
            feature={"type": "Feature",
                     "id": id_num,
                     "geometry":geometry,
                     "properties": {"type":road_type}
                     }
            features.append(feature)
            id_num+=1

print('Small Area')
count=0
for l in link_data_small['elements']:
    if count%1000==0: print(count)
    count+=1
    road_type=l['tags']['highway']
    if road_type in ADD_ROAD_TYPES_SMALL:
        coordinates=[]
        for n in l['nodes']:
            try:
                node_index=node_id_order.index(n)
                ll=[node_data['elements'][node_index]['lon'] ,  node_data['elements'][node_index]['lat']] 
                coordinates.append(ll)
            except:
                print('couldnt find node '+str(n))
        if len(coordinates)>1:
            geometry={"type": "LineString",
                     "coordinates": coordinates
                    }
            feature={"type": "Feature",
                     "id": id_num,
                     "geometry":geometry,
                     "properties": {"type":road_type}
                     }
            features.append(feature)
            id_num+=1        
# =============================================================================
# build as geojson
# =============================================================================
geojson_object={
  "type": "FeatureCollection",
  "crs": {
    "type": "name",
    "properties": {
      "name": "EPSG:4326"
    }
  },
  "features": features
}
     

json.dump(geojson_object, open(CLEAN_NETWORK_PATH, 'w'))