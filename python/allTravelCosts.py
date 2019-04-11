#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:51:25 2018

@author: doorleyr
"""
    
import json 
from time import sleep
import urllib.request
from shapely.geometry import shape

def getOSRMDirections(mode, startLat, startLon, endLat, endLon):
    strLL=str(startLon) + ','+str(startLat)+';'+str(endLon)+ ','+str(endLat)
    try:
        with urllib.request.urlopen('http://router.project-osrm.org/route/v1/'+str(mode)+'/'+strLL+'?geometries=geojson') as url:
            data=json.loads(url.read().decode())
            #in meters and seconds
        return data['routes'][0]['geometry']['coordinates']
        # if the call request is unsuccessful, wait and try again
    except:
        print('Sleeping')
        sleep(1)
        coordinates=getOSRMDirections(mode, startLat, startLon, endLat, endLon)
        return coordinates

 
# zones shape file
city='Hamburg'
CLEAN_ZONES_PATH='./'+city+'/clean/zones.geojson'
ZONE_ROUTES_PATH='./'+city+'/clean/routes.json'
CONNECTION_ROUTES_PATH='./'+city+'/clean/connections.json'
CONNECTION_POINTS_PATH='./'+city+'/clean/connection_points.json'


# get the centroid of each geoId
zones=json.load(open(CLEAN_ZONES_PATH))
lon_lat_list= [[shape(f['geometry']).centroid.x, shape(f['geometry']).centroid.y] for f in zones['features']]   
# initialise routes
routes={fromGeoId:{toGeoId:{} for toGeoId in range(len(lon_lat_list))} for fromGeoId in range(len(lon_lat_list))}

for fromGeoId in range(len(lon_lat_list)):
    print(fromGeoId)
    for toGeoId in range(len(lon_lat_list)):
#        routes[fromGeoId][toGeoId] ={'coordinates': []}
        if fromGeoId==toGeoId:
            routes[fromGeoId][toGeoId]['coordinates']=[lon_lat_list[fromGeoId]]
        else:
            #get driving directions
            startLat, startLon, endLat, endLon=[lon_lat_list[fromGeoId][1], lon_lat_list[fromGeoId][0],
                                                lon_lat_list[toGeoId][1], lon_lat_list[toGeoId][0]]
            coordinates=getOSRMDirections('driving', startLat, startLon, endLat, endLon)
            routes[fromGeoId][toGeoId]['coordinates']=coordinates

# get the routes between the zones and the connection points
connection_points=json.load(open(CONNECTION_POINTS_PATH))
route_to_grid={}
route_from_grid={}
for fromGeoId in range(len(lon_lat_list)):
    print(fromGeoId)
    route_to_grid[fromGeoId]={}
    for cp in range(len(connection_points)):
        startLat, startLon, endLat, endLon=[lon_lat_list[fromGeoId][1], lon_lat_list[fromGeoId][0],
                                                connection_points[cp]['lat'], connection_points[cp]['lon']]
        coordinates=getOSRMDirections('driving', startLat, startLon, endLat, endLon)
        route_to_grid[fromGeoId][cp]={}
        route_to_grid[fromGeoId][cp]['coordinates']=coordinates
    
for cp in range(len(connection_points)):
    route_from_grid[cp]={}
    for toGeoId in range(len(lon_lat_list)):
        print(toGeoId)
        route_from_grid[cp][toGeoId]={}
        startLat, startLon, endLat, endLon=[connection_points[cp]['lat'], connection_points[cp]['lon'],
                                            lon_lat_list[toGeoId][1], lon_lat_list[toGeoId][0]]
        coordinates=getOSRMDirections('driving', startLat, startLon, endLat, endLon)
        route_from_grid[cp][toGeoId]['coordinates']=coordinates

connections={'route_to_grid': route_to_grid, 'route_from_grid': route_from_grid} 
       
json.dump(routes, open(ZONE_ROUTES_PATH, 'w'))    
json.dump(connections, open(CONNECTION_ROUTES_PATH, 'w'))         
