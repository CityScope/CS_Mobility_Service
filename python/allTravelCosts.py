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
        sleep(5)
        data=getOSRMDirections(mode, startLat, startLon, endLat, endLon)
        return data['routes']

 
# zones shape file
city='Hamburg'
CLEAN_ZONES_PATH='./Hamburg/clean/zones.geojson'
ZONE_ROUTES_PATH='./'+city+'/clean/routes.json'

# get the centroid of each geoId
zones=json.load(open(CLEAN_ZONES_PATH))
lon_lat_list= [[shape(f['geometry']).centroid.x, shape(f['geometry']).centroid.y] for f in zones['features']]   


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
json.dump(routes, open(ZONE_ROUTES_PATH, 'w'))            
