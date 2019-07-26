# -*- coding: utf-8 -*-

'''
This program will take a geojson file of POIs and another geojson file of PUMAs, and then match the POIs to PUMAs, including: (1) add a attribute to POI
file indicating which PUMA it belongs to; (2) delete the POI which do not lay inside any PUMA, this is normal as we usually use a very rough boundary when 
requesting POIs from OSM; (3) add two attributes to PUMA file: the count and density of POIs in each PUMA.

usage: 
The program requests 3 argument:
    The first argument is the name (without extension) of PUMA geojson file.
    The second argument is the name (without extension) of POI geojson file, which is generally derived using "extract_tags_from_OSM.py".
    The third and argument is the name to describe the POIs and then be added (with "_count" and "_den") as PUMA attribute. 

e.g.:
the cmd command of [python "**\point_in_polygon.py" michiganPUMA michigan_poi_amenity_hospital_doctors_pharmacy medical]
will match the POIs in "michigan_poi_amenity_hospital_doctors_pharmacy.geojson" and PUMAs in "michiganPUMA.geojson". These POIs are descirbed as "medical", 
thus two new properties, "medical_count" and "medical_den", would be added to "michiganPUMA.geojson" as count and density of POIs for each PUMA.

'''

import json
import sys
from os import path, chdir, makedirs, listdir
import matplotlib.pyplot as plt
#use relative path
chdir(path.dirname(sys.argv[0]))

import pyproj
from shapely.geometry import Point, shape, GeometryCollection
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon

polygonFileName = sys.argv[1] + '.geojson'
pointFileName = sys.argv[2] + '.geojson'
saveFieldName = sys.argv[3]


polygonFileFullPath = r'./' + polygonFileName
pointFileFullPath = r'./' + pointFileName

polygonsData = json.loads(open(polygonFileFullPath, 'r', encoding='utf-8').read())
pointsData = json.loads(open(pointFileFullPath, 'r', encoding='utf-8').read())

polygonFeatures, pointFeatures = polygonsData['features'], pointsData['features']

pointShapes = [shape(feature["geometry"]) for feature in pointFeatures]
polygonShapes = [shape(feature["geometry"]) for feature in polygonFeatures]

# add count field 
for polygon in polygonFeatures:
    polygon['properties'][saveFieldName+'_count']=0


keepPointFeatures = [] 
p0 = pyproj.Proj("+init=EPSG:4326")         #coord system of POIs from OSM
p2 = pyproj.Proj("+init=ESRI:102003")       #coord system of PUMAs from a shapefile
for point, pointShape in zip(pointFeatures, pointShapes):
    # initialize "findPUMA" to False and iterate all PUMAs, if any PUMA matched, set it to True and break the iteration.
    findPUMA = False
    x0,y0 = pointShape.x, pointShape.y
    x2,y2 = pyproj.transform(p0,p2, x0, y0)
    pointShape2 = Point(x2, y2)
    for polygon, polygonShape in zip(polygonFeatures, polygonShapes):
        if polygonShape.contains(pointShape2):
            point['properties']['PUMA'] = polygon['properties']['PUMA']
            polygon['properties'][saveFieldName+'_count'] += 1
            findPUMA = True
            break
    if findPUMA:
        keepPointFeatures.append(point)


# add density field
for polygon in polygonFeatures:
    polygon['properties'][saveFieldName+'_den'] = polygon['properties'][saveFieldName+'_count'] / polygon['properties']['ALAND']

pointsDataOut, polygonsDataOut = pointsData.copy(), polygonsData.copy()
pointsDataOut['features'] = keepPointFeatures
polygonsDataOut['features'] = polygonFeatures


with open(pointFileFullPath, 'w', encoding='utf-8') as f:
    json.dump(pointsDataOut, f)
    
with open(polygonFileFullPath, 'w', encoding='utf-8') as f:
    json.dump(polygonsDataOut, f)