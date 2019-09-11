import re
import pickle
import json

import sys
from os import path, chdir, makedirs, listdir
chdir(path.dirname(sys.argv[0]))                        #use relative path

import pyproj
from shapely.geometry import Point, shape, GeometryCollection
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPolygon
import geopandas as gpd

# ==========================================
# Some Constants
# ==========================================
amenityHealthList = ['hospital', 'doctors', 'pharmacy', 'baby_hatch', 'clinic', 'dentist', 'nursing_home', 'social_facility', 'veterinary']
amenityRestaurantList = ['restaurant', 'fast_food', 'cafe', 'bar', 'bbq', 'biergarten', 'drinking_water', 'food_court', 'ice_cream', 'pub']
amenitySchoolList = ['school', 'university', 'library', 'college', 'driving_school', 'kindergarten', 'language_school']
amenityTransportList = ['parking', 'bus_station', 'fuel', 'car_sharing', 'bicycle_parking', 'bicycle_repair_station', 'bicycle_rental', 'boat_rental', 
    'boat_sharing', 'car_rental', 'car_wash', 'vehicle_inspection', 'charging_station', 'ferry_terminal', 'grit_bin', 'motorcycle_parking', 
    'parking_entrance', 'parking_space', 'taxi']
amenityFinancialList = ['atm', 'bank', 'bureau_de_change']
amenityEntertainmentList = ['cinema', 'theatre', 'arts_centre', 'brothel', 'casino', 'community_centre', 'fountain', 'gambling', 'nightclub', 'planetarium', 'public_bookcase', 'social_centre', 'stripclub', 'studio', 'swingerclub']
amenityOtherList = ['police', 'toilets', 'post_office', 'post_box', 'post_depot', 'animal_boarding', 'animal_shelter', 'baking_oven', 'bench', 'clock', 'courthouse', 'crematorium', 'dive_centre', 
    'embassy', 'fire_station', 'grave_yard', 'hunting_stand', 'internet_cafe', 'kitchen', 'kneipp_water_cure', 'marketplace', 'monastery', 'photo_booth', 'place_of_worship', 'prison', 'public_bath', 
    'ranger_station', 'recycling', 'sanitary_dump_station', 'shelter', 'shower', 'telephone', '', 'townhall', 'vending_machine', 'waste_basket', 'waste_disposal', 'waste_transfer_station', 'watering_place', 'water_point']


def extract_certain_tag(sourceFile, extractRule, saveFile=None):
    """
    Extract certain kinds of POIs from sourceFile
    
    Arguments:
    ------------------------------------------
    sourceFile: a string indicating the path of the source geojson file containing all kinds of POIs
    extractRule: a dict with key representing tag name and value being a list of tag value, an empty list indicating all values under that tag would be extracted.
    saveFile: the path of the geojson file to be saved in the same folder as sourceFile. Default=None: do not save.
    
    Returns:
    ------------------------------------------
    json dict containing object POIs in ['features'] key.
    """
    d = json.loads(open(sourceFile, 'r', encoding='utf-8').read())
    keepFeatures = []
    for feature in d['features']:
        if 'other_tags' not in feature['properties']:
            continue
        crtOtherTags = feature['properties']['other_tags']
        if crtOtherTags is None:
            continue
        tagList = re.findall('"(.*?)"=>"(.*?)"', crtOtherTags)
        thisTagDict = {}
        for tag in tagList:
            tagName, tagValue = tag
            tagName.replace('"', '')
            tagValue.replace('"', '')
            thisTagDict[tagName] = tagValue
        keepThis = False
        for objTag in extractRule.keys():
            if objTag in thisTagDict:
                if len(extractRule[objTag]) == 0:
                    keepThis = True
                    break
                else:
                    thisTagValue = thisTagDict[objTag]
                    if thisTagValue in extractRule[objTag]:
                        keepThis = True
                        break
        if keepThis:     
            keepFeatures.append(feature)
    d['features'] = keepFeatures
    if saveFile:
        with open(saveFile, 'w', encoding='utf-8') as f:
            json.dump(d, f)
    return d


def point_in_polygon(polygons, points, attrName, polyProjStr='EPSG:4326', pointProjStr='EPSG:4326', savePolygonFile=None, savePointFile=None):
    """
    Spatial join points (POTs) with polygons (PUMAs): 1)filter out the points which do not belong any polygon; 2)add 'PUMA' field to point geojson indicating 
    which PUMA the point is in; 3) add "??_count" and "??_den" fields to polygon geojson indicating the count and density of the points in each polygon
    
    Arguments:
    ---------------------------------------------------------------------------------------
    polygons: a string indicating the path of the geojson file of the polygons (PUMAs), or a dict which is loaded from the polygon geojson file.
    points: a string indicating the path of the geojson file of the points (POIs), or a dict which is loaded from the point geojson file.
    attrName: attribute name when adding "??_count" and "??_den" to polygon geojson.
    polyProjStr / pointProjStr: strings indicating projection specifications for polygons and points
    savePolygonFile / savePointFile: the paths of files to be saved. Default=None: do not save.
    
    Returns:
    -----------------------------------------------------------------------------------------
    polygonsDataOut: a copy of original polygon geojson-data (in dict), with POI-count fields (??_count) and POI-density fields (??_den) added 
    pointsDataOut: a copy of original point geojson-data (in dict), but deleteing points who do not match any polygon. A "PUMA" field is added to indicating which polygon matches the point.
    """
    if isinstance(polygons, str):
        polygonsData = json.loads(open(polygons, 'r', encoding='utf-8').read())
    elif isinstance(polygons, dict):
        polygonsData = polygons
        
    if isinstance(points, str):
        pointsData = json.loads(open(points, 'r', encoding='utf-8').read())
    elif isinstance(points, dict):
        pointsData = points
    
    polygonFeatures, pointFeatures = polygonsData['features'], pointsData['features']
    pointShapes = [shape(feature["geometry"]) for feature in pointFeatures]
    polygonShapes = [shape(feature["geometry"]) for feature in polygonFeatures]

    # add count field 
    for polygon in polygonFeatures:
        polygon['properties'][attrName+'_count']=0     
    keepPointFeatures = [] 

    p1 = pyproj.Proj("+init=" + pointProjStr)
    p2 = pyproj.Proj("+init=" + polyProjStr)
    for point, pointShape in zip(pointFeatures, pointShapes):
        findPUMA = False
        if polyProjStr != pointProjStr:
            x2,y2 = pyproj.transform(p1,p2, pointShape.x, pointShape.y)
            pointShape2 = Point(x2, y2)
        else:
            pointShape2 = pointShape
        
        for polygon, polygonShape in zip(polygonFeatures, polygonShapes):
            if polygonShape.contains(pointShape2):
                point['properties']['PUMA'] = polygon['properties']['PUMACE10']
                polygon['properties'][attrName+'_count'] += 1
                findPUMA = True
                break
        if findPUMA:
            keepPointFeatures.append(point)

    # add density field
    for polygon in polygonFeatures:
        polygon['properties'][attrName+'_den'] = polygon['properties'][attrName+'_count'] / polygon['properties']['ALAND10']

    pointsDataOut, polygonsDataOut = pointsData.copy(), polygonsData.copy()
    pointsDataOut['features'] = keepPointFeatures
    polygonsDataOut['features'] = polygonFeatures

    if savePointFile:
        with open(savePointFile, 'w', encoding='utf-8') as f:
            json.dump(pointsDataOut, f)
    if savePolygonFile:   
        with open(savePolygonFile, 'w', encoding='utf-8') as f:
            json.dump(polygonsDataOut, f)
    
    return polygonsDataOut, pointsDataOut


def process_poi(POIsSourceFile, PUMAsSourceFile, poiConfigure, outFilePUMAsJoinPOIs=None):
    """
    Integrated working flow to process POI data: from raw data to POI densities inside PUMAs
    
    Arguments:
    --------------------------------------------------------------------------------------
    POIsSourceFile: a string indicating the path of the source geojson file containing all kinds of POIs
    PUMAsSourceFile: a string indicating the path of the source PUMA file
    poiConfigure: a dict indicating how different categories of POIs are extracted and summarized
    outFilePUMAsJoinPOIs: the paths of output file to be saved. Default=None: do not save.
    
    Returns:
    ---------------------------------------------------------------------------------------
    PUMAsJointPOIsData: a copy of original polygon geojson-data (in dict), with POI-count fields (??_count) and POI-density fields (??_den) added 
    
    """
    PUMAsJointPOIsData = json.loads(open(PUMAsSourceFile, 'r', encoding='utf-8').read())
    print('')
    for attrName, extractRule in poiConfigure.items():
        print('[info] Extracting "{}" from "{}"'.format(attrName, path.basename(POIsSourceFile)))
        extractedPOIsData = extract_certain_tag(POIsSourceFile, extractRule)
        print('[info] Spatial joining "{}" to "{}"'.format(attrName, path.basename(PUMAsSourceFile)))
        PUMAsJointPOIsData, tmp = point_in_polygon(PUMAsJointPOIsData, extractedPOIsData, attrName)
    if outFilePUMAsJoinPOIs:
        with open(outFilePUMAsJoinPOIs, 'w', encoding='utf-8') as f:
            json.dump(PUMAsJointPOIsData, f)
    return PUMAsJointPOIsData
