# -*- coding: utf-8 -*-

'''
This program will extract certain points with objectve tags(=values) from all OSM points.

usage: 
The program requests 2 or more argument:
    The first argument is the name (without extension) of geojson file of OSM points.
    The second argument is the objective tag name.
    The third and following argument(s) would form a list of objective value(s) of the tag. If not provided, the program will extract all points with 
        the objective tag included in "other_tags" attribute; if provided, the program will extract all points with the objective tag included in 
        "other_tags" attribute and its value in the list of objective values(s).
    
    The extracted points would be saved as a geojson file with suffix of "objective_tag_name+[first 3 objective_value_names]" 

e.g.:
the cmd command of [python "***\extract_tags_from_OSM.py" michigan_poi amenity]
will extract all amenities from "michigan_poi.geojson" and save it as "michigan_poi_amenity.geojson"

the cmd command of 
[python "***\extract_tags_from_OSM.py" michigan_poi amenity hospital doctors pharmacy baby_hatch clinic dentist nursing_home social_facility veterinary]
will extract all medical amenties (hospital, doctors, pharmacy, baby_hatch, clinic,...) from "michigan_poi.geojson" and save it as 
"michigan_poi_amenity_hospital_doctors_pharmacy.geojson"
'''

import re
import pickle
import json
import sys
from os import path, chdir, makedirs, listdir
#use relative path
chdir(path.dirname(sys.argv[0]))


fileName = sys.argv[1] + '.geojson'
objTag = sys.argv[2]
if len(sys.argv)>=4:
    objTagValueList = sys.argv[3:]
    outFileName = fileName.split('.')[0] + '_' + objTag + '_' + '_'.join(objTagValueList[:3]) + '.geojson'
else:
    objTagValueList = None
    outFileName = fileName.split('.')[0] + '_' + objTag + '.geojson'

inFileFullPath = r'./' + fileName
outFileFullPath = r'./' + outFileName

d = json.loads(open(inFileFullPath, 'r', encoding='utf-8').read())
oldFeatures = d['features']

# do with the "other_tags" attribute
keepFeatures = []
for feature in oldFeatures:
    if 'other_tags' not in feature['properties']:
        continue
    crtOtherTags = feature['properties']['other_tags']
    if crtOtherTags is None:
        continue
    tagList = re.findall('"(.*?)"=>"(.*?)"', crtOtherTags)
    tagDict = {}
    for tag in tagList:
        tagName, tagValue = tag
        tagName = ''.join([x for x in tagName if x != '"'])
        tagValue = ''.join([x for x in tagValue if x != '"'])
        tagDict[tagName] = tagValue
    if objTag not in list(tagDict.keys()):
        continue
    finalTagValue = tagDict[objTag]
    if objTagValueList is not None and finalTagValue not in objTagValueList:
        continue
    feature['properties'][objTag] = finalTagValue
    keepFeatures.append(feature)

dout = d.copy()
dout['features'] = keepFeatures
with open(outFileFullPath, 'w', encoding='utf-8') as f:
    json.dump(dout, f)