import urllib.request
import numpy as np
import pyproj
import json
from shapely.geometry import shape
import pandas as pd


def get_distance(nodes, node_dict):
    distance=0
    for i in range(len(nodes)-1):
        distance+=np.sqrt((node_dict[nodes[i]]['x']-node_dict[nodes[i+1]]['x'])**2 + 
                          (node_dict[nodes[i]]['y']-node_dict[nodes[i+1]]['y'])**2)
    return distance

# ******************************************************************#
# ******************** Constants ***********************************#
# ******************************************************************#
CLEAN_SHP_PATH='clean/zones.geojson'
OVERPASS_NODES_ROOT='https://lz4.overpass-api.de/api/interpreter?data=[out:json][bbox];node;out;&bbox='
OVERPASS_LINKS_ROOT='https://lz4.overpass-api.de/api/interpreter?data=[out:json][bbox];way[~"^(highway)$"~"."];out;&bbox='
NETWORK_PATH='clean/network.csv'
NODES_LL_PATH='clean/nodes_lon_lat.json'
ROAD_TYPES= ['motorway', 'trunk', 'primary', 'secondary'
#             , 'tertiary', 'unclassified',
             'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 
#             'tertiary_link'
            ]
LANE_CAPACITY=980.0
WGS_84=pyproj.Proj("+init=EPSG:4326")
UTM=pyproj.Proj("+init=EPSG:32619") #19N

# ******************************************************************#
# ******************** Get the OSM data ****************************#
# ******************************************************************#
# open geojson of the area
zones_shp=json.load(open(CLEAN_SHP_PATH))
GBAarea=[shape(f['geometry']) for f in zones_shp['features']]
bounds=[shp.bounds for shp in GBAarea]
boundsAll=[min([b[0] for b in bounds]), #W
               min([b[1] for b in bounds]), #S
               max([b[2] for b in bounds]), #E
               max([b[3] for b in bounds])] #N
# get BB of the area
# request data from Overpass API
strBounds=str(boundsAll[0])+','+str(boundsAll[1])+','+str(boundsAll[2])+','+str(boundsAll[3])
node_url=OVERPASS_NODES_ROOT+strBounds
link_url=OVERPASS_LINKS_ROOT+strBounds
with urllib.request.urlopen(node_url) as url:
    node_data=json.loads(url.read().decode())
with urllib.request.urlopen(link_url) as url:
    link_data=json.loads(url.read().decode())

                
# ******************************************************************#
# ******************** Parse the OSM network data*******************#
# ******************************************************************#
node_dict={}
for e in node_data['elements']:
    node_id=e['id']
    node_dict[node_id]={}
    node_dict[node_id]['lon']=e['lon']
    node_dict[node_id]['lat']=e['lat']
    node_dict[node_id]['x'], node_dict[node_id]['y']=pyproj.transform(WGS_84, UTM, e['lon'], e['lat'])
    
links=[]
link_num=0
for e in link_data['elements']:
    if e['tags']['highway'] in ROAD_TYPES and all(n in node_dict for n in e['nodes']):
        # there can be some links at the edges of the boundary which include nodes that are outside the boundary
        this_link={'id':link_num,
                   'osm_id': e['id'],
                   'a_node':e['nodes'][0],
                   'b_node':e['nodes'][-1]}
        for t in ['maxspeed', 'lanes']:
            if t in e['tags']:
                this_link[t]=e['tags'][t]
            else:
                this_link[t]=float('nan') 
        this_link['length']=get_distance(e['nodes'], node_dict)
        links.append(this_link)
        link_num+=1
        if 'oneway' in e['tags']:
            if e['tags']['oneway']=='no':
                #add a similar link in opposite direction
                rev_link={'id':link_num,
                          'osm_id': e['id'],
                          'a_node':e['nodes'][-1],
                          'b_node':e['nodes'][0]}
                for t in ['maxspeed', 'lanes', 'length']:
                    rev_link[t]=this_link[t]
                links.append(rev_link)
# ******************************************************************#
# ******************** Clean up and create link and node tables*****#
# ******************************************************************#
# renumber nodes from 0 to num(nodes)
nodes_included=list(set([l['a_node'] for l in links]+[l['b_node'] for l in links]))
node_num_dict=dict(zip(nodes_included, range(len(nodes_included))))
nodes_ll=[]
for n in node_num_dict:
    nodes_ll.append([int(node_dict[n]['lon']*10e6)/10e6,  int(node_dict[n]['lat']*10e6)/10e6])

net=pd.DataFrame(links)
#fix the node numbers
net['a_node']=net.apply(lambda row: node_num_dict[row['a_node']], axis=1)
net['b_node']=net.apply(lambda row: node_num_dict[row['b_node']], axis=1)
# TODO fix the speeds
net['maxspeed']=50
#net.loc[~net['maxspeed'].isnull(),'maxspeed']=net.loc[~net['maxspeed'].isnull()].apply(lambda row: row['maxspeed'].split(' ')[0], axis=1)
#net.loc[net['maxspeed'].isnull(), 'maxspeed']=net['maxspeed'].mean()
net['drive_time']=net.apply(lambda row: row['length']/row['maxspeed'], axis=1)
# find node closest to centroid of each zone and add to shape file 
net.to_csv(NETWORK_PATH)
json.dump(nodes_ll, open(NODES_LL_PATH, 'w'))


       