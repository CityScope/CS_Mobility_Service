#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:21:11 2019

@author: doorleyr
"""
import osmnet
import json
from shapely.geometry import shape
import pandas as pd
import networkx as nx
from shapely.geometry import shape
from scipy import spatial
import numpy as np

city='Hamburg'

CLEAN_ZONES_PATH='./'+city+'/clean/zones.geojson'
ZONE_NODE_ROUTES_PATH='./'+city+'/clean/route_nodes.json'
CONNECTION_NODE_ROUTES_PATH='./'+city+'/clean/connection_route_nodes.json'
CONNECTION_POINTS_PATH='./'+city+'/clean/connection_points.json'
NODES_PATH='./'+city+'/clean/nodes.csv'

# set the bounding box to include all zones
zones_shp=json.load(open(CLEAN_ZONES_PATH))
largeArea=[shape(f['geometry']) for f in zones_shp['features']]
bounds=[shp.bounds for shp in largeArea]
boundsAll=[min([b[0] for b in bounds]), #W
               min([b[1] for b in bounds]), #S
               max([b[2] for b in bounds]), #E
               max([b[3] for b in bounds])] #N

nodes,edges=osmnet.load.network_from_bbox(lat_min=boundsAll[1], lng_min=boundsAll[0], lat_max=boundsAll[3], 
                              lng_max=boundsAll[2], bbox=None, network_type='drive', 
                              two_way=True, timeout=180, 
                              custom_osm_filter=None)

#edges=edges[edges['highway']!='residential']

edges=edges.reset_index()

        
node_id_map={nodes.iloc[i]['id']:i for i in range(len(nodes))}
nodes=nodes.rename(columns={'id':'osm_id', 'x': 'lon', 'y': 'lat'})
nodes['id']=range(len(nodes))

edges=edges.rename(columns={'from':'from_osm','to':'to_osm' })
edges['from']=edges.apply(lambda row: node_id_map[row['from_osm']], axis=1)
edges['to']=edges.apply(lambda row: node_id_map[row['to_osm']], axis=1)

G=nx.Graph()
for i, row in edges.iterrows():
    G.add_edge(row['from'], row['to'], attr_dict={'distance':row['distance']})

#pos={p:[nodes.iloc[p]['lon'], nodes.iloc[p]['lat']] for p in list(G.nodes())}
#nx.draw(G, pos=pos)
    
# get the N closest nodes to the centre of each zone
lon_lat_list= [[shape(f['geometry']).centroid.x, shape(f['geometry']).centroid.y] for f in zones_shp['features']]   
kdtree_nodes=spatial.KDTree(np.array(nodes[['lon', 'lat']]))
closest_nodes=[]
for i in range(len(lon_lat_list)):
    dist, c_nodes=kdtree_nodes.query(lon_lat_list[i], 10)
    closest_nodes.append({'node_ids':list(c_nodes)})
# loop through all zone pairs
routes={fromGeoId:{toGeoId:{} for toGeoId in range(len(lon_lat_list))} for fromGeoId in range(len(lon_lat_list))}

print('Zone to Zone')
for fromGeoId in range(len(lon_lat_list)):
    #TODO: weights
    print(fromGeoId)
    for toGeoId in range(len(lon_lat_list)):
        try:
            node_route=nx.shortest_path(G, closest_nodes[fromGeoId]['node_ids'][0],
                                   closest_nodes[toGeoId]['node_ids'][0], weight='distance')
            node_route=[int(n) for n in node_route]
            distances=[G[node_route[i]][node_route[i+1]]['attr_dict']['distance'] for i in range(len(node_route)-1)]
            routes[fromGeoId][toGeoId]={'nodes':tuple(node_route), 'distances': tuple(distances)}
        except:
            routes[fromGeoId][toGeoId]={'nodes':(), 'distances': ()}
            print('No path from zone'+str(fromGeoId)+' to ' +str(toGeoId))

# get closest node to each connection point
connection_points=json.load(open(CONNECTION_POINTS_PATH))
closest_nodes_cp=[]
for i in range(len(connection_points)):
    dist, c_nodes=kdtree_nodes.query([connection_points[i]['lon'],connection_points[i]['lat']], 10)
    closest_nodes_cp.append({'node_ids':list(c_nodes)})

connection_routes=[{'to': [], 'from':[]} for g in range(len(lon_lat_list))]
# loop through all connction point and zone pairs
print('Zone to and from CP')
for zone in range(len(lon_lat_list)):
    print(zone)
    for cp in range(len(connection_points)):
        try:
            node_route_to_cp=nx.shortest_path(G, closest_nodes[zone]['node_ids'][0],
                                       closest_nodes_cp[cp]['node_ids'][0], weight='distance')
            node_route_to_cp=[int(n) for n in node_route_to_cp]
            distances=[G[node_route_to_cp[i]][node_route_to_cp[i+1]]['attr_dict']['distance'] for i in range(len(node_route_to_cp)-1)]
            connection_routes[zone]['to'].append({'nodes': tuple(node_route_to_cp), 'distances': tuple(distances)})
            connection_routes[zone]['from'].append({'nodes':tuple(reversed(node_route_to_cp)), 'distances':tuple(reversed(distances))})
        except:
            connection_routes[zone]['to'].append({'nodes':(), 'distances': ()})
            connection_routes[zone]['from'].append({'nodes':(), 'distances': ()})
            print('No path from zone'+str(zone)+' to CP ' +str(cp)) 
# TODO: no point in using tupes if saving to json- they get converted to lists anyway
json.dump(routes, open(ZONE_NODE_ROUTES_PATH, 'w'))    
json.dump(connection_routes, open(CONNECTION_NODE_ROUTES_PATH, 'w')) 
nodes.to_csv(NODES_PATH)    