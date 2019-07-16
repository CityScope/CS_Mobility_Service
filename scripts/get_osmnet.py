#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:21:11 2019

@author: doorleyr
"""
import osmnet
import json
from shapely.geometry import shape
import networkx as nx
from scipy import spatial
import numpy as np
import pickle

city='Boston'

CLEAN_ZONES_PATH='./'+city+'/clean/zones.geojson'
ZONE_NODE_ROUTES_PATH='./'+city+'/clean/route_nodes.json'
CONNECTION_NODE_ROUTES_PATH='./'+city+'/clean/connection_route_nodes.json'
CONNECTION_POINTS_PATH='./'+city+'/clean/connection_points.json'
NODES_PATH='./'+city+'/clean/nodes.csv'
GRAPH_PATH='./'+city+'/clean/graph.p'

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

types=[
 'living_street',
 'motorway',
 'motorway_link',
 'primary',
 'primary_link',
# 'residential',
 'road',
 'secondary',
 'secondary_link',
 'tertiary',
 'tertiary_link',
 'trunk',
 'trunk_link',
 'unclassified']

edges=edges.loc[((edges['highway'].isin(types)) | (edges['bridge']=='yes'))]

edges=edges.reset_index()
used_nodes=set(list(edges['from'])+list(edges['to']))

nodes=nodes.loc[nodes['id'].isin(used_nodes)]

        
node_id_map={nodes.iloc[i]['id']:i for i in range(len(nodes))}
nodes=nodes.rename(columns={'id':'osm_id', 'x': 'lon', 'y': 'lat'})
nodes['id']=range(len(nodes))

edges=edges.rename(columns={'from':'from_osm','to':'to_osm' })
edges['from']=edges.apply(lambda row: node_id_map[row['from_osm']], axis=1)
edges['to']=edges.apply(lambda row: node_id_map[row['to_osm']], axis=1)

G=nx.Graph()
for i, row in edges.iterrows():
    G.add_edge(row['from'], row['to'], attr_dict={'distance':row['distance']})

pickle.dump(G, open(GRAPH_PATH, 'wb'))
nodes.to_csv(NODES_PATH)  
#pos={p:[nodes.iloc[p]['lon'], nodes.iloc[p]['lat']] for p in list(G.nodes())}
#nx.draw(G, pos=pos)
    
## get the N closest nodes to the centre of each zone
#lon_lat_list= [[shape(f['geometry']).centroid.x, shape(f['geometry']).centroid.y] for f in zones_shp['features']]   
#kdtree_nodes=spatial.KDTree(np.array(nodes[['lon', 'lat']]))
#closest_nodes=[]
#for i in range(len(lon_lat_list)):
#    dist, c_nodes=kdtree_nodes.query(lon_lat_list[i], 10)
#    closest_nodes.append({'node_ids':list(c_nodes)})
## loop through all zone pairs
#routes={fromGeoId:{toGeoId:{} for toGeoId in range(len(lon_lat_list))} for fromGeoId in range(len(lon_lat_list))}
#
#print('Zone to Zone')
#for fromGeoId in range(len(lon_lat_list)):
#    #TODO: weights
#    print(fromGeoId)
#    for toGeoId in range(len(lon_lat_list)):
#        try:
#            node_route=nx.shortest_path(G, closest_nodes[fromGeoId]['node_ids'][0],
#                                   closest_nodes[toGeoId]['node_ids'][0], weight='distance')
#            node_route=[int(n) for n in node_route]
#            distances=[G[node_route[i]][node_route[i+1]]['attr_dict']['distance'] for i in range(len(node_route)-1)]
#            routes[fromGeoId][toGeoId]={'nodes':tuple(node_route), 'distances': tuple(distances)}
#        except:
#            routes[fromGeoId][toGeoId]={'nodes':(), 'distances': ()}
#            print('No path from zone'+str(fromGeoId)+' to ' +str(toGeoId))
#
## get closest node to each connection point
#connection_points=json.load(open(CONNECTION_POINTS_PATH))
#closest_nodes_cp=[]
#for i in range(len(connection_points)):
#    dist, c_nodes=kdtree_nodes.query([connection_points[i]['lon'],connection_points[i]['lat']], 10)
#    closest_nodes_cp.append({'node_ids':list(c_nodes)})

#json.dump(routes, open(ZONE_NODE_ROUTES_PATH, 'w'))    
  