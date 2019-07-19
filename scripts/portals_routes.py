#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:10:02 2019

@author: doorleyr
"""
import osmnet
import json
from shapely.geometry import Point, shape
import networkx as nx
from scipy import spatial
import numpy as np
import pickle
import pandas as pd

# =============================================================================
# Functions
# =============================================================================
def rename_nodes(nodes_df, edges_df, node_id_name, to_name, from_name):
    nodes_df['old_node_id']=nodes_df[node_id_name].copy()
    nodes_df['node_id']=range(len(nodes_df))
    node_name_map={nodes_df.iloc[i]['old_node_id']: i for i in range(len(nodes_df))}
    edges_df['from_node_id']=edges_df.apply(lambda row: node_name_map[row[from_name]], axis=1)
    edges_df['to_node_id']=edges_df.apply(lambda row: node_name_map[row[to_name]], axis=1)
    return nodes_df, edges_df, node_name_map

def find_route_multi(start_nodes, end_nodes, graph):
    """
    tries to find paths between lists of possible start and end nodes
    Once a path is successfully found it is returned. Otherwise returns None
    """
    for sn in start_nodes:
        for en in end_nodes:
            try:
                node_path=nx.shortest_path(graph,sn,en)
                return node_path
            except:
                pass
    return None
# =============================================================================
# Constants
# =============================================================================
city='Hamburg'

ALL_ZONES_PATH='./cities/'+city+'/clean/model_area.geojson'
SIM_ZONES_PATH='./cities/'+city+'/clean/sim_area.geojson'
PORTALS_PATH='./cities/'+city+'/clean/portals.geojson'
ROUTE_COSTS_PATH='./cities/'+city+'/clean/route_costs.json'
SIM_GRAPHS_PATH='./cities/'+city+'/clean/sim_area_nets.p'

pt_network_urls={
        'Boston': None,
        'Hamburg': 'https://raw.githubusercontent.com/CityScope/CS_Accessibility/master/python/Hamburg/data/',
        'Detroit': None}

# =============================================================================
# Load network data
# =============================================================================
# get the area bounds
all_zones_shp=json.load(open(ALL_ZONES_PATH))
all_zones_geoid_order=[f['properties']['geoid'] for f in all_zones_shp['features']]
sim_zones_shp=json.load(open(SIM_ZONES_PATH))
portals=json.load(open(PORTALS_PATH))


largeArea=[shape(f['geometry']) for f in all_zones_shp['features']]
bounds=[shp.bounds for shp in largeArea]
boundsAll=[min([b[0] for b in bounds]), #W
               min([b[1] for b in bounds]), #S
               max([b[2] for b in bounds]), #E
               max([b[3] for b in bounds])] #N

# get the osm as nodes and edges dfs
# TODO: try one-way and find routes between highest degree node in each zone
drive_nodes,drive_edges=osmnet.load.network_from_bbox(lat_min=boundsAll[1], lng_min=boundsAll[0], lat_max=boundsAll[3], 
                              lng_max=boundsAll[2], bbox=None, network_type='drive', 
                              two_way=True, timeout=180, 
                              custom_osm_filter=None)

# get the pt net as nodes and edges dfs
pt_edges=pd.read_csv(pt_network_urls[city]+'combined_network_edges.csv')
pt_nodes=pd.read_csv(pt_network_urls[city]+'combined_network_nodes.csv')

# renumber nodes in both networks as 1 to N
pt_nodes, pt_edges, _ =rename_nodes(pt_nodes, pt_edges, 'id_int', 'to_int', 'from_int')
drive_nodes, drive_edges, _=rename_nodes(drive_nodes, drive_edges, 'id', 'to', 'from')

network_dfs={'driving': {'edges':drive_edges, 'nodes': drive_nodes} ,
              'pt': {'edges':pt_edges, 'nodes': pt_nodes}}

# =============================================================================
# Create graphs and add portal links
# =============================================================================
#for each network, create a networkx graph and add the links to/from portals
G_drive=nx.Graph()
for i, row in network_dfs['driving']['edges'].iterrows():
    G_drive.add_edge(row['from_node_id'], row['to_node_id'], attr_dict={'distance':row['distance']})
G_pt=nx.Graph()
for i, row in network_dfs['pt']['edges'].iterrows():
    G_pt.add_edge(row['from_node_id'], row['to_node_id'], 
                     attr_dict={'weight_minutes':row['weight'],
                                'net_type': row['net_type']})

network_dfs['driving']['graph']=G_drive
network_dfs['pt']['graph']=G_pt

#for each network
#for each portal
#find nodes inside the portal and add zero_cost links to/from those
for net in network_dfs:
    for p in range(len(portals['features'])):
        p_shape=shape(portals['features'][p]['geometry'])
        nodes_inside=[n for n in range(len(network_dfs[net]['nodes'])) if p_shape.contains(
                Point([network_dfs[net]['nodes'].iloc[n]['x'],
                       network_dfs[net]['nodes'].iloc[n]['y']]))]
        for ni in nodes_inside:
            network_dfs[net]['graph'].add_edge('p'+str(p), ni,
                       attr_dict={'net_type': 'from_portal', 'weight_minutes':0, 'distance': 0})
            network_dfs[net]['graph'].add_edge(ni, 'p'+str(p),
                       attr_dict={'net_type': 'to_portal', 'weight_minutes':0, 'distance': 0})

#import matplotlib.pyplot as plt
#colors={'driving': 'red', 'pt': 'green'}
#plt.figure()
## Test: find the nodes that connect to each portal
neighbours={}
for net in network_dfs:
    neighbours[net]={}
    for p in range(len(portals['features'])):
        nb=[n for n in network_dfs[net]['graph'].neighbors('p'+str(p))]
        neighbours[net][p]=nb       
#xs=[network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][0]]+[network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][1]] +    [network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][2]] + [network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][3]] 
#ys=[network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][0]]+[network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][1]] +    [network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][2]] + [network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][3]] 
#plt.scatter(xs, ys)

# =============================================================================
# Find routes
# =============================================================================
## get the N closest nodes to the centre of each zone
lon_lat_list= [[shape(f['geometry']).centroid.x, shape(f['geometry']).centroid.y
                ] for f in all_zones_shp['features']]   
kdtree_drive_nodes=spatial.KDTree(np.array(network_dfs['driving']['nodes'][['x', 'y']]))
kdtree_pt_nodes=spatial.KDTree(np.array(network_dfs['pt']['nodes'][['x', 'y']]))
closest_nodes=[]
for i in range(len(lon_lat_list)):
    _, drive_c_nodes=kdtree_drive_nodes.query(lon_lat_list[i], 10)
    _, pt_c_nodes=kdtree_pt_nodes.query(lon_lat_list[i], 10)
    closest_nodes.append({'driving':list(drive_c_nodes), 'pt': list(pt_c_nodes)})

#create empty route_costs object
route_costs={}
#for each zone and each portal:

# TODO: DRY code for each network
route_costs['driving']={}
route_costs['pt']={}
for z in range(len(all_zones_shp['features'])):
    print(z)
    route_costs['driving'][all_zones_geoid_order[z]]={}
    route_costs['pt'][all_zones_geoid_order[z]]={}
    for p in range(len(portals['features'])):
        drive_node_route_z2p=find_route_multi(closest_nodes[z]['driving'], 
                                              ['p'+str(p)], 
                                              network_dfs['driving']['graph'])
        drive_distance=sum([network_dfs['driving']['graph'][
                drive_node_route_z2p[i]][
                drive_node_route_z2p[i+1]
                ]['attr_dict']['distance'
                 ] for i in range(len(drive_node_route_z2p)-1)])
        pt_node_route_z2p=find_route_multi(closest_nodes[z]['pt'], 
                                              ['p'+str(p)], 
                                              network_dfs['pt']['graph'])
        if pt_node_route_z2p:
            pt_route_net_types=[network_dfs['pt']['graph'][
                    pt_node_route_z2p[i]][
                    pt_node_route_z2p[i+1]
                    ]['attr_dict']['net_type'
                     ] for i in range(len(pt_node_route_z2p)-1)]
            pt_route_weights=[network_dfs['pt']['graph'][
                    pt_node_route_z2p[i]][
                    pt_node_route_z2p[i+1]
                    ]['attr_dict']['weight_minutes'
                     ] for i in range(len(pt_node_route_z2p)-1)]
#            pt_wait_time=sum([pt_route_weights[n] for n in range(len(pt_route_weights
#                          ))  if pt_route_net_types[n] =='osm to transit'])
            # Wait times are greatly overestimated so just use first wait
            pt_wait_times=[pt_route_weights[n] for n in range(len(pt_route_weights
                          ))  if pt_route_net_types[n] =='osm to transit']
            if pt_wait_times:
                pt_wait_time=pt_wait_times[0]
            else:
                pt_wait_time=0
            pt_walk_time=sum([pt_route_weights[n] for n in range(len(pt_route_weights
                  ))  if pt_route_net_types[n] =='walk'])
            pt_pt_time=sum([pt_route_weights[n] for n in range(len(pt_route_weights
                  ))  if pt_route_net_types[n] =='transit'])
        else:
            print('No PT route')
            pt_wait_time, pt_walk_time, pt_pt_time=float('nan'), float('nan'), float('nan')
        route_costs['pt'][all_zones_geoid_order[z]][p]={
                'wait_time': pt_wait_time,
                'walk_time': pt_walk_time,
                'pt_time': pt_pt_time,
                'total_time': pt_wait_time+pt_walk_time+pt_pt_time,
#                'net_types': pt_route_net_types,
#                'weights':pt_route_weights,
#                'nodes': pt_node_route_z2p
                }
        route_costs['driving'][all_zones_geoid_order[z]][p]={'drive_distance': drive_distance}

## plot routes
#z=26
#p=1
#xs= [network_dfs['pt']['nodes'].iloc[n]['x'] for n in route_costs['pt'][z][p]['nodes']
#                 if isinstance(n, int)]
#ys= [network_dfs['pt']['nodes'].iloc[n]['y'] for n in route_costs['pt'][z][p]['nodes']
#                 if isinstance(n, int)]
#import gmplot 
#gmap=gmplot.GoogleMapPlotter(0,0,13)
#gmap.plot(ys, xs,  
#           'cornflowerblue', edge_width = 2.5)
#gmap.draw( "/Users/doorleyr/Desktop/map11.html" )

# Save the results
json.dump(route_costs, open(ROUTE_COSTS_PATH, 'w')) 

# =============================================================================
#  Make smaller graphs for the simulation area
# =============================================================================
sim_area_nets={}
node_name_maps={}
for net in network_dfs:
    sim_area_nodes=set()
#    check if each node in any sim area, if so add to list
    for n in range(len(network_dfs[net]['nodes'])):
        for z in range(len(sim_zones_shp['features'])):
            if shape(sim_zones_shp['features'][z]['geometry']).contains(Point(
                    network_dfs[net]['nodes'].iloc[n]['x'], 
                    network_dfs[net]['nodes'].iloc[n]['y'])):
                sim_area_nodes.add(n)
#    add portals to list
#    sim_area_nodes.add(['p'+str(p) for p in range(len(portals['features']))])
    sim_area_edges_df=network_dfs[net]['edges'].loc[
            ((network_dfs[net]['edges']['from_node_id'].isin(sim_area_nodes)) | 
            (network_dfs[net]['edges']['to_node_id'].isin(sim_area_nodes)))]
    #    subset nodes df and edges df by nodes in list
    # update the node list to include other edges of partially contained links 
    sim_area_nodes=set(list(sim_area_edges_df['from_node_id'])+list(sim_area_edges_df['to_node_id']))
    sim_area_nodes_df=network_dfs[net]['nodes'].loc[
            network_dfs[net]['nodes']['node_id'].isin(sim_area_nodes)]
    #    rename nodes
    sim_area_nodes_df, sim_area_edges_df, node_name_maps[net]= rename_nodes(sim_area_nodes_df, sim_area_edges_df, 
                                                   'node_id', 'to_node_id', 'from_node_id')
    sim_area_nets[net]={'nodes': sim_area_nodes_df, 'edges': sim_area_edges_df}
    
G_drive_sim=nx.Graph()
for i, row in sim_area_nets['driving']['edges'].iterrows():
    G_drive.add_edge(row['from_node_id'], row['to_node_id'], attr_dict={'distance':row['distance']})
G_pt_sim=nx.Graph()
for i, row in sim_area_nets['pt']['edges'].iterrows():
    G_pt.add_edge(row['from_node_id'], row['to_node_id'], 
                     attr_dict={'weight_minutes':row['weight'],
                                'net_type': row['net_type']})

sim_area_nets['pt']['graph']= G_pt_sim  
sim_area_nets['driving']['graph']= G_drive_sim              

#    go through neighbour list for each portal
for net in sim_area_nets:
    for p in neighbours[net]:
        for nb in neighbours[net][p]:
            sim_area_nets[net]['graph'].add_edge('p'+str(p), node_name_maps[net][nb],
                       attr_dict={'net_type': 'from_portal', 'weight_minutes':0, 'distance': 0})
            sim_area_nets[net]['graph'].add_edge( node_name_maps[net][nb],'p'+str(p),
                       attr_dict={'net_type': 'to_portal', 'weight_minutes':0, 'distance': 0})
            
# Plot
#net='driving'
#gmap=gmplot.GoogleMapPlotter(0,0,1)
#for i, row in sim_area_nets[net]['edges'].iterrows(): 
#    from_node=row['from_node_id']
#    to_node=row['to_node_id']
#    xs=[sim_area_nets[net]['nodes'].iloc[from_node]['x'], sim_area_nets[net]['nodes'].iloc[to_node]['x']]
#    ys=[sim_area_nets[net]['nodes'].iloc[from_node]['y'], sim_area_nets[net]['nodes'].iloc[to_node]['y']]
#    gmap.plot(ys, xs,  
#               'cornflowerblue', edge_width = 1)
#gmap.draw( '/Users/doorleyr/Desktop/map_'+net+'.html' )
pickle.dump(sim_area_nets, open(SIM_GRAPHS_PATH, 'wb'))
