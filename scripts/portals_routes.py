#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:10:02 2019

@author: doorleyr
"""
# shapely and osmnet dont work on anaconda default


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

def find_route_multi(start_nodes, end_nodes, graph, weight):
    """
    tries to find paths between lists of possible start and end nodes
    Once a path is successfully found it is returned. Otherwise returns None
    """
    for sn in start_nodes:
        for en in end_nodes:
            try:
                node_path=nx.shortest_path(graph,sn,en, weight=weight)
                return node_path
            except:
                pass
    return None

def df_to_geojson(edges_df, nodes_df, net_type):
    features=[]
    for e_ind, e_row in edges_df.iterrows():
        from_node=e_row['from_node_id']
        to_node=e_row['to_node_id']
        if net_type=='pt':
            edge_type=pandana_link_types[e_row['net_type']]
            weight_minutes= e_row['weight']
        else:
            edge_type=net_type
            weight_minutes= e_row['distance']/SPEEDS_MET_S[net_type]
        coordinates=[[nodes_df.iloc[from_node]['x'], nodes_df.iloc[from_node]['y']],
                     [nodes_df.iloc[to_node]['x'], nodes_df.iloc[to_node]['y']]]
        features.append({"type": "Feature",
                         "properties": {'edge_type': edge_type,
                                        'weight_minutes': weight_minutes},
                         'geometry': {
                                 "type": "LineString",
                                 'coordinates':coordinates },
                         })
    return {"type": "FeatureCollection",
            "crs": { "type": "name", "properties": { "name": "epsg:4326" } },
            "features": features}
# =============================================================================
# Constants
# =============================================================================
city='Detroit'

ALL_ZONES_PATH='./scripts/cities/'+city+'/clean/model_area.geojson'
PORTALS_PATH='./scripts/cities/'+city+'/clean/portals.geojson'
SIM_AREA_PATH='./scripts/cities/'+city+'/clean/sim_area.geojson'

# networks from CS_Accessibility- placed in folder manually for now
PT_NODES_PATH='./scripts/cities/'+city+'/clean/comb_network_nodes.csv'
PT_EDGES_PATH='./scripts/cities/'+city+'/clean/comb_network_edges.csv'
PED_NODES_PATH='./scripts/cities/'+city+'/clean/osm_ped_network_nodes.csv'
PED_EDGES_PATH='./scripts/cities/'+city+'/clean/osm_ped_network_edges.csv'


ROUTE_COSTS_PATH='./scripts/cities/'+city+'/clean/route_costs.json'
SIM_GRAPHS_PATH='./scripts/cities/'+city+'/clean/sim_area_nets.p'
SIM_NET_GEOJSON_PATH='./scripts/cities/'+city+'/clean/'

SPEEDS_MET_S={'driving':30/3.6,
        'cycling':15/3.6,
        'walking':4.8/3.6}

pandana_link_types={'osm to transit': 'waiting',
                    'transit to osm': 'waiting',
                    'walk': 'walking',
                    'transit': 'pt'
                    }
# =============================================================================
# Load network data
# =============================================================================
# get the area bounds
all_zones_shp=json.load(open(ALL_ZONES_PATH))
if city=='Hamburg':
    all_zones_geoid_order=[f['properties']['GEO_ID'] for f in all_zones_shp['features']]
else:
    all_zones_geoid_order=[f['properties']['GEO_ID'].split('US')[1] for f in all_zones_shp['features']]

portals=json.load(open(PORTALS_PATH))
sim_area=json.load(open(SIM_AREA_PATH))


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

cycle_nodes,cycle_edges= drive_nodes.copy(),drive_edges.copy()

# get the pt net as nodes and edges dfs
pt_edges=pd.read_csv(PT_EDGES_PATH)
pt_nodes=pd.read_csv(PT_NODES_PATH)

walk_edges=pd.read_csv(PED_EDGES_PATH)
walk_nodes=pd.read_csv(PED_NODES_PATH)

# renumber nodes in both networks as 1 to N
pt_nodes, pt_edges, _ =rename_nodes(pt_nodes, pt_edges, 'id_int', 'to_int', 'from_int')
drive_nodes, drive_edges, _=rename_nodes(drive_nodes, drive_edges, 'id', 'to', 'from')
walk_nodes, walk_edges, _=rename_nodes(walk_nodes, walk_edges, 'id', 'to', 'from')
cycle_nodes, cycle_edges, _=rename_nodes(cycle_nodes, cycle_edges, 'id', 'to', 'from')

network_dfs={'driving': {'edges':drive_edges, 'nodes': drive_nodes} ,
              'pt': {'edges':pt_edges, 'nodes': pt_nodes},
              'walking': {'edges':walk_edges, 'nodes': walk_nodes},
              'cycling': {'edges':cycle_edges, 'nodes': cycle_nodes}}

# =============================================================================
# Create graphs and add portal links
# =============================================================================
#for each network, create a networkx graph and add the links to/from portals

for osm_mode in ['driving', 'walking', 'cycling']:
    G=nx.Graph()
    for i, row in network_dfs[osm_mode]['edges'].iterrows():
        G.add_edge(row['from_node_id'], row['to_node_id'], attr_dict={
                'weight_minutes':(row['distance']/SPEEDS_MET_S[osm_mode])/60,
                'type': osm_mode})
    network_dfs[osm_mode]['graph']=G
    
G_pt=nx.Graph()
for i, row in network_dfs['pt']['edges'].iterrows():
    G_pt.add_edge(row['from_node_id'], row['to_node_id'], 
                     attr_dict={'weight_minutes':row['weight'],
                                'type': pandana_link_types[row['net_type']]})
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
                       attr_dict={'type': 'from_portal', 'weight_minutes':0, 'distance': 0})
            network_dfs[net]['graph'].add_edge(ni, 'p'+str(p),
                       attr_dict={'type': 'to_portal', 'weight_minutes':0, 'distance': 0})

#import matplotlib.pyplot as plt
#colors={'driving': 'red', 'pt': 'green'}
#plt.figure()
## Test: find the nodes that connect to each portal
neighbours={}
for net in network_dfs:
    neighbours[net]={}
    for p in range(len(portals['features'])):
        if network_dfs[net]['graph'].has_node('p'+str(p)):
            nb=[n for n in network_dfs[net]['graph'].neighbors('p'+str(p))]
            neighbours[net][p]=nb 
        else:
            neighbours[net][p]=[]
#xs=[network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][0]]+[network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][1]] +    [network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][2]] + [network_dfs[net]['nodes'].iloc[n]['x'] for n in neighbours[net][3]] 
#ys=[network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][0]]+[network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][1]] +    [network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][2]] + [network_dfs[net]['nodes'].iloc[n]['y'] for n in neighbours[net][3]] 
#plt.scatter(xs, ys)

# =============================================================================
# Find routes
# =============================================================================
## get the N closest nodes to the centre of each zone
lon_lat_list= [[shape(f['geometry']).centroid.x, shape(f['geometry']).centroid.y
                ] for f in all_zones_shp['features']]  
closest_nodes={}
for net in network_dfs:
    closest_nodes[net]=[]
    kdtree_nodes=spatial.KDTree(np.array(network_dfs[net]['nodes'][['x', 'y']]))
    for i in range(len(lon_lat_list)):
        _, c_nodes=kdtree_nodes.query(lon_lat_list[i], 10)
        closest_nodes[net].append(list(c_nodes))

#create empty route_costs object
route_costs={}
#for each zone and each portal:

for mode in network_dfs:
    route_costs[mode]={}
    for z in range(len(all_zones_shp['features'])):
        print(mode+ ' ' + str(z))
        route_costs[mode][all_zones_geoid_order[z]]={}
        for p in range(len(portals['features'])):
            route_costs[mode][all_zones_geoid_order[z]][p]={}
            node_route_z2p=find_route_multi(closest_nodes[mode][z], 
                                                  ['p'+str(p)], 
                                                  network_dfs[mode]['graph'],
                                                  'weight_minutes')
            if node_route_z2p:
                route_net_types=[network_dfs[mode]['graph'][
                        node_route_z2p[i]][
                        node_route_z2p[i+1]
                        ]['attr_dict']['type'
                         ] for i in range(len(node_route_z2p)-1)]
                route_weights=[network_dfs[mode]['graph'][
                        node_route_z2p[i]][
                        node_route_z2p[i+1]
                        ]['attr_dict']['weight_minutes'
                         ] for i in range(len(node_route_z2p)-1)]
                for l_type in ['walking', 'cycling', 'driving', 'pt', 
                               'waiting']:
                    route_costs[mode][all_zones_geoid_order[z]][p][l_type]=sum(
                            [route_weights[l] for l in range(len(route_weights)
                            ) if route_net_types[l]==l_type])
            else:
                for l_type in ['walking', 'cycling', 'driving', 'pt', 
                               'waiting']:
                    route_costs[mode][all_zones_geoid_order[z]][p][l_type]=10000


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
#   TODO: stop checking when one is found
    for n in range(len(network_dfs[net]['nodes'])):
        if shape(sim_area['features'][0]['geometry']).contains(Point(
                network_dfs[net]['nodes'].iloc[n]['x'], 
                network_dfs[net]['nodes'].iloc[n]['y'])):
            sim_area_nodes.add(n)
#    add portals to list
#    sim_area_nodes.add(['p'+str(p) for p in range(len(portals['features']))])
    sim_area_edges_df=network_dfs[net]['edges'].loc[
            ((network_dfs[net]['edges']['from_node_id'].isin(sim_area_nodes)) | # either from or to node is in the sim area
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

# Create geojson for each network
net_geojson={}
for net in sim_area_nets:
    net_geojson[net]=df_to_geojson(sim_area_nets[net]['edges'], 
                                     sim_area_nets[net]['nodes'], 
                                     net)
    json.dump(net_geojson[net], 
              open(SIM_NET_GEOJSON_PATH+str(net)+'_net.geojson', 'w'))
          
# Create networkx graphs for each
for osm_mode in ['driving', 'walking', 'cycling']:   
    G_sim=nx.Graph()
    for i, row in sim_area_nets[osm_mode]['edges'].iterrows():
        G_sim.add_edge(row['from_node_id'], row['to_node_id'], attr_dict={
                'weight_minutes':(row['distance']/SPEEDS_MET_S[osm_mode])/60,
                'type': osm_mode})
    sim_area_nets[osm_mode]['graph']=G_sim

G_pt_sim=nx.Graph()
for i, row in sim_area_nets['pt']['edges'].iterrows():
    G_pt_sim.add_edge(row['from_node_id'], row['to_node_id'], 
                     attr_dict={'weight_minutes':row['weight'],
                                'type': pandana_link_types[row['net_type']]})
sim_area_nets['pt']['graph']= G_pt_sim           

# go through neighbour list for each portal
# and add the dummy links
for net in sim_area_nets:
    for p in neighbours[net]:
        for nb in neighbours[net][p]:
            if nb in node_name_maps[net]:
                sim_area_nets[net]['graph'].add_edge('p'+str(p), node_name_maps[net][nb],
                           attr_dict={'type': 'from_portal', 'weight_minutes':0})
                sim_area_nets[net]['graph'].add_edge( node_name_maps[net][nb],'p'+str(p),
                           attr_dict={'type': 'to_portal', 'weight_minutes':0})
            else:
                print(str(nb)+' not in sim area net for '+net+
                      '. Node not on any  valid links')
            
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
