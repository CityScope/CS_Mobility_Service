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
#import pickle
import pandas as pd
import math
#import urllib

# =============================================================================
# Functions
# =============================================================================
def rename_nodes(nodes_df, edges_df, node_id_name, to_name, from_name):
    nodes_df['old_node_id']=nodes_df[node_id_name].copy()
    nodes_df['node_id']=range(len(nodes_df))
    node_name_map={nodes_df.iloc[i]['old_node_id']:i for i in range(len(nodes_df))}
    rev_node_name_map={v:str(k) for k,v in node_name_map.items()}
    edges_df['from_node_id']=edges_df.apply(lambda row: node_name_map[row[from_name]], axis=1)
    edges_df['to_node_id']=edges_df.apply(lambda row: node_name_map[row[to_name]], axis=1)
    return nodes_df, edges_df, rev_node_name_map

def find_route_multi(start_nodes, end_nodes, graph, weight):
    """
    tries to find paths between lists of possible start and end nodes
    Once a path is successfully found it is returned. Otherwise returns None
    """
    for sn in start_nodes:
        for en in end_nodes:
            try:
                node_path=nx.dijkstra_path(graph,sn,en, weight=weight)
                return node_path
            except:
                pass
    return None
    
def get_haversine_distance(point_1, point_2):
    """
    Calculate the distance between any 2 points on earth given as [lon, lat]
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [point_1[0], point_1[1], 
                                                point_2[0], point_2[1]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in kilometers. Use 3956 for miles
    return c * r
    
def createGridGraphs(grid_coords_ll, graphs, nrows, ncols, cell_size):
    """
    returns new networks including roads around the cells
    """
    for mode in graphs:
#    create graph internal to the grid
        graphs[mode]['graph'].add_nodes_from('g'+str(n) for n in range(len(grid_coords_ll)))
        for c in range(ncols):
            for r in range(nrows):
                # if not at the end of a row, add h link
                if not c==ncols-1:
                    graphs[mode]['graph'].add_edge('g'+str(r*ncols+c), 'g'+str(r*ncols+c+1), 
                          weight=(cell_size/SPEEDS_MET_S[mode])/(60),
                          attr_dict={'type': mode})
                    graphs[mode]['graph'].add_edge('g'+str(r*ncols+c+1), 'g'+str(r*ncols+c), 
                          weight=(cell_size/SPEEDS_MET_S[mode])/(60),
                          attr_dict={'type': mode})
                # if not at the end of a column, add v link
                if not r==nrows-1:
                    graphs[mode]['graph'].add_edge('g'+str(r*ncols+c), 'g'+str((r+1)*ncols+c), 
                          weight=(cell_size/SPEEDS_MET_S[mode])/(60),
                          attr_dict={'type': mode})
                    graphs[mode]['graph'].add_edge('g'+str((r+1)*ncols+c), 'g'+str(r*ncols+c), 
                          weight=(cell_size/SPEEDS_MET_S[mode])/(60),
                          attr_dict={'type': mode})
        # create links between the 4 corners of the grid and the road network
        kd_tree_nodes=spatial.KDTree(np.array(graphs[mode]['nodes'][['x', 'y']]))
        for n in [0, ncols-1, (nrows-1)*ncols, (nrows*ncols)-1]: 
            closest=kd_tree_nodes.query(grid_coords_ll[n], k=1)[1]
            distance_m=get_haversine_distance(grid_coords_ll[n], list(graphs[mode]['nodes'].iloc[closest][['x', 'y']]))
            graphs[mode]['graph'].add_edge('g'+str(n), closest, 
                          weight=(distance_m/SPEEDS_MET_S[mode])/(60),
                          attr_dict={'type': mode})
            graphs[mode]['graph'].add_edge(closest, 'g'+str(n), 
                          weight=(distance_m/SPEEDS_MET_S[mode])/(60),
                          attr_dict={'type': mode})
    return graphs 
# =============================================================================
# Constants
# =============================================================================
city='Hamburg'

table_name_map={'Boston':"mocho",
     'Hamburg':"grasbrook",
     'Detroit': "corktown"}
host='https://cityio.media.mit.edu/'
cityIO_grid_url=host+'api/table/'+table_name_map[city]

ALL_ZONES_PATH='./scripts/cities/'+city+'/clean/model_area.geojson'
PORTALS_PATH='./scripts/cities/'+city+'/clean/portals.geojson'
SIM_AREA_PATH='./scripts/cities/'+city+'/clean/table_area.geojson'

# networks from CS_Accessibility- placed in folder manually for now
PT_NODES_PATH='./scripts/cities/'+city+'/clean/comb_network_nodes.csv'
PT_EDGES_PATH='./scripts/cities/'+city+'/clean/comb_network_edges.csv'
PED_NODES_PATH='./scripts/cities/'+city+'/clean/osm_ped_network_nodes.csv'
PED_EDGES_PATH='./scripts/cities/'+city+'/clean/osm_ped_network_edges.csv'


ROUTE_COSTS_PATH='./scripts/cities/'+city+'/clean/route_costs.json'
INTERNAL_COSTS_PATH='./scripts/cities/'+city+'/clean/internal_route_costs.json'
PORTAL_INTERNAL_COSTS_PATH='./scripts/cities/'+city+'/clean/portal_internal_route_costs.json'
SIM_GRAPHS_PATH='./scripts/cities/'+city+'/clean/sim_area_nets.p'
SIM_NET_GEOJSON_PATH='./scripts/cities/'+city+'/clean/'
NEIGHBOURS_PATH='./scripts/cities/'+city+'/clean/neighbours.json'

SPEEDS_MET_S={'driving':30/3.6,
        'cycling':15/3.6,
        'walking':4.8/3.6,
        'pt': 4.8/3.6 # only used for grid use walking speed for pt
        }

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
pt_nodes, pt_edges, pt_node_name_map =rename_nodes(pt_nodes, pt_edges, 'id_int', 'to_int', 'from_int')
drive_nodes, drive_edges, drive_node_name_map=rename_nodes(drive_nodes, drive_edges, 'id', 'to', 'from')
walk_nodes, walk_edges, walk_node_name_map=rename_nodes(walk_nodes, walk_edges, 'id', 'to', 'from')
cycle_nodes, cycle_edges, cycle_node_name_map=rename_nodes(cycle_nodes, cycle_edges, 'id', 'to', 'from')

network_dfs={'driving': {'edges':drive_edges, 'nodes': drive_nodes, 'node_name_map': drive_node_name_map} ,
              'pt': {'edges':pt_edges, 'nodes': pt_nodes, 'node_name_map': pt_node_name_map},
              'walking': {'edges':walk_edges, 'nodes': walk_nodes, 'node_name_map': walk_node_name_map},
              'cycling': {'edges':cycle_edges, 'nodes': cycle_nodes, 'node_name_map': cycle_node_name_map}}

# =============================================================================
# Create graphs and add portal links
# =============================================================================
#for each network, create a networkx graph and add the links to/from portals

for osm_mode in ['driving', 'walking', 'cycling']:
    G=nx.Graph()
    for i, row in network_dfs[osm_mode]['edges'].iterrows():
        G.add_edge(row['from_node_id'], row['to_node_id'], 
                   weight=(row['distance']/SPEEDS_MET_S[osm_mode])/60,
                   attr_dict={'type': osm_mode})
    network_dfs[osm_mode]['graph']=G
    
G_pt=nx.Graph()
for i, row in network_dfs['pt']['edges'].iterrows():
    G_pt.add_edge(row['from_node_id'], row['to_node_id'], 
                  weight=row['weight'],
                  attr_dict={'type': pandana_link_types[row['net_type']]})
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
            network_dfs[net]['graph'].add_edge('p'+str(p), ni,weight=0,
                       attr_dict={'type': 'from_portal', 'distance': 0})
            network_dfs[net]['graph'].add_edge(ni, 'p'+str(p),weight=0,
                       attr_dict={'type': 'to_portal',  'distance': 0})

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

#create empty ext_route_costs object
ext_route_costs={}
#for each zone and each portal:

for mode in network_dfs:
    ext_route_costs[mode]={}
    for z in range(len(all_zones_shp['features'])):
        print(mode+ ' ' + str(z))
        ext_route_costs[mode][all_zones_geoid_order[z]]={}
        for p in range(len(portals['features'])):
            ext_route_costs[mode][all_zones_geoid_order[z]][p]={}
            node_route_z2p=find_route_multi(closest_nodes[mode][z], 
                                                  ['p'+str(p)], 
                                                  network_dfs[mode]['graph'],
                                                  'weight')
            if node_route_z2p:
                route_net_types=[network_dfs[mode]['graph'][
                        node_route_z2p[i]][
                        node_route_z2p[i+1]
                        ]['attr_dict']['type'
                         ] for i in range(len(node_route_z2p)-1)]
                route_weights=[network_dfs[mode]['graph'][
                        node_route_z2p[i]][
                        node_route_z2p[i+1]
                        ]['weight'
                         ] for i in range(len(node_route_z2p)-1)]
                for l_type in ['walking', 'cycling', 'driving', 'pt', 
                               'waiting']:
                    ext_route_costs[mode][all_zones_geoid_order[z]][p][l_type]=sum(
                            [route_weights[l] for l in range(len(route_weights)
                            ) if route_net_types[l]==l_type])
            else:
                for l_type in ['walking', 'cycling', 'driving', 'pt', 
                               'waiting']:
                    ext_route_costs[mode][all_zones_geoid_order[z]][p][l_type]=10000

# Save the results
json.dump(ext_route_costs, open(ROUTE_COSTS_PATH, 'w')) 
