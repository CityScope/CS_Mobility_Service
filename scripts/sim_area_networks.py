#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:40:36 2019

@author: doorleyr
"""
import json
from shapely.geometry import Point, shape
import osmnet

def rename_nodes(nodes_df, edges_df, node_id_name, to_name, from_name):
    nodes_df['old_node_id']=nodes_df[node_id_name].copy()
    nodes_df['node_id']=range(len(nodes_df))
    node_name_map={nodes_df.iloc[i]['old_node_id']:i for i in range(len(nodes_df))}
    rev_node_name_map={v:str(k) for k,v in node_name_map.items()}
    edges_df['from_node_id']=edges_df.apply(lambda row: node_name_map[row[from_name]], axis=1)
    edges_df['to_node_id']=edges_df.apply(lambda row: node_name_map[row[to_name]], axis=1)
    return nodes_df, edges_df, rev_node_name_map


def df_to_geojson(edges_df, nodes_df, net_type):
    features=[]
    for e_ind, e_row in edges_df.iterrows():
        from_node=e_row['from_node_id']
        to_node=e_row['to_node_id']
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

city='Hamburg'
NEIGHBOURS_PATH='./scripts/cities/'+city+'/clean/neighbours.json'
SIM_AREA_PATH='./scripts/cities/'+city+'/clean/table_area.geojson'
SIM_NETWORK_PATH_ROOT='./scripts/cities/'+city+'/clean/'

sim_area=json.load(open(SIM_AREA_PATH))


full_area=[shape(f['geometry']) for f in sim_area['features']]
bounds=[shp.bounds for shp in full_area]
boundsAll=[min([b[0] for b in bounds]), #W
               min([b[1] for b in bounds]), #S
               max([b[2] for b in bounds]), #E
               max([b[3] for b in bounds])] #N

all_nodes,all_edges=osmnet.load.network_from_bbox(lat_min=boundsAll[1], lng_min=boundsAll[0], lat_max=boundsAll[3], 
                              lng_max=boundsAll[2], bbox=None, 
                              two_way=True, timeout=180, 
                              custom_osm_filter='')


# create list of nodes inside sim area
sim_area_nodes=set()
# check if each node in any sim area, if so add to list
for ind, row in all_nodes.iterrows():
    if shape(sim_area['features'][0]['geometry']).contains(Point(
            row['x'], 
            row['y'])):
        sim_area_nodes.add(row['id'])
all_edges=all_edges.loc[((all_edges['from'].isin(sim_area_nodes))|
        (all_edges['to'].isin(sim_area_nodes)))]

sim_area_nodes=set(all_edges['from'])|set(all_edges['to'])


edges_types=set(all_edges['highway'])

drive_types=[t for t in edges_types if t not in ['footway', 'path', 'steps']]
cycle_types=[t for t in edges_types if t not in [ 'steps']]
pt_types=edges_types
walk_types=edges_types


drive_edges=all_edges.loc[all_edges['highway'].isin(drive_types)]
drive_nodes=all_nodes.loc[((all_nodes['id'].isin(drive_edges['to']))|
        (all_nodes['id'].isin(drive_edges['from'])))]
drive_nodes, drive_edges, _=rename_nodes(drive_nodes, drive_edges, 'id', 'to', 'from')
drive_net_geo=df_to_geojson(drive_edges, drive_nodes, 'driving')

walk_edges=all_edges.loc[all_edges['highway'].isin(walk_types)]
walk_nodes=all_nodes.loc[((all_nodes['id'].isin(walk_edges['to']))|
        (all_nodes['id'].isin(walk_edges['from'])))]
walk_nodes, walk_edges, _=rename_nodes(walk_nodes, walk_edges, 'id', 'to', 'from')
walk_net_geo=df_to_geojson(walk_edges, walk_nodes, 'walking')

pt_edges=all_edges.loc[all_edges['highway'].isin(pt_types)]
pt_nodes=all_nodes.loc[((all_nodes['id'].isin(pt_edges['to']))|
        (all_nodes['id'].isin(pt_edges['from'])))]
pt_nodes, pt_edges, _=rename_nodes(pt_nodes, pt_edges, 'id', 'to', 'from')
pt_net_geo=df_to_geojson(pt_edges, pt_nodes, 'pt')

cycle_edges=all_edges.loc[all_edges['highway'].isin(pt_types)]
cycle_nodes=all_nodes.loc[((all_nodes['id'].isin(cycle_edges['to']))|
        (all_nodes['id'].isin(cycle_edges['from'])))]
cycle_nodes, cycle_edges, _=rename_nodes(cycle_nodes, cycle_edges, 'id', 'to', 'from')
cycle_net_geo=df_to_geojson(cycle_edges, cycle_nodes, 'cycling')





SPEEDS_MET_S={'driving':30/3.6,
        'cycling':15/3.6,
        'walking':4.8/3.6,
        'pt': 20/3.6}

json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'driving_net.geojson', 'w'))
json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'walking_net.geojson', 'w'))
json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'pt_net.geojson', 'w'))
json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'cycling_net.geojson', 'w'))

