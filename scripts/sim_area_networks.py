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

city='Detroit'
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

drive_nodes,drive_edges=osmnet.load.network_from_bbox(lat_min=boundsAll[1], lng_min=boundsAll[0], lat_max=boundsAll[3], 
                              lng_max=boundsAll[2], bbox=None, 
                              two_way=True, timeout=180, 
                              custom_osm_filter='')


# create list of nodes inside sim area
sim_area_nodes=set()
# check if each node in any sim area, if so add to list
for ind, row in drive_nodes.iterrows():
    if shape(sim_area['features'][0]['geometry']).contains(Point(
            row['x'], 
            row['y'])):
        sim_area_nodes.add(row['id'])
drive_edges=drive_edges.loc[((drive_edges['from'].isin(sim_area_nodes))|
        (drive_edges['to'].isin(sim_area_nodes)))]

sim_area_nodes=set(drive_edges['from'])|set(drive_edges['to'])

drive_nodes, drive_edges, _=rename_nodes(drive_nodes, drive_edges, 'id', 'to', 'from')

SPEEDS_MET_S={'driving':30/3.6,
        'cycling':15/3.6,
        'walking':4.8/3.6}

drive_net_geo=df_to_geojson(drive_edges, drive_nodes, 'driving')

json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'driving_net1.geojson', 'w'))
json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'walking_net1.geojson', 'w'))
json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'pt_net1.geojson', 'w'))
json.dump(drive_net_geo, open(SIM_NETWORK_PATH_ROOT+'cycling_net1.geojson', 'w'))

