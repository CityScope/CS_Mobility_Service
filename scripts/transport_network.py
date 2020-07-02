#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:10:02 2019

@author: doorleyr
"""
# shapely and osmnet dont work on anaconda default


import json
from shapely.geometry import Point, shape
import networkx as nx
from scipy import spatial
import numpy as np
#import pickle
import pandas as pd
import math
#import urllib
import matplotlib.path as mplPath

#import xml.etree.ElementTree as et
#import urllib.request
import gzip


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

def approx_shape_centroid(geometry):
    if geometry['type']=='Polygon':
        centroid=list(np.mean(geometry['coordinates'][0], axis=0))
        return centroid
    elif geometry['type']=='MultiPolygon':
        centroid=list(np.mean(geometry['coordinates'][0][0], axis=0))
        return centroid
    else:
        print('Unknown geometry type')
        
def dict_to_gzip(data, write_location):
    json_data = json.dumps(data)
    # Convert to bytes
    encoded = json_data.encode('utf-8')
    with gzip.open(write_location, 'wb') as f:
        f.write(encoded)

def gzip_to_dict(location):
    with gzip.open(location, 'rb') as f:
        file_content = f.read()        
    test2=json.loads(file_content.decode('utf-8'))
    return test2

class Mode():
    def __init__(self, mode_descrip, mode_id):
        self.speed_met_s=mode_descrip['speed_m_s']
        self.name=mode_descrip['name']
        self.activity=mode_descrip['activity']
        self.internal_net=mode_descrip['internal_net']
        self.co2_emissions_kg_met=mode_descrip['co2_emissions_kg_met']
        self.fixed_costs=mode_descrip['fixed_costs']
        self.id=mode_id
        if 'weight' in mode_descrip:
            # not needed for new modes
            self.weight=mode_descrip['weight']
        if 'copy_route' in mode_descrip:
            self.copy_route=mode_descrip['copy_route']
    
class Polygon_Location():
    def __init__(self, geometry, area_type, in_sim_area, geoid=None):
        self.area_type=area_type
        self.geometry=geometry
        self.centroid=approx_shape_centroid(geometry)
        self.in_sim_area=in_sim_area
        self.geoid=geoid
    def get_close_nodes(self, transport_network):
        self.close_nodes=transport_network.get_closest_internal_nodes(self.centroid, 5)
        # get close nodes
    def get_containing_poly(self, polygon_location_list):
        for ind_pl, pl in enumerate(polygon_location_list):
            if pl.geometry['type'] == 'Polygon':
                polygon_paths=[mplPath.Path(pl.geometry["coordinates"][0])]
            elif pl.geometry['type'] == 'MultiPolygon':
                polygon_paths=[mplPath.Path(c) for c in pl.geometry["coordinates"][0]]
            if any(pp.contains_point(self.centroid) for pp in polygon_paths):
                return ind_pl, pl
                break
                
class Route():
    def __init__(self, internal_route, costs, pre_time=0, post_time=0):
        self.internal_route=internal_route
        self.pre_time=pre_time
        self.post_time=post_time
        self.costs=costs

class Transport_Network():
      
    def __init__(self, table_name, city_folder):
        mode_descrips=json.load(open('./cities/'+city_folder+'/clean/mode_descriptions.json'))
        self.table_name=table_name
        self.city_folder=city_folder
        self.PORTALS_PATH='./cities/'+city_folder+'/clean/portals.geojson' 
        self.ROUTE_COSTS_PATH='./cities/'+city_folder+'/clean/route_costs.json'
#        self.FLOYD_PREDECESSOR_PATH='./cities/'+city_folder+'/clean/fw_result.json'
#        self.INT_NET_DF_FLOYD_PATH='./cities/'+city_folder+'/clean/sim_net_df_floyd.csv'
        self.INT_NET_PATH='./cities/'+city_folder+'/clean/'
        self.SIM_AREA_PATH='./cities/'+city_folder+'/clean/table_area.geojson'            
        self.base_modes=[Mode(d, mode_id) for mode_id, d in enumerate(mode_descrips)]
        self.new_modes=[]
        # External route costs
        try:
            self.external_costs=json.load(open(self.ROUTE_COSTS_PATH))
        except:
            print('External routes not yet prepared. Preparing now')
            self.prepare_external_routes()
            self.external_costs=json.load(open(self.ROUTE_COSTS_PATH))
        # load internal routes
        self.sim_net_floyd_results={}
        sim_net_floyd_df={}
#        self.sim_net_floyd_results['driving']=json.load(open(self.INT_NET_PATH+'fw_result.json'))
#        self.sim_net_floyd_results['pt']=json.load(open(self.INT_NET_PATH+'fw_result_pt.json'))
#        self.sim_net_floyd_results['active']=json.load(open(self.INT_NET_PATH+'fw_result_active.json'))
        self.sim_net_floyd_results['driving']=gzip_to_dict(self.INT_NET_PATH+'fw_result.txt.gz')
        self.sim_net_floyd_results['pt']=gzip_to_dict(self.INT_NET_PATH+'fw_result_pt.txt.gz')
        self.sim_net_floyd_results['active']=gzip_to_dict(self.INT_NET_PATH+'fw_result_active.txt.gz')
        sim_net_floyd_df['driving']=pd.read_csv(self.INT_NET_PATH+'sim_net_df_floyd.csv')
        sim_net_floyd_df['pt']=pd.read_csv(self.INT_NET_PATH+'sim_net_df_floyd_pt.csv')
        sim_net_floyd_df['active']=pd.read_csv(self.INT_NET_PATH+'sim_net_df_floyd_active.csv')


        # create dictionaries for node coordinates and attributes
        self.nodes_to_link_attributes={}
        self.node_to_lon_lat={}
        self.sim_node_ids={}
        self.internal_nodes_kdtree={}
        for mode in ['driving', 'pt', 'active']:
            self.nodes_to_link_attributes[mode]={}
            self.node_to_lon_lat[mode]={}
            weight_columns=[col for col in sim_net_floyd_df[mode] if 'minutes' in col]
            for ind, row in sim_net_floyd_df[mode].iterrows():
                node_key='{}_{}'.format(row['aNodes'], row['bNodes'])
                self.nodes_to_link_attributes[mode][node_key]={
                    'distance': row['distance'],
                    'from_coord': [float(row['aNodeLon']), float(row['aNodeLat'])],
                    'to_coord': [float(row['bNodeLon']), float(row['bNodeLat'])]}
                for col in weight_columns:
                    self.nodes_to_link_attributes[mode][node_key][col]=row[col]
                if mode=='pt':
                    self.nodes_to_link_attributes[mode][node_key]['activity']=row['activity']
                else:
                    # activity attribute only used in the pt network
                    # for other modes, activity is the same on every link
                    self.nodes_to_link_attributes[mode][node_key]['activity']=None
                if row['aNodes'] not in self.node_to_lon_lat[mode]:
                   self.node_to_lon_lat[mode][str(row['aNodes'])]  = [float(row['aNodeLon']), float(row['aNodeLat'])]
                if row['bNodes'] not in self.node_to_lon_lat[mode]:
                   self.node_to_lon_lat[mode][str(row['bNodes'])]  = [float(row['bNodeLon']), float(row['bNodeLat'])]     
            self.sim_node_ids[mode]=[node for node in self.node_to_lon_lat[mode]]
            sim_node_lls=[self.node_to_lon_lat[mode][node] for node in self.sim_node_ids[mode]]
            self.internal_nodes_kdtree[mode]=spatial.KDTree(np.array(sim_node_lls))
        # LOad portals    
        portals_geojson=json.load(open(self.PORTALS_PATH))
        self.portals=[]
        for feature in portals_geojson['features']:
            new_portal=Polygon_Location(geometry=feature['geometry'],
                                                 area_type='portal',
                                                 in_sim_area=True)
            new_portal.get_close_nodes(transport_network=self)
            self.portals.append(new_portal)
            
    def set_new_modes(self, new_mode_specs):
        n_base_modes=len(self.base_modes)
        self.new_modes=[Mode(spec, n_base_modes+i_s) for i_s, spec in enumerate(new_mode_specs)]
            
    def get_external_costs(self, zone_geoid, portal_id):
        return {mode: self.external_costs[mode][zone_geoid][str(portal_id)] for mode in self.external_costs}
    
    def get_closest_internal_nodes(self, from_coordinates, n_nodes):
        node_ids={}
        for mode in self.internal_nodes_kdtree:
            node_ids[mode]=[self.sim_node_ids[mode][n_ind] for n_ind in 
                       self.internal_nodes_kdtree[mode].query(from_coordinates, n_nodes)[1]]
        return node_ids
    
    def get_node_path_from_fw_try_multi(self, from_list, to_list, mode):
        for fn in from_list:
            for tn in to_list:
                try: 
                    node_path=self.get_node_path_from_fw(fn, tn, mode)
                    return node_path
                except:
                    pass
        return None
              
    def get_node_path_from_fw(self, from_node, to_node, internal_net):
        if from_node==to_node:
            return []
        pred=to_node
        path=[pred]
        while not pred==from_node:
            pred=str(self.sim_net_floyd_results[internal_net][from_node][pred])
            path.insert(0,pred)
        return path
            
    def get_path_coords_distances(self, path, internal_net, weight, mode):
        """
        takes a list of node ids and returns:
            a list of coordinates of each node
            a list of distances of each link
        may return empty lists if the path has length of 0 or 1
        """
        coords, distances, activities,  minutes=[], [], [], []
        costs= {'driving':0, 'walking':0, 'waiting':0,'cycling':0, 'pt':0}
        if len(path)>1:
            for node_ind in range(len(path)-1):
                from_node=path[node_ind]
                to_node=path[node_ind+1]
                link_attributes=self.nodes_to_link_attributes[internal_net]['{}_{}'.format(from_node, to_node)]
                distances+=[link_attributes['distance']]
                coords+=[link_attributes['from_coord']]
                minutes+=[link_attributes[weight]]
                if mode.name=='pt':
                    costs[link_attributes['activity']]+=link_attributes[weight]
                    activities.append(link_attributes['activity'])
                else:
                    costs[mode.activity]+=link_attributes[weight]
            # add the final coordinate of the very last segment
            coords+= [link_attributes['to_coord']]       
        return coords, distances, activities, minutes, costs
    
    def get_internal_routes(self, from_loc, to_loc):
        routes={}
        for im, mode in enumerate(self.base_modes):
            internal_net=mode.internal_net
            path=self.get_node_path_from_fw_try_multi(from_loc.close_nodes[internal_net], to_loc.close_nodes[internal_net], internal_net)  
            if path is None:
                coords, distances, total_distance, activities, minutes=[], [], float('1e10') , [], []
                costs= {'driving':0, 'walking':0, 'waiting':0,'cycling':0, 'pt':0}
            else:
                coords, distances, activities, minutes, costs=self.get_path_coords_distances(path, internal_net, mode.weight, mode=mode)
                total_distance=sum(distances)
            routes[mode.name]={
                    'costs': costs,
                    'internal_route':{
                            'node_path':path, 'distances': distances,
                            'activities': activities, 'minutes': minutes,
                            'total_distance': total_distance, 'coords': coords}}
        return routes
    
    def get_routes(self, from_loc, to_loc):
        """
        gets the best route by each mode between 2 locations
        returns a Route object
        If the from_loc or to_loc is not a grid cell (i.e. is outside the sim area)
        the Route returned will contain the internal part of the route as well
        as well the time duration of the external portion (pre or post)
        
        """
        if ((from_loc.area_type=='grid') and (to_loc.area_type=='grid')):
            routes=self.get_internal_routes(from_loc, to_loc)
            return {mode: Route(internal_route=routes[mode], costs=routes[mode]['costs']) for mode in routes}
        elif to_loc.area_type=='grid':
            # trip arriving into the site
            external_routes_by_portal={ip : self.get_external_costs(from_loc.geoid, ip) for ip, portal in enumerate(self.portals)}
            internal_routes_by_portal={ip: self.get_internal_routes(portal, to_loc) for ip, portal in enumerate(self.portals)}
            best_routes=self.get_best_portal_routes(external_routes_by_portal, internal_routes_by_portal, 'in')
            return best_routes
        elif from_loc.area_type=='grid':
            external_routes_by_portal={ip : self.get_external_costs(to_loc.geoid, ip) for ip, portal in enumerate(self.portals)}
            internal_routes_by_portal={ip: self.get_internal_routes(portal, from_loc) for ip, portal in enumerate(self.portals)}
            best_routes=self.get_best_portal_routes(external_routes_by_portal, internal_routes_by_portal, 'out')
            return best_routes
        else:
            routes=self.get_approx_routes(from_loc, to_loc)
            return {mode: Route(internal_route=routes[mode], costs=routes[mode]['costs']) for mode in routes}

    def get_best_portal_routes(self, external_routes_by_portal, internal_routes_by_portal, direction):
        """
        Takes a dict containing the internal routes by each mode and portal and
        a dict containing the external routes by each mode and portal
        returns the best portal route for each mode
        """
        best_routes={}
        for mode in external_routes_by_portal[0]:
            best_portal_route=None
            best_portal_time=float('inf')
            for pid, portal in enumerate(self.portals):
                total_external_time=sum([external_routes_by_portal[pid][mode][t
                                     ] for t in external_routes_by_portal[pid][mode]])
                all_times={t: internal_routes_by_portal[pid][mode]['costs'][t] + 
                           external_routes_by_portal[pid][mode][t] for t in external_routes_by_portal[pid][mode]}
                total_time=sum(all_times[t] for t in all_times)
                if total_time< best_portal_time:
                    best_portal_time=total_time
                    best_portal_route=internal_routes_by_portal[pid][mode]
                    best_costs=all_times
            if direction =='in':
                best_routes[mode]= Route(internal_route=best_portal_route, costs=best_costs, pre_time=total_external_time)
            else:
                best_routes[mode]= Route(internal_route=best_portal_route, costs=best_costs, post_time=total_external_time)
        return best_routes
    
    def get_approx_routes(self, from_loc, to_loc):
        routes={}
        distance=1.4*get_haversine_distance(from_loc.centroid, to_loc.centroid)
        for im, mode in enumerate(self.base_modes):
            routes[mode.name]={
                    'costs': {'driving':0, 'walking':0, 'waiting':0,
                                              'cycling':0, 'pt':0}, 
                    'internal_route':{'node_path':[], 'distances': [],
                             'total_distance': 0, 'coords': []}}
            routes[mode.name]['costs'][mode.activity]=(distance/mode.speed_met_s)/60
            for f_act in mode.fixed_costs:
                routes[mode.name]['costs'][f_act]+=mode.fixed_costs[f_act]
        return routes
        
    def prepare_external_routes(self):
        import osmnet
        ALL_ZONES_PATH='./cities/'+self.city_folder+'/clean/model_area.geojson'
               
        # networks from CS_Accessibility- placed in folder manually for now
        PT_NODES_PATH='./cities/'+self.city_folder+'/clean/comb_network_nodes.csv'
        PT_EDGES_PATH='./cities/'+self.city_folder+'/clean/comb_network_edges.csv'
        PED_NODES_PATH='./cities/'+self.city_folder+'/clean/osm_ped_network_nodes.csv'
        PED_EDGES_PATH='./cities/'+self.city_folder+'/clean/osm_ped_network_edges.csv'

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
        if self.city_folder=='Hamburg':
            all_zones_geoid_order=[f['properties']['GEO_ID'] for f in all_zones_shp['features']]
        else:
            all_zones_geoid_order=[f['properties']['GEO_ID'].split('US')[1] for f in all_zones_shp['features']]
        
        portals=json.load(open(self.PORTALS_PATH))
                
        largeArea=[shape(f['geometry']) for f in all_zones_shp['features']]
        bounds=[shp.bounds for shp in largeArea]
        boundsAll=[min([b[0] for b in bounds]), #W
                       min([b[1] for b in bounds]), #S
                       max([b[2] for b in bounds]), #E
                       max([b[3] for b in bounds])] #N
        
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
        json.dump(ext_route_costs, open(self.ROUTE_COSTS_PATH, 'w'))  

