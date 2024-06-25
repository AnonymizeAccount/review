import os
import gc
import copy
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics.pairwise import pairwise_distances, haversine_distances
from .funcs import get_transition_matrix, get_adj_matrix, get_network_distance, construct_graph_RCM_SH, print_graph_attr


class StandardScaler:
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class TrajDataLoader(object):
    def __init__(self, x, y, direction_x, direction_y, length, batch_size, pad_with_last_sample=True):
        self.x = x
        self.y = y
        self.direction_x = direction_x
        self.direction_y = direction_y
        self.length = length
        self.batch_size = batch_size
        if pad_with_last_sample:
            num_padding = (batch_size - (len(self.x) % batch_size)) % batch_size
            # x_padding = np.repeat(self.x[-1:], num_padding, axis=0)
            # y_padding = np.repeat(self.y[-1:], num_padding, axis=0)
            # print(f'x_padding: {x_padding.shape}')
            # print(f'y_padding: {y_padding.shape}')
            if num_padding > self.x.shape[0]:
                idx_padding = np.random.choice(self.x.shape[0], num_padding, replace=True)
            else:
                idx_padding = np.random.choice(num_padding, num_padding, replace=False)
            x_padding, y_padding = self.x[idx_padding], self.y[idx_padding]
            direction_x_padding, direction_y_padding = self.direction_x[idx_padding], self.direction_y[idx_padding]
            # print(f'x_padding: {x_padding.shape}')
            # print(f'y_padding: {y_padding.shape}')
            # assert 1==2

            self.x = np.concatenate([self.x, x_padding], axis=0)
            self.y = np.concatenate([self.y, y_padding], axis=0)
            self.direction_x = np.concatenate([self.direction_x, direction_x_padding], axis=0)
            self.direction_y = np.concatenate([self.direction_y, direction_y_padding], axis=0)
            for i in range(num_padding):
                self.length.append(length[-1])

        self.size = self.x.shape[0]

        self.num_batch = int(self.size // self.batch_size)
        if self.num_batch == 0:
            self.num_batch = 1

    def get_num_batch(self):
        return self.num_batch

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.x[start_ind: end_ind, ...]
                y_i = self.y[start_ind: end_ind, ...]
                direction_x_i = self.direction_x[start_ind: end_ind, ...]
                direction_y_i = self.direction_y[start_ind: end_ind, ...]
                length_i = self.length[start_ind: end_ind]
                yield (x_i, y_i, direction_x_i, direction_y_i, length_i)
                self.current_ind += 1

        return _wrapper()


def get_direction_labels(direction, direction_lb):
    for i in range(len(direction_lb)):
        if i == 0 and direction < direction_lb[1]:
            return i

        elif i == len(direction_lb) - 1:
            return i

        elif direction >= direction_lb[i] and direction < direction_lb[i+1]:
            return i

    raise ValueError


def construct_graph(dir, args):
    lookup_idx_osmid, lookup_osmid_idx = {}, {}
    original_edges = gpd.read_file(os.path.join(dir, "edges.shp"))
    original_nodes = gpd.read_file(os.path.join(dir, "nodes.shp"))
    # map_data same as original_edges
    map_data = pd.read_csv(os.path.join(dir, 'edges.csv'))
    # map: LINESTRING geometry for all edges (routes), e.g., LINESTRING (104.0921872 30.7219643, 104.089735...
    map = map_data.to_numpy()[:, -1]
    # nodes_coor: node GPS, e.g., POINT (104.10029 30.71479)
    nodes_coor = original_nodes.to_numpy()[:, -1]

    # print(f'original_edges: \n{original_edges}')
    # print(f'original_nodes: \n{original_nodes[["osmid", "y", "x"]]}')
    # original_edges.to_csv('edge.txt')
    # original_nodes[["osmid", "y", "x"]].to_csv('node.txt', index=False)
    # assert 1==2
    # print(f'map: \n{map}')
    # print(f'nodes_coor: {nodes_coor.shape}\n{nodes_coor}')

    nodes = []
    for node in nodes_coor:
        link = str(node)
        link = link[7:-1].split(", ")
        lon, lat = float(link[0].split(" ")[0]), float(link[0].split(" ")[1])
        nodes.append(np.array([lon, lat]))
    # nodes_coor: 2D numpy array, [[lon, lat], [lon, lat], ...]
    nodes_coor = np.array(nodes)

    # edges: 2D numpy array, shape: (num_edges, 2), [[node_osmid, node_osmid], [node_osmid, node_osmid], ...]
    edges = original_edges.to_numpy()[:, :2]
    edges = edges.astype(np.int64)
    # nodes: 2D numpy array, shape: (num_nodes, 4), [[node_osmid, lat, lon, street_count], ...]
    nodes = original_nodes.to_numpy()[:, :4]

    # print(f'edges: \n{edges}')
    # print(f'nodes: \n{nodes}')
    # assert 1==2

    # lookup_idx_osmid: {1: node_osmid, 2: node_osmid, ...}, idx: 1 ~ num_nodes
    # lookup_osmid_idx: {node_osmid: 1, node_osmid: 2, ...}, value: 1 ~ num_nodes
    G = nx.MultiDiGraph()
    print(f'Reading nodes...')
    for nid in tqdm(range(nodes.shape[0])):
        lookup_idx_osmid[nid + 1] = nodes[nid, 0]
        # 1 ~ num_nodes
        lookup_osmid_idx[nodes[nid, 0]] = nid + 1
        G.add_node(nid + 1, lat=nodes[nid, 1], lon=nodes[nid, 2], street_count=nodes[nid, 3])

    print(f'Reading edges...')
    # idx=eid, eid: 0 ~ num_edges-1
    for eid in tqdm(range(edges.shape[0])):
        G.add_edge(lookup_osmid_idx[edges[eid, 0]], lookup_osmid_idx[edges[eid, 1]], highway=original_edges.iloc[eid]['highway'],
                   oneway=original_edges.iloc[eid]['oneway'], length=original_edges.iloc[eid]['length'],
                   geometry=str(original_edges.iloc[eid]['geometry'])[12:-1], idx=eid)


    if not os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', 'shortest_paths.npy')) or \
        not os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', 'length_shortest_paths.npy')):

        shortest_paths, length_shortest_paths = [], []
        for edge_1_id, edge_1 in enumerate(list(G.edges)):
            print(edge_1_id)
            start_node_1, end_node_1, _ = edge_1
            edge_1_shortest_paths, edge_1_length_shortest_paths = [], []

            for edge_2_id, edge_2 in enumerate(list(G.edges)):
                start_node_2, end_node_2, _ = edge_2
                # print(f'start_node_1: {start_node_1}, end_node_1: {end_node_1}; start_node_2: {start_node_2}, end_node_2: {end_node_2}')
                if edge_1 != edge_2:
                    try:
                        shortest_path = nx.shortest_path(G=G, source=start_node_1, target=start_node_2, method='dijkstra')
                        length_shortest_path = len(shortest_path)
                    except:
                        shortest_path, length_shortest_path = None, -1
                else:
                    shortest_path = []
                    length_shortest_path = 0
                # print(f'shortest: \n{shortest_path}')
                edge_1_shortest_paths.append(shortest_path)
                edge_1_length_shortest_paths.append(length_shortest_path)

            shortest_paths.append(edge_1_shortest_paths)
            length_shortest_paths.append(edge_1_length_shortest_paths)

        length_shortest_paths = np.array(length_shortest_paths)
        shortest_paths = np.array(shortest_paths)
        np.save(os.path.join(args.data_dir, args.dataset, 'map', 'shortest_paths.npy'), shortest_paths)
        np.save(os.path.join(args.data_dir, args.dataset, 'map', 'length_shortest_paths.npy'), length_shortest_paths)
    else:
        sp_time = time.time()
        # shortest_paths = np.load(os.path.join(args.data_dir, args.dataset, 'map', 'shortest_paths.npy'), allow_pickle=True)
        length_shortest_paths = np.load(os.path.join(args.data_dir, args.dataset, 'map', 'length_shortest_paths.npy'), allow_pickle=True)
        # print(f'shortest_paths: {shortest_paths.shape}')
        # print(f'length_shortest_paths: max: {np.max(length_shortest_paths)}, min: {np.min(length_shortest_paths)}, {length_shortest_paths.shape}\n{length_shortest_paths}')
        print(f'Done Loading shortest paths, time: {time.time() - sp_time:.3f}')

    if args.dataset == 'shanghai':
        args.data_dir = '/home/yihong/RNTraj/data'

    if os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', 'node_adj_edges_efficient.npy')):
        node_adj_edges_efficient = np.load(os.path.join(args.data_dir, args.dataset, 'map', 'node_adj_edges_efficient.npy'), allow_pickle=True)
    else:
        # out_edges: {node: [out_node, out_node, ...], ...}, out_node are neighbors of node
        out_edges = {}

        # node: 1 ~ num_nodes
        for node in G.nodes:
            # G.out_edges(node): [(node, out_node), (node, out_node), ...], type: list, same for .in_edges()
            for edge in G.out_edges(node):
                start_node, end_node = edge
                if start_node not in out_edges.keys():
                    out_edges[start_node] = [end_node]
                else:
                    if end_node not in out_edges[start_node]:
                        out_edges[start_node].append(end_node)

        # complete keys for all edges
        for i in range(1, len(G.nodes) + 1):
            if i not in out_edges.keys():
                out_edges[i] = []

        node_adj_edges, node_adj_edges_efficient = [], []
        # sort out_edges by keys
        out_edges = dict(sorted(out_edges.items()))
        print(f'Getting node_adj_edges...')
        for i_, start_node in enumerate(tqdm(out_edges.keys())):
            # indicating node-edge adjacency
            adj_edge_list = [True for i in range(len(G.edges))]
            # G[end_node][start_node][path]["idx"] == G[end_node][start_node][path]["idx"] only happens when start_node == end_node (self loop)
            for end_node in out_edges[start_node]:
                if G.has_edge(start_node, end_node):
                    # MultiDiGraph may have more than one edge connecting same start-end node pairs
                    for path in G[start_node][end_node].keys():
                        # Set adjacent nodes to False in the adj_edge_list
                        if adj_edge_list[G[start_node][end_node][path]["idx"]] is True:
                            adj_edge_list[G[start_node][end_node][path]["idx"]] = False

                # if G.has_edge(end_node, start_node):
                #     # MultiDiGraph may have more than one edge connecting same start-end node pairs
                #     for path in G[end_node][start_node].keys():
                #         # Set adjacent nodes to False in the adj_edge_list
                #         if adj_edge_list[G[end_node][start_node][path]["idx"]] is True:
                #             adj_edge_list[G[end_node][start_node][path]["idx"]] = False

            # TODO: whether there exist some path go through end to start (check direction)

            # turn [True, False, ..] into [1, 0, ...]
            adj_edge_list = np.array(adj_edge_list, dtype=np.int)
            # get all adjacent edge idx list, idx: 0 ~ num_edges - 1
            idx = np.where(adj_edge_list == 0)[0]
            # node_adj_edges_efficient[i]: np.array(idx), len(node_adj_edges_efficient) == num_nodes
            node_adj_edges_efficient.append(np.array(idx, dtype=np.int))

        # convert list to np array
        node_adj_edges_efficient = np.array(node_adj_edges_efficient)
        np.save(os.path.join(args.data_dir, args.dataset, 'map', 'node_adj_edges_efficient.npy'), node_adj_edges_efficient)
    node_adj_edges = node_adj_edges_efficient

    # if os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', f'nodes_links_{args.direction}directions.npy')):
    #     print(f'Loading nodes_links_directions...')
    #     # shape: (num_nodes, )
    #     nodes_links_directions = np.load(os.path.join(args.data_dir, args.dataset, 'map', f'nodes_links_{args.direction}directions.npy'),
    #                                      allow_pickle=True)
    # else:
    #     edges_ori = original_edges.to_numpy()
    #     nodes_links_directions = []
    #     # node: 0 ~ num_nodes - 1
    #     for node in tqdm(range(len(node_adj_edges))):
    #         # get osmid of node
    #         osm_nodeid = lookup_idx_osmid[node + 1]
    #         # get node adjacent edge list: [adj_edge1, adj_edge2, adj_edge3, ...]
    #         paths = node_adj_edges[node]
    #         node_links_directions = []
    #         for path in paths:
    #             # link: list of geometry, e.g., ['104.100294 30.7147948', '104.1000959 30.7148658']
    #             link = edges_ori[path, -1]
    #             link = str(link)[12:-1].split(", ")
    #             # if node == start_node
    #             if edges_ori[path, 0] == osm_nodeid:
    #                 s_lon, s_lat = float(link[0].split(" ")[0]), float(link[0].split(" ")[1])
    #                 e_lon, e_lat = float(link[-1].split(" ")[0]), float(link[-1].split(" ")[1])
    #             # if node == end_node
    #             elif edges_ori[path, 1] == osm_nodeid:
    #                 e_lon, e_lat = float(link[0].split(" ")[0]), float(link[0].split(" ")[1])
    #                 s_lon, s_lat = float(link[-1].split(" ")[0]), float(link[-1].split(" ")[1])
    #             else:
    #                 raise ValueError
    #             direction = torch.angle(torch.tensor(complex(e_lon - s_lon, e_lat - s_lat))).item() * 180 / 3.14159
    #             direction_lb = torch.arange(-180., 180., step=360. / args.direction)
    #             node_links_directions.append(get_direction_labels(direction, direction_lb))
    #
    #         nodes_links_directions.append(np.array(node_links_directions))
    #     # shape: (num_nodes, )
    #     nodes_links_directions = np.array(nodes_links_directions)
    #
    #     assert 1==2
    #     np.save(os.path.join(args.data_dir, args.dataset, 'map', f'nodes_links_{args.direction}directions.npy'), nodes_links_directions)
    #
    # print(f'nodes_links_directions: {nodes_links_directions.shape}\n{nodes_links_directions}')

    '''
    Returns
    
    G: networkx MultiDiGraph
    node_adj_edges: numpy array, shape: (num_nodes, ), with different length in the second dimension
    nodes_coor: 2D numpy array, [[lon, lat], [lon, lat], ...]
    map: LINESTRING geometry for all edges (routes), e.g., LINESTRING (104.0921872 30.7219643, 104.089735...
    '''

    return G, node_adj_edges, nodes_coor, map, length_shortest_paths


def split_data(path_x, path_y, path_direction_x, path_direction_y, length, args):
    '''
    path_direction_x: 1 ~ 8
    path_direction_y: 0 ~ 7
    path_x: 1 ~ num_path
    path_y: 0 ~ num_path - 1
    '''

    path_x = pad_sequence(path_x, batch_first=True)
    path_y = pad_sequence(path_y, batch_first=True)
    path_direction_x = pad_sequence(path_direction_x, batch_first=True)
    path_direction_y = pad_sequence(path_direction_y, batch_first=True)
    length = np.array(length)

    idx_shuffle = np.random.choice(path_x.shape[0], path_x.shape[0], replace=False)
    if args.shuffle:
        path_x, path_y, path_direction_x, path_direction_y, length = \
            path_x[idx_shuffle], path_y[idx_shuffle], path_direction_x[idx_shuffle], path_direction_y[idx_shuffle], length[idx_shuffle]
    train_samples, val_samples = int(path_x.shape[0] * args.train_ratio), int(path_x.shape[0] * args.val_ratio)

    train_x = path_x[:train_samples, :]
    train_y = path_y[:train_samples, :]
    train_direction_x = path_direction_x[:train_samples, :]
    train_direction_y = path_direction_y[:train_samples, :]
    train_length = length[:train_samples]

    all_train = torch.cat((train_x-1, train_y), dim=-1)
    # print(f'all_train: {all_train.shape}')
    # print(f'all_train, max: {torch.max(all_train)}, min: {torch.min(all_train)}')
    TM = get_transition_matrix(all_train, args.num_edges)

    val_x = path_x[train_samples:train_samples+val_samples, :]
    val_y = path_y[train_samples:train_samples+val_samples, :]
    val_direction_x = path_direction_x[train_samples:train_samples+val_samples, :]
    val_direction_y = path_direction_y[train_samples:train_samples+val_samples, :]
    val_length = length[train_samples:train_samples+val_samples]

    test_x = path_x[train_samples+val_samples:, :]
    test_y = path_y[train_samples+val_samples:, :]
    test_direction_x = path_direction_x[train_samples+val_samples:, :]
    test_direction_y = path_direction_y[train_samples+val_samples:, :]
    test_length = length[train_samples+val_samples:]

    return train_x, train_y, train_direction_x, train_direction_y, train_length.tolist(), \
           val_x, val_y, val_direction_x, val_direction_y, val_length.tolist(), \
           test_x, test_y, test_direction_x, test_direction_y, test_length.tolist(), TM


def load_data(args):
    assert args.dataset in ['chengdu', 'shanghai', 'RCM']

    load_start = time.time()
    print(f'Loading {args.dataset}\'s road network')
    train_loaders, val_loaders, test_loaders = [], [], []

    if args.dataset in ['chengdu']:
        map_dir = os.path.join(args.data_dir, args.dataset, 'map')
        graph, node_adj_edges, nodes_coor, map, length_shortest_paths = construct_graph(map_dir, args)
        args.num_nodes = len(graph.nodes)
        args.num_edges = len(graph.edges)
        graph_edges = list(graph.edges)
        done_graph = time.time()
        print(f'Done loading graph, time: {done_graph - load_start:.3f}')
        unique_drivers, total_orders, total_path, total_GPS_points, total_trajs, max_traj_length, max_GPS_length = \
            set([]), 0, 0, 0, 0, 0, 0

        direction_dir = os.path.join(args.data_dir, args.dataset, 'map', 'direction.npy')
        location_dir = os.path.join(args.data_dir, args.dataset, 'map', 'location.npy')
        # tensor([-180., -135.,  -90.,  -45.,    0.,   45.,   90.,  135.])
        direction_lb = torch.arange(-180., 180., step=360. / args.direction)
        if not os.path.exists(direction_dir) or not os.path.exists(location_dir):
            # 0 ~ num_edges - 1
            all_paths = [i for i in range(len(graph.edges))]
            direction_labels, all_location = [], []
            for path in all_paths:
                x_text, y_text = [], []
                # road: list of geometry, e.g., ['104.100294 30.7147948', '104.1000959 30.7148658']
                road = str(map[path])[12:-1].split(", ")
                # append [lon, lat] of the endpoint of a road
                all_location.append([float(road[-1].split(" ")[0]), float(road[-1].split(" ")[1])])
                for i__, point in enumerate(road):
                    x, y = float(point.split(" ")[0]), float(point.split(" ")[1])
                    # start point of the road
                    if i__ == 0:
                        x_text.append(x)
                        y_text.append(y)
                    # end point of the road
                    if i__ == len(road) - 1:
                        x_text.append(x)
                        y_text.append(y)

                direction_labels.append(get_direction_labels(
                    torch.angle(torch.tensor(complex(x_text[-1] - x_text[0], y_text[-1] - y_text[0]))).item() * 180 / 3.14159, direction_lb))
            direction_labels = np.array(direction_labels)
            locations = np.array(all_location)
            np.save(direction_dir, direction_labels)
            np.save(location_dir, locations)
        else:
            # 0 ~ 7
            direction_labels = np.load(direction_dir)
            # shape: (num_edges, 2), [[lon, lat], [lon, lat], ...], endpoint of a road
            locations = np.load(location_dir)

        if not os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', "loc_nxdist_matrix.npy")):
            loc_nxdist_matrix = get_network_distance(G=graph)
            np.save(os.path.join(args.data_dir, args.dataset, 'map', 'loc_nxdist_matrix.npy'), loc_nxdist_matrix)
        else:
            loc_nxdist_matrix = np.load(os.path.join(args.data_dir, args.dataset, 'map', 'loc_nxdist_matrix.npy'))

        if not os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', 'pw_distances.npy')) or \
            not os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', 'pw_directions.npy')) or \
                not os.path.exists(os.path.join(args.data_dir, args.dataset, 'map', 'loc_direct_matrix.npy')):

            print(f'Calculating pair-wise distances, directions')
            pair_wise_time = time.time()
            # (num_edges, )
            lon, lat = locations[:, 0], locations[:, 1]
            # (num_edges, num_edges)
            lon_diff, lat_diff = lon[:, np.newaxis] - lon, lat[:, np.newaxis] - lat
            # (num_edges, num_edges)
            loc_direct_matrix = (np.arctan2(lat_diff, lon_diff) * 180 / np.pi).T

            # loc_dlabels_matrix: 0 ~ 7, shape: (num_edges, num_edges)
            loc_dlabels_matrix = np.zeros(loc_direct_matrix.shape)
            for i in range(len(direction_lb)):
                if i == len(direction_lb) - 1:
                    mask = (loc_direct_matrix >= direction_lb[i].item()) * 1 * i
                else:
                    mask = (loc_direct_matrix >= direction_lb[i].item()) * (loc_direct_matrix < direction_lb[i+1].item()) * 1 * i
                loc_dlabels_matrix += mask

            loc_dist_matrix = pairwise_distances(locations, metric='euclidean')
            # shape: (num_edges, num_edges)
            loc_dist_matrix = loc_dist_matrix / np.max(loc_dist_matrix)
            print(f'Done pair-wise distances and directions, time: {time.time()-pair_wise_time:.3f}')

            np.save(os.path.join(args.data_dir, args.dataset, 'map', 'pw_distances.npy'), loc_dist_matrix)
            np.save(os.path.join(args.data_dir, args.dataset, 'map', 'pw_directions.npy'), loc_dlabels_matrix)
            np.save(os.path.join(args.data_dir, args.dataset, 'map', 'loc_direct_matrix.npy'), loc_direct_matrix)

        else:
            print(f'Loading pair-wise distances, directions')
            loc_dist_matrix = np.load(os.path.join(args.data_dir, args.dataset, 'map', 'pw_distances.npy'))
            loc_dlabels_matrix = np.load(os.path.join(args.data_dir, args.dataset, 'map', 'pw_directions.npy'))
            loc_direct_matrix = np.load(os.path.join(args.data_dir, args.dataset, 'map', 'loc_direct_matrix.npy'))
            print(f'Done loading pair-wise distances, directions')

        # edge_A: (num_edges, num_edges)
        edge_A_dir = os.path.join(args.data_dir, args.dataset, 'map', 'edge_A.npy')
        if not os.path.exists(edge_A_dir):
            edge_A = get_adj_matrix(graph)
            np.save(edge_A_dir, edge_A)
        else:
            edge_A = np.load(edge_A_dir)

        path_x, path_y, path_direction_x, path_direction_y, length = [], [], [], [], []
        for i in range(1, args.days+1):
            if len(str(i)) == 1:
                day = '0' + str(i)
            else:
                day = str(i)

            start_data = time.time()
            # data_dir = os.path.join(args.data_dir, args.dataset, 'cleaned', str(args.min_length), day + '.npy')
            data_dir = os.path.join(args.data_dir, args.dataset, 'new_processed', f'thresh_10', day + '.npy')
            data = np.load(data_dir, allow_pickle=True).item()
            done_data = time.time()
            print(f'Done loading {day}\'s data, time: {done_data - start_data:.3f}')

            unique_drivers = unique_drivers.union(set(data.keys()))

            for driver in data.keys():
                driver_data = data[driver]
                for order in driver_data.keys():
                    '''
                    matched_paths: 0 ~ num_edges-1
                    matched_paths_directions: 0~7
                    path_direction_x: 1~8
                    path_direction_y: 0~7
                    path_x: 1 ~ num_edges
                    path_y: 0 ~ num_edges-1
                    '''
                    if len(driver_data[order]) == 4:
                        mapped_trajs, matched_paths_each_point, matched_paths, _ = driver_data[order]
                    else:
                        mapped_trajs, matched_paths_each_point, matched_paths = driver_data[order]
                    # matched_paths_directions = direction_data[driver][order]
                    # print(f'mapped_trajs: {len(mapped_trajs)}\n{mapped_trajs}')
                    # print(f'matched_paths_each_point: {len(matched_paths_each_point)}\n{matched_paths_each_point}')
                    # print(f'matched_paths: {len(matched_paths)}\n{matched_paths}')
                    matched_paths_directions = direction_labels[matched_paths]
                    # if len(matched_paths) >= args.min_length:
                    if len(matched_paths) >= args.seq_len + args.pre_len:
                        # padding for pad_sequence
                        path_direction_x.append(torch.LongTensor(matched_paths_directions[-args.pre_len - args.seq_len:-args.pre_len]) + 1)
                        path_x.append(torch.LongTensor(matched_paths[-args.pre_len - args.seq_len:-args.pre_len]) + 1)
                        # path_x.append(torch.LongTensor(matched_paths[:-args.pre_len]) + 1)
                        path_y.append(torch.LongTensor(matched_paths[-args.pre_len:]))
                        path_direction_y.append(torch.LongTensor(matched_paths_directions[-args.pre_len:]))
                        length.append(len(torch.LongTensor(matched_paths[-args.pre_len - args.seq_len:-args.pre_len])))
                        # length.append(len(torch.LongTensor(matched_paths[:-args.pre_len])))

                        if len(matched_paths) > max_traj_length:
                            max_traj_length = len(matched_paths)
                        if len(matched_paths_each_point) > max_GPS_length:
                            max_GPS_length = len(matched_paths_each_point)
                        total_orders += 1
                        total_trajs += 1
                        total_path += len(matched_paths)
                        total_GPS_points += len(matched_paths_each_point)

            # path_x, path_y, path_direction_x, path_direction_y, length
            # print(f'path_x: {torch.max(torch.stack(path_x))}')

            # print(f'path_x: {torch.max(torch.stack(path_x)), torch.min(torch.stack(path_x))}')
            # print(f'path_y: {torch.max(torch.stack(path_y)), torch.min(torch.stack(path_y))}')
            # print(f'path_direction_x: {torch.max(torch.stack(path_direction_x)), torch.min(torch.stack(path_direction_x))}')
            # print(f'path_direction_y: {torch.max(torch.stack(path_direction_y)), torch.min(torch.stack(path_direction_y))}')
            # print(f'length: {torch.min(torch.tensor(length)), torch.min(torch.tensor(length))}')
            # assert 1==2

            del data
            gc.collect()

    elif args.dataset in ['shanghai']:
        args.data_dir = '/home/yihong/RNTraj/data'
        map_dir = os.path.join(args.data_dir, args.dataset, 'map')
        graph, node_adj_edges, nodes_coor, map = construct_graph(map_dir, args)
        args.num_nodes = len(graph.nodes)
        args.num_edges = len(graph.edges)
        graph_edges = list(graph.edges)
        done_graph = time.time()
        print(f'Done loading graph, time: {done_graph - load_start:.3f}')

        start_data = time.time()
        args.data_dir = '/home/yihong/data'
        data_dir = os.path.join(args.data_dir, args.dataset, 'thresh_data', f'thresh_{args.min_length}', f'thresh_{args.min_length}.npy')
        data = np.load(data_dir, allow_pickle=True).item()
        done_data = time.time()
        print(f'Done loading data, time: {done_data - start_data:.3f}')

        location_dir = os.path.join(args.data_dir, args.dataset, 'map', 'location.npy')
        direction_dir = os.path.join(args.data_dir, args.dataset, 'map', 'direction.npy')
        if not os.path.exists(direction_dir) or not os.path.exists(location_dir):
            all_paths = [i for i in range(len(graph.edges))]
            # node_seq = []
            # for i_, path in enumerate(all_paths):
            #     start, end, _ = graph_edges[path]
            #     if i_ == 0:
            #         node_seq.append(start)
            #         node_seq.append(end)
            #     else:
            #         node_seq.append(end)
            # matched_paths_directions = []
            # for i_ in range(len(node_seq) - 1):
            #     s_lon, s_lat = nodes_coor[node_seq[i_] - 1, 0], nodes_coor[node_seq[i_] - 1, 1]
            #     e_lon, e_lat = nodes_coor[node_seq[i_ + 1] - 1, 0], nodes_coor[node_seq[i_ + 1] - 1, 1]
            #     direction = torch.angle(torch.tensor(complex(e_lon - s_lon, e_lat - s_lat))).item() * 180 / 3.14159
            #     matched_paths_directions.append(direction)
            # map_directions = np.array(matched_paths_directions)
            # np.save(direction_dir, map_directions)
            direction_lb = torch.arange(-180., 180., step=360. / args.direction)
            direction_labels, all_location = [], []
            for path in all_paths:
                x_text, y_text = [], []
                road = str(map[path])[12:-1].split(", ")
                all_location.append([float(road[-1].split(" ")[0]), float(road[-1].split(" ")[1])])
                for i__, point in enumerate(road):
                    x, y = float(point.split(" ")[0]), float(point.split(" ")[1])
                    if i__ == 0:
                        x_text.append(x)
                        y_text.append(y)
                    if i__ == len(road) - 1:
                        x_text.append(x)
                        y_text.append(y)
                # print(f'x: {len(x_text)}, y: {len(y_text)}')
                direction_labels.append(get_direction_labels(
                    torch.angle(torch.tensor(complex(x_text[-1] - x_text[0], y_text[-1] - y_text[0]))).item() * 180 / 3.14159, direction_lb))

            direction_labels = np.array(direction_labels)
            locations = np.array(all_location)
            # print(direction_labels.shape)
            # assert 1==2
            np.save(direction_dir, direction_labels)
            np.save(location_dir, locations)
        else:
            direction_labels = np.load(direction_dir)
            locations = np.load(location_dir)

        # direction_lb = torch.arange(-180., 180., step=360. / args.direction)
        # direction_labels = []
        # for di in range(len(map_directions)):
        #     direction_labels.append(get_direction_labels(map_directions[di], direction_lb))
        # direction_labels = np.array(direction_labels)

        # direction_dir = os.path.join(args.data_dir, args.dataset, 'thresh_data', f'thresh_{args.min_length}', 'direction')
        # if not os.path.exists(direction_dir):
        #     os.mkdir(direction_dir)
        # k_directions_dir = os.path.join(direction_dir, f'{args.direction}')
        # if not os.path.exists(k_directions_dir):
        #     os.mkdir(k_directions_dir)
        # if not os.path.exists(os.path.join(k_directions_dir, f'direction.npy')):
        #     # print(f'Calculating directions ...')
        #     direction_start = time.time()
        #     new_data = {}
        #     for driver in data.keys():
        #         driver_data = data[driver]
        #         new_driver_data = []
        #         for i_ in range(len(driver_data)):
        #             order = driver_data[i_]
        #             mapped_trajs, matched_paths_each_point, matched_paths = order
        #
        #             node_seq = []
        #             for i_, path in enumerate(matched_paths):
        #                 start, end, _ = graph_edges[path]
        #                 if i_ == 0:
        #                     node_seq.append(start)
        #                     node_seq.append(end)
        #                 else:
        #                     node_seq.append(end)
        #             matched_paths_directions = []
        #             for i_ in range(len(node_seq) - 1):
        #                 s_lon, s_lat = nodes_coor[node_seq[i_] - 1, 0], nodes_coor[node_seq[i_] - 1, 1]
        #                 e_lon, e_lat = nodes_coor[node_seq[i_ + 1] - 1, 0], nodes_coor[node_seq[i_ + 1] - 1, 1]
        #                 direction = torch.angle(torch.tensor(complex(e_lon - s_lon, e_lat - s_lat))).item() * 180 / 3.14159
        #                 direction_lb = torch.arange(-180., 180., step=360. / args.direction)
        #                 matched_paths_directions.append(get_direction_labels(direction, direction_lb))
        #
        #             new_driver_data.append(matched_paths_directions)
        #         new_data[driver] = new_driver_data
        #
        #     np.save(os.path.join(k_directions_dir, f'direction.npy'), new_data)
        #     direction_data = new_data
        # else:
        #     # print(f'loading directions ...')
        #     direction_start = time.time()
        #     direction_data = np.load(os.path.join(k_directions_dir, f'direction.npy'), allow_pickle=True).item()
        # print(f'Done direction data, time: {time.time() - direction_start:.3f}')

        unique_drivers, total_orders, total_path, total_GPS_points, total_trajs, max_traj_length, max_GPS_length = \
            set([]), 0, 0, 0, 0, 0, 0
        path_x, path_y, path_direction_x, path_direction_y, length = [], [], [], [], []
        # max_path = -1
        for driver in data.keys():
            driver_data = data[driver]
            # for order in driver_data:
            #     mapped_trajs, matched_paths_each_point, matched_paths = order


            for i_ in range(len(driver_data)):
                order = driver_data[i_]
                mapped_trajs, matched_paths_each_point, matched_paths = order
                # matched_paths_directions = direction_data[driver][i_]
                # print(f'mapped_trajs: {len(mapped_trajs)}\n{mapped_trajs}')
                # print(f'matched_paths_each_point: {len(matched_paths_each_point)}\n{matched_paths_each_point}')
                # print(f'matched_paths: {len(matched_paths)}\n{matched_paths}')
                # print(f'matched_paths_directions: {len(matched_paths_directions)}\n{matched_paths_directions}')
                # assert 1==2
                matched_paths_directions = direction_labels[matched_paths]

                if len(matched_paths) >= args.seq_len + args.pre_len:
                    # padding for pad_sequence
                    path_direction_x.append(torch.LongTensor(matched_paths_directions[-args.pre_len - args.seq_len:-args.pre_len]) + 1)
                    path_x.append(torch.LongTensor(matched_paths[-args.pre_len - args.seq_len:-args.pre_len]) + 1)
                    # path_x.append(torch.LongTensor(matched_paths[:-args.pre_len]) + 1)
                    path_y.append(torch.LongTensor(matched_paths[-args.pre_len:]))
                    path_direction_y.append(torch.LongTensor(matched_paths_directions[-args.pre_len:]))
                    length.append(len(torch.LongTensor(matched_paths[-args.pre_len - args.seq_len:-args.pre_len])))
                    # length.append(len(torch.LongTensor(matched_paths[:-args.pre_len])))

                    if len(matched_paths) > max_traj_length:
                        max_traj_length = len(matched_paths)
                    if len(matched_paths_each_point) > max_GPS_length:
                        max_GPS_length = len(matched_paths_each_point)
                    total_orders += 1
                    total_trajs += 1
                    total_path += len(matched_paths)
                    total_GPS_points += len(matched_paths_each_point)

    elif args.dataset in ['RCM']:
        RCM_dir = os.path.join(args.data_dir, 'RCM_shanghai', 'data')
        graph, node_adj_edges, nodes_coor, map, length_shortest_paths = construct_graph_RCM_SH(RCM_dir, args)
        args.num_nodes = len(graph.nodes)
        args.num_edges = len(graph.edges)
        graph_edges = list(graph.edges)
        done_graph = time.time()
        print(f'Done loading graph, time: {done_graph - load_start:.3f}')
        unique_drivers, total_orders, total_path, total_GPS_points, total_trajs, max_traj_length, max_GPS_length = set([]), 0, 0, 0, 0, 0, 0

        direction_dir = os.path.join(RCM_dir, 'map', f'direction_{args.direction}.npy')
        location_dir = os.path.join(RCM_dir, 'map', 'location.npy')
        # tensor([-180., -135.,  -90.,  -45.,    0.,   45.,   90.,  135.])
        direction_lb = torch.arange(-180., 180., step=360. / args.direction)
        if not os.path.exists(direction_dir) or not os.path.exists(location_dir):
            # 0 ~ num_edges - 1
            all_paths = [i for i in range(len(graph.edges))]
            direction_labels, all_location = [], []
            for path in all_paths:
                x_text, y_text = [], []
                # road: list of geometry, e.g., ['104.100294 30.7147948', '104.1000959 30.7148658']
                road = str(map[path])[12:-1].split(", ")
                # append [lon, lat] of the endpoint of a road
                all_location.append([float(road[-1].split(" ")[0]), float(road[-1].split(" ")[1])])
                for i__, point in enumerate(road):
                    x, y = float(point.split(" ")[0]), float(point.split(" ")[1])
                    # start point of the road
                    if i__ == 0:
                        x_text.append(x)
                        y_text.append(y)
                    # end point of the road
                    if i__ == len(road) - 1:
                        x_text.append(x)
                        y_text.append(y)

                direction_labels.append(get_direction_labels(
                    torch.angle(torch.tensor(complex(x_text[-1] - x_text[0], y_text[-1] - y_text[0]))).item() * 180 / 3.14159, direction_lb))
            direction_labels = np.array(direction_labels)
            locations = np.array(all_location)
            np.save(direction_dir, direction_labels)
            np.save(location_dir, locations)
        else:
            # 0 ~ 7
            direction_labels = np.load(direction_dir)
            # shape: (num_edges, 2), [[lon, lat], [lon, lat], ...], endpoint of a road
            locations = np.load(location_dir)

        if not os.path.exists(os.path.join(RCM_dir, 'map', "loc_nxdist_matrix.npy")):
            loc_nxdist_matrix = get_network_distance(G=graph)
            np.save(os.path.join(RCM_dir, 'map', 'loc_nxdist_matrix.npy'), loc_nxdist_matrix)
        else:
            loc_nxdist_matrix = np.load(os.path.join(RCM_dir, 'map', 'loc_nxdist_matrix.npy'))

        if not os.path.exists(os.path.join(RCM_dir, 'map', 'pw_distances.npy')) or \
                not os.path.exists(os.path.join(RCM_dir, 'map', 'pw_directions.npy')) or \
                not os.path.exists(os.path.join(RCM_dir, 'map', 'loc_direct_matrix.npy')):

            print(f'Calculating pair-wise distances, directions')
            pair_wise_time = time.time()
            # (num_edges, )
            lon, lat = locations[:, 0], locations[:, 1]
            # (num_edges, num_edges)
            lon_diff, lat_diff = lon[:, np.newaxis] - lon, lat[:, np.newaxis] - lat
            # (num_edges, num_edges)
            loc_direct_matrix = (np.arctan2(lat_diff, lon_diff) * 180 / np.pi).T

            # loc_dlabels_matrix: 0 ~ 7, shape: (num_edges, num_edges)
            loc_dlabels_matrix = np.zeros(loc_direct_matrix.shape)
            for i in range(len(direction_lb)):
                if i == len(direction_lb) - 1:
                    mask = (loc_direct_matrix >= direction_lb[i].item()) * 1 * i
                else:
                    mask = (loc_direct_matrix >= direction_lb[i].item()) * (loc_direct_matrix < direction_lb[i + 1].item()) * 1 * i
                loc_dlabels_matrix += mask

            loc_dist_matrix = pairwise_distances(locations, metric='euclidean')
            # shape: (num_edges, num_edges)
            loc_dist_matrix = loc_dist_matrix / np.max(loc_dist_matrix)
            print(f'Done pair-wise distances and directions, time: {time.time() - pair_wise_time:.3f}')

            np.save(os.path.join(RCM_dir, 'map', 'pw_distances.npy'), loc_dist_matrix)
            np.save(os.path.join(RCM_dir, 'map', 'pw_directions.npy'), loc_dlabels_matrix)
            np.save(os.path.join(RCM_dir, 'map', 'loc_direct_matrix.npy'), loc_direct_matrix)

        else:
            print(f'Loading pair-wise distances, directions')
            loc_dist_matrix = np.load(os.path.join(RCM_dir, 'map', 'pw_distances.npy'))
            loc_dlabels_matrix = np.load(os.path.join(RCM_dir, 'map', 'pw_directions.npy'))
            loc_direct_matrix = np.load(os.path.join(RCM_dir, 'map', 'loc_direct_matrix.npy'))
            print(f'Done loading pair-wise distances, directions')

        # edge_A: (num_edges, num_edges)
        edge_A_dir = os.path.join(RCM_dir, 'map', 'edge_A.npy')
        if not os.path.exists(edge_A_dir):
            edge_A = get_adj_matrix(graph)
            np.save(edge_A_dir, edge_A)
        else:
            edge_A = np.load(edge_A_dir)

        if os.path.exists(os.path.join(RCM_dir, 'path.csv')):
            all_path = pd.read_csv(os.path.join(RCM_dir, 'path.csv')).to_numpy()
            # print(f'all_path: {all_path.shape}\n{all_path}')
        else:
            raise FileNotFoundError

        state_action = np.load(os.path.join(RCM_dir, 'state_action.npy'))
        state_action = state_action[:-1, :]

        # thresh out unsatisfied routes, all_path: 0 ~ 713
        # assert 1==2
        all_path = all_path[np.where(all_path[:, -1] >= args.seq_len + args.pre_len)[0]]
        path_x, path_y, path_direction_x, path_direction_y, length = [], [], [], [], []
        unique_drivers, total_orders, total_path, total_GPS_points, total_trajs, max_traj_length, max_GPS_length = set([]), 0, 0, 0, 0, 0, 0

        for i in range(all_path.shape[0]):
            whole_path = np.array([int(i) for i in all_path[i, 2].split("_")])
            # print(f'whole_path: {len(whole_path)}\n{whole_path}')
            path_x.append(torch.LongTensor(whole_path[-args.pre_len - args.seq_len:-args.pre_len]) + 1)
            path_y.append(torch.LongTensor(whole_path[-args.pre_len:]))
            # print(f'direction_labels: {direction_labels.shape}\n{direction_labels}')
            path_direction_x.append((torch.LongTensor(direction_labels[whole_path[-args.pre_len - args.seq_len:-args.pre_len]]) + 1))
            path_direction_y.append((torch.LongTensor(direction_labels[whole_path[-args.pre_len:]])))
            length.append(len(torch.LongTensor(whole_path[-args.pre_len - args.seq_len:-args.pre_len])))

            if len(whole_path) > max_traj_length:
                max_traj_length = len(whole_path)
            max_GPS_length = 0
            total_orders += 1
            total_trajs += 1
            total_path += len(whole_path)
            total_GPS_points = 0


    # print(f'path_x: {len(path_x)}\n{path_x[0].shape}')
    # print(f'path_y: {len(path_y)}\n{path_y[0].shape}')
    # print(f'path_direction_x: {len(path_direction_x)}\n{path_direction_x[0].shape}')
    # print(f'path_direction_y: {len(path_direction_y)}\n{path_direction_y[0].shape}')
    # print(f'length: {len(length)}\n{length}')
    # assert 1==2

    train_x, train_y, train_direction_x, train_direction_y, train_length, \
    val_x, val_y, val_direction_x, val_direction_y, val_length, \
    test_x, test_y, test_direction_x, test_direction_y, test_length, TM = \
        split_data(path_x, path_y, path_direction_x, path_direction_y, length, args)

    print(f'train_x: {train_x.shape} | val_x: {val_x.shape} | test_x: {test_x.shape}')
    print(f'train_y: {train_y.shape} | val_y: {val_y.shape} | test_y: {test_y.shape}')
    # print(f'test: {torch.cat((test_x-1, test_y), dim=-1)}')
    torch.save(torch.cat((train_x-1, train_y), dim=-1), '/home/yihong/RL_learn/RCM-AIRL/data/chengdu/cd7_train.pt')
    torch.save(torch.cat((val_x-1, val_y), dim=-1), '/home/yihong/RL_learn/RCM-AIRL/data/chengdu/cd7_val.pt')
    torch.save(torch.cat((test_x-1, test_y), dim=-1), '/home/yihong/RL_learn/RCM-AIRL/data/chengdu/cd7_test.pt')
    assert 1==2

    train_loader = TrajDataLoader(x=train_x, y=train_y, direction_x=train_direction_x, direction_y=train_direction_y,
                                  length=train_length, batch_size=args.batch_size, pad_with_last_sample=True)
    val_loader = TrajDataLoader(x=val_x, y=val_y, direction_x=val_direction_x, direction_y=val_direction_y,
                                length=val_length, batch_size=args.batch_size, pad_with_last_sample=True)
    test_loader = TrajDataLoader(x=test_x, y=test_y, direction_x=test_direction_x, direction_y=test_direction_y,
                                 length=test_length, batch_size=args.batch_size, pad_with_last_sample=True)

    train_loaders.append(train_loader)
    val_loaders.append(val_loader)
    test_loaders.append(test_loader)

    print(f'================= Statistics =================\n'
          # f'Total {len(unique_drivers)} drivers, {total_orders} orders, {total_orders / len(unique_drivers):.3f} orders per driver\n'
          f'Total {total_trajs} trajectories, {total_path} paths, {total_GPS_points} GPS points, {total_GPS_points / total_trajs:.3f} GPS points per trajectory \n'
          f'max_traj_length: {max_traj_length}, max_GPS_length: {max_GPS_length}\n'
          f'Num days: {args.days}\n'
          # f'Num train trajectories: {train_x.shape[0]}, val trajectories: {val_x.shape[0]}, test trajectories: {test_x.shape[0]}\n'
          f'Road network: {len(graph.nodes)} nodes, {len(graph.edges)} edges')

    '''
    Return
    
    graph: networkx MultiDiGraph
    node_adj_edges: numpy array, shape: (num_nodes, ), with different length in the second dimension
    direction_labels: 0 ~ 7, shape: (num_edges, )
    loc_direct_matrix: -180. ~ 180., shape: (num_edges, num_edges)
    loc_dist_matrix: 0. ~ 1., shape: (num_edges, num_edges)
    loc_dlabels_matrix: 0 ~ 7, shape: (num_edges, num_edges)
    '''

    # print(f'node_adj_edges: {node_adj_edges.shape}\n{node_adj_edges}')
    # print(f'graph_edges: {len(graph_edges)}\n{graph_edges}')
    #
    # adj_edge_pair = []
    # for edge_id, (_, end_node, _) in enumerate(graph_edges):
    #     # print(f'edge: {edge_id} | end_node: {end_node-1}')
    #     end_node -= 1
    #     edge_adj_list = node_adj_edges[end_node]
    #     action_mask = [True for _ in range(8)]
    #     for adj_edge in edge_adj_list:
    #         dlabel = int(loc_dlabels_matrix[(edge_id, adj_edge)])
    #         # print(f'edge_id: {edge_id} | adj_edge: {adj_edge} | dlabel: {dlabel}')
    #         if action_mask[dlabel] == True:
    #             action_mask[dlabel] = False
    #         else:
    #             while action_mask[dlabel] != True:
    #                 dlabel += 1
    #                 if dlabel == 8:
    #                     dlabel = 0
    #             action_mask[dlabel] = False
    #         adj_edge_pair.append([edge_id, dlabel, adj_edge])
    #
    # transit = np.array(adj_edge_pair)
    # print(f'transit: {transit.shape}\n{transit}')
    # np.save('transit.npy', transit)
    # assert 1==2

    args.direction_dim = args.edge_dim
    args.connection_dim = args.edge_dim
    args.consistent_dim = args.edge_dim
    args.distance_dim = args.edge_dim
    if args.dataset in ['RCM']:
        args.state_action = torch.from_numpy(state_action).to(args.device)

    return [train_loaders, val_loaders, test_loaders], graph, node_adj_edges, direction_labels, loc_direct_matrix, \
           loc_dist_matrix, loc_dlabels_matrix, TM, edge_A, length_shortest_paths