import os
import copy
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd

from tqdm import tqdm


def eval_one_epoch(loaders, args, G, G_edges):
    time_10k_list = []

    # for _ in range(20):
    #     route, traj, all_route, all_traj = 0, 0, 0, 0
    #     start_time = time.time()
    route, traj, all_route, all_traj = 0, 0, 0, 0
    for loader in loaders:
        for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
            # print(f'x: {x.shape}, y: {y.shape}')
            # all_route += x.shape[0] * (args.pre_len - 1)
            all_route += x.shape[0] * args.pre_len
            all_traj += x.shape[0]
            for j in range(x.shape[0]):
                # print(G_edges[x[j][-1]-1])
                # print(np.concatenate((G_edges[x[j][-1]-1][1], G_edges[y[j]][:, 1][:-1]), axis=0))
                # print(np.insert(G_edges[y[j]][:, 1], 0, G_edges[x[j][-1]-1][1]))
                # gt = np.insert(G_edges[y[j]][:, 1], 0, G_edges[x[j][-1]-1][1])
                source = G_edges[x[j][-1]-1][0]
                target = G_edges[y[j][-1]][1]
                # print(f'source: {source}, target: {target}')
                # paths = np.array(nx.shortest_path(G, source=source, target=target)[1:])
                # nx.all_shortest_paths(G, source=source, target=target)
                # print(G[source][target])

                # paths = np.array([np.array(p) for p in nx.all_shortest_paths(G, source=source, target=target, method='bellman-ford')])
                # paths = np.array([np.array(p) for p in nx.all_shortest_paths(G, source=source, target=target)])
                # paths = np.array([np.array(p) for p in nx.all_shortest_paths(G, source=source, target=target, method='bellman-ford', weight='length')])
                paths = np.array([np.array(p) for p in nx.all_shortest_paths(G, source=source, target=target, method='bellman-ford', weight='length')])
                # paths = np.array([np.array(p) for p in nx.all_shortest_paths(G, source=source, target=target, weight='length')])

                idx = np.random.randint(0, high=len(paths), size=1)

                # paths = paths[idx[0]][1:-1]
                paths = paths[idx[0]][1:]

                # print(f'paths: {paths}')
                # assert 1==2

                edge_path = []
                for k in range(len(paths)-1):
                    edges = G.get_edge_data(paths[k], paths[k+1])
                    idx = np.random.randint(0, high=len(edges), size=1)[0]
                    edge_path.append(edges[idx]['idx'])
                    # if len(G.get_edge_data(paths[k], paths[k+1])) > 1:
                    #     print(G.get_edge_data(paths[k], paths[k+1])[0])
                    #     assert 1==2
                edge_path = np.array(edge_path)
                # print(f'y[j]: {y[j]}')
                # print(f'edge_path: {edge_path.shape}\n{edge_path}')
                # assert 1==2

                # correct = np.sum((edge_path == y[j][:-1]) * 1)
                correct = np.sum((edge_path == y[j]) * 1)
                route += correct

                # if correct == args.pre_len-1:
                if correct == args.pre_len:
                    traj += 1


        # end_time = time.time()

    print(route / all_route, traj / all_traj)
    # print(f'Time: {end_time-start_time:.3f}?')
    #
    #     time_10k = (end_time - start_time) / all_traj * 1e4 * 1e3
    #     print(f'time_10k: {time_10k}')
    #
    #     time_10k_list.append(time_10k)
    #
    # print(f'mean: {np.mean(time_10k_list)}, std: {np.std(time_10k_list)}')


def train(loader, args, model, optimizer):
    if args.dataset == 'chengdu':
        dir = '/home/yihong/RNTraj/data/chengdu/map'
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
        # print(f'original_nodes: \n{original_nodes}')
        # assert 1==2

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
        edges = edges.astype(np.int)
        # nodes: 2D numpy array, shape: (num_nodes, 4), [[node_osmid, lat, lon, street_count], ...]
        nodes = original_nodes.to_numpy()[:, :4]

        # lookup_idx_osmid: {1: node_osmid, 2: node_osmid, ...}, idx: 1 ~ num_nodes
        # lookup_osmid_idx: {node_osmid: 1, node_osmid: 2, ...}, value: 1 ~ num_nodes
        G = nx.MultiDiGraph()
        print(f'Reading nodes...')
        for nid in tqdm(range(nodes.shape[0])):
            lookup_idx_osmid[nid + 1] = int(nodes[nid, 0])
            # 1 ~ num_nodes
            lookup_osmid_idx[int(nodes[nid, 0])] = nid + 1
            G.add_node(nid + 1, lat=nodes[nid, 1], lon=nodes[nid, 2], street_count=nodes[nid, 3])

        print(f'Reading edges...')
        # idx=eid, eid: 0 ~ num_edges-1
        # for eid in tqdm(range(edges.shape[0])):
        for eid in range(edges.shape[0]):
            # print(f'f: {original_edges.iloc[eid]["length"]}')
            # if original_edges.iloc[eid]['length'] < 0.01:
            #     assert 1==2
            G.add_edge(lookup_osmid_idx[edges[eid, 0]], lookup_osmid_idx[edges[eid, 1]], highway=original_edges.iloc[eid]['highway'],
                       oneway=original_edges.iloc[eid]['oneway'], length=float(original_edges.iloc[eid]['length']),
                       geometry=str(original_edges.iloc[eid]['geometry'])[12:-1], idx=eid)
        # assert 1==2

    elif args.dataset == 'RCM':
        RCM_dir = os.path.join(args.data_dir, 'RCM_shanghai', 'data')

        print(f'RCM_dir: {RCM_dir}')
        edge = pd.read_csv(os.path.join(RCM_dir, 'edge.txt'), delimiter=',')
        node = pd.read_csv(os.path.join(RCM_dir, 'node.txt'), delimiter=',')
        point = pd.read_csv(os.path.join(RCM_dir, 'network_point.csv'), delimiter=',')
        vertices = pd.read_csv(os.path.join(RCM_dir, 'network_vertices.csv'), delimiter=',')

        nodes_coor = node[['x', 'y']].to_numpy()

        edge["geometry"] = [""] * len(edge)
        node["street_count"] = [0] * len(node)
        node = node.rename(columns={"x": "lon", "y": "lat"})

        nodes = node.to_numpy()
        edges = edge.to_numpy()[:, :2].astype(np.int)

        point['u'] = point['u'].round().astype('int')
        point['v'] = point['v'].round().astype('int')
        point['source'] = point['source'].round().astype('int')
        point['target'] = point['target'].round().astype('int')

        vertices['u'] = vertices['u'].round().astype('int')
        vertices['v'] = vertices['v'].round().astype('int')
        vertices['source'] = vertices['source'].round().astype('int')
        vertices['target'] = vertices['target'].round().astype('int')

        # print(f'point: {point.shape}\n{point}')
        # print(f'vertices: {vertices.shape}\n{vertices}')

        unique_osmid_pairs = np.vstack(edge.groupby(['u', 'v']).apply(lambda x: (x.name[0], x.name[1])).unique())
        # print(f'unique_osmid_pairs: {unique_osmid_pairs.shape}\n{unique_osmid_pairs}')

        for i in tqdm(range(unique_osmid_pairs.shape[0])):
            # print(f'uv: {unique_osmid_pairs[i][0], unique_osmid_pairs[i][1]}')
            geometry_uv_pair = point[(point['u'] == unique_osmid_pairs[i][0]) & (point['v'] == unique_osmid_pairs[i][1])][["lng", "lat"]].to_numpy()
            # print(geometry_uv_pair.shape)
            # print(geometry_uv_pair)
            geometry = "LINESTRING ("
            for j in range(geometry_uv_pair.shape[0]):
                geometry += str(geometry_uv_pair[j][0]) + " " + str(geometry_uv_pair[j][1])
                if j != geometry_uv_pair.shape[0] - 1:
                    geometry += ", "
            geometry += ")"

            row_index = edge.loc[(edge['u'] == unique_osmid_pairs[i][0]) & (edge['v'] == unique_osmid_pairs[i][1])].index[0]
            edge.loc[row_index, "geometry"] = geometry
            # print(f'{geometry}')

        # print(f'nodes: \n{nodes}')
        # print(f'edges: \n{edges}')
        # print(f'node: \n{node}')
        # print(f'edge: \n{edge}')
        map = edge.to_numpy()[:, -1]

        lookup_idx_osmid, lookup_osmid_idx = {}, {}
        # lookup_idx_osmid: {1: node_osmid, 2: node_osmid, ...}, idx: 1 ~ num_nodes
        # lookup_osmid_idx: {node_osmid: 1, node_osmid: 2, ...}, value: 1 ~ num_nodes
        G = nx.MultiDiGraph()
        print(f'Reading nodes...')
        for nid in tqdm(range(nodes.shape[0])):
            lookup_idx_osmid[nid + 1] = int(nodes[nid, 0])
            # 1 ~ num_nodes
            lookup_osmid_idx[int(nodes[nid, 0])] = nid + 1
            G.add_node(nid + 1, lat=float(nodes[nid, 1]), lon=float(nodes[nid, 2]), street_count=int(nodes[nid, 3]))

        print(f'Reading edges...')
        # idx=eid, eid: 0 ~ num_edges-1
        for eid in tqdm(range(edges.shape[0])):
            G.add_edge(lookup_osmid_idx[edges[eid, 0]], lookup_osmid_idx[edges[eid, 1]], highway=edge.iloc[eid]['highway'],
                       oneway=edge.iloc[eid]['oneway'], length=edge.iloc[eid]['length'],
                       geometry=str(edge.iloc[eid]['geometry'])[12:-1], idx=eid)

    # line_G = nx.line_graph(G)
    # print(line_G.nodes)
    # assert 1==2

    G_edges = np.array(G.edges)
    train_loaders, val_loaders, test_loaders = loader
    best_acc, best_traj_acc, best_d_acc, best_goal_d_acc, count_patience, state, dur = -1, -1, -1, 0, 0, None, []

    start_time = time.time()
    eval_one_epoch(test_loaders, args, G, G_edges)
    dur.append(time.time() - start_time)