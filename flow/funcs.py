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
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics.pairwise import pairwise_distances, haversine_distances
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, Walker


def print_graph_attr(G):
    # print number of nodes and edges
    print("Number of nodes:", len(G.nodes))
    print("Number of edges:", len(G.edges))

    # print degree statistics
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    print("Average in-degree:", sum(in_degrees.values()) / len(in_degrees))
    print("Average out-degree:", sum(out_degrees.values()) / len(out_degrees))
    print("Maximum in-degree:", max(in_degrees.values()))
    print("Maximum out-degree:", max(out_degrees.values()))
    print("Minimum in-degree:", min(in_degrees.values()))
    print("Minimum out-degree:", min(out_degrees.values()))

    # print diameter
    print("Diameter:", nx.diameter(G))

    # print density
    print("Density:", nx.density(G))

    # print other available attributes
    print("Available attributes:")
    for attr in G.graph:
        print(f"- {attr}: {G.graph[attr]}")


def get_network_distance(G):
    edges = list(G.edges)
    # value: 1 ~ num_nodes
    nodes = list(G.nodes)

    loc_nxdist_matrix = np.zeros((len(edges), len(edges)))
    for i, s_edge in tqdm(enumerate(edges)):
        s_edge_s_node, s_edge_e_node, _ = s_edge

        for j, e_edge in enumerate(edges):
            if e_edge == s_edge:
                continue

            e_edge_s_node, e_edge_e_node, _ = e_edge

            try:
                path = nx.shortest_path(G, source=s_edge_s_node, target=e_edge_e_node)
            except nx.exception.NetworkXNoPath:
                continue

            loc_nxdist_matrix[i][j] = len(path) - 1

    return loc_nxdist_matrix


def get_adj_matrix(G):
    # shape: (num_edges, num_edges)
    edge_A = np.zeros((len(G.edges), len(G.edges)))

    print(f'Calculating Edge-Adjacency Matrix...')
    start_time = time.time()
    for idx_1, (start_node_1, end_node_1, _) in enumerate(G.edges):
        for idx_2, (start_node_2, end_node_2, _) in enumerate(G.edges):
            if start_node_2 == end_node_1:
                edge_A[idx_1][idx_2] = 1

    print(f'Done, time elapse: {time.time()-start_time:.3f}')
    return edge_A


def get_transition_matrix(seqs, num_edges):
    '''
    seqs: 0 ~ num_edges-1
    '''
    print(f'Computing transition matrix')
    start_time = time.time()

    TM = np.zeros((num_edges, num_edges))
    for r in tqdm(range(seqs.shape[0])):
        traj = seqs[r]
        for c in range(seqs.shape[1]-1):
            TM[traj[c]][traj[c+1]] += 1

    print(f'Done computing, time: {time.time() - start_time}')
    return TM


def construct_graph_RCM_SH(RCM_dir, args):
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

    if not os.path.exists(os.path.join(RCM_dir, 'map', 'shortest_paths.npy')) or \
        not os.path.exists(os.path.join(RCM_dir, 'map', 'length_shortest_paths.npy')):

        shortest_paths, length_shortest_paths = [], []
        for edge_1_id, edge_1 in enumerate(list(G.edges)):
            # print(edge_1_id)
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
        np.save(os.path.join(RCM_dir, 'map', 'shortest_paths.npy'), shortest_paths)
        np.save(os.path.join(RCM_dir, 'map', 'length_shortest_paths.npy'), length_shortest_paths)
    else:
        sp_time = time.time()
        # shortest_paths = np.load(os.path.join(RCM_dir, 'map', 'shortest_paths.npy'), allow_pickle=True)
        length_shortest_paths = np.load(os.path.join(RCM_dir, 'map', 'length_shortest_paths.npy'), allow_pickle=True)
        # print(f'shortest_paths: {shortest_paths.shape}')
        # print(f'length_shortest_paths: max: {np.max(length_shortest_paths)}, min: {np.min(length_shortest_paths)}, {length_shortest_paths.shape}\n{length_shortest_paths}')
        print(f'Done Loading shortest paths, time: {time.time() - sp_time:.3f}')

    if os.path.exists(os.path.join(RCM_dir, 'map', 'node_adj_edges_efficient.npy')):
        node_adj_edges_efficient = np.load(os.path.join(RCM_dir, 'map', 'node_adj_edges_efficient.npy'), allow_pickle=True)
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
        np.save(os.path.join(RCM_dir, 'map', 'node_adj_edges_efficient.npy'), node_adj_edges_efficient)
    node_adj_edges = node_adj_edges_efficient

    return G, node_adj_edges, nodes_coor, map, length_shortest_paths


# This compute ranking_loss compare to the new_compute_acc function
def compute_recall_mrr(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood, batch_long,
                        obs=None, type=None, epoch=None, model=None, mlp_topk=None, preds_ori=None, preds_d=None):
    '''
    pred: edge predictions,                     shape: (batch_size, num_edges * pre_len),           value: 0 ~ num_edges - 1 (expected)
    pred_d: directions predictions,             shape: (batch_size, num_directions * pre_len),      value: 0 ~ num_directions - 1 (expected)
    gt: actual future edges,                    shape: (batch_size, pre_len),                       value: 0 ~ num_edges - 1
    direction_gt: actual future directions,     shape: (batch_size, pre_len),                       value: 0 ~ num_directions - 1
    obs: last edges in input sequences,         shape: (batch_size),                                value: 0 ~ num_edges - 1
    batch_long:  [pre_len, pre_len, ...]        shape: (batch_size)
    '''
    best_path_total, best_path_right, best_d_total, best_d_right, best_traj_total, best_traj_right, best_prediction = 0, 0, 1, 0, 0, 0, None
    topk_list = [1, 5, 10, 20, args.multi**args.pre_len]
    loss_ranking = 0
    if args.multi > 1:
        # preds: (batch_size, pre_len, topk), values: 0 ~ num_edges - 1
        # preds_d: (batch_size, pre_len)
        multi_start = time.time()
        rerank_start = time.time()
        if preds_ori is None:
            preds_ori, preds_d = get_multi_predictions(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset,
                                                       padding_loglikelihood, obs)
            loss_ranking, preds_ori, top1_pred = model.rerank(preds_ori, gt_ori)
            mlp_topk, _ = get_predictions(top1_pred, pred_d, gt_ori, direction_gt, args, graph_edges,
                                          all_node_adj_edges, ids, offset, padding_loglikelihood, obs)
            mlp_topk = torch.unsqueeze(mlp_topk, dim=-1)
            if mlp_topk is not None:
                # preds_ori[:, :, :args.mlpk] = mlp_topk[..., :args.mlpk]
                preds_ori = torch.cat((mlp_topk[..., :args.mlpk], preds_ori), dim=-1)
                preds_ori = preds_ori[:, :, :topk_list[-1]]
        end_time = time.time()
        if type == 'topk':
            k_best_path_right, k_best_traj_right, k_best_d_right, k_mrr = [], [], [], []
            for topk in topk_list:
                preds = preds_ori[:, :, :topk]
                # (batch_size, topk, pre_len)
                gt = torch.unsqueeze(gt_ori, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
                # (batch_size, topk, pre_len)
                preds = preds.permute(0, 2, 1)

                print(gt.shape, preds.shape)
                equal = torch.eq(gt, preds)
                print(f'equal: {equal.shape}\n{equal}')
                equal = torch.all(equal, dim=-1)
                print(f'equal: {equal.shape}\n{equal}')
                cols = torch.argmax(equal * 1, dim=1) + 1
                print(f'cols: {cols.shape}\n{cols}')
                rows = torch.unique(torch.where(equal == True)[0])
                print(f'rows: {rows.shape}\n{rows}')
                mrr = torch.sum(1 / cols[rows])
                print(f'mrr: {mrr.shape}\n{mrr}')
                # assert 1==2
                # if args.multi**args.pre_len == topk:
                #     # print(f'gt: {gt.shape}, preds: {preds.shape}')
                #     # print(f'gt: {gt[2, ...]}')
                #     # print(f'preds: {preds[2, ...]}')
                #     # print(f'gt_ori: {gt_ori.shape}')
                #     equal = torch.all(torch.eq(gt, preds), dim=-1)
                #     cols = torch.argmax(equal * 1, dim=1) + 1
                #     # print(f'cols: {cols.shape}\n{cols}')
                #     rows = torch.unique(torch.where(equal == True)[0])
                #     # print(f'rows: {rows.shape}\n{rows}')
                #     mrr = torch.sum(1 / cols[rows]) / gt.shape[0]
                #     # print(f'mrr: {mrr.shape}\n{mrr}')
                #     # assert 1==2

                # (batch_size, topk)
                ele_right = torch.sum((gt == preds) * 1, dim=-1)
                # (batch_size), find best in topk
                batch_right = torch.max(ele_right, dim=-1)[0]
                best_traj_right = torch.sum((batch_right == batch_long) * 1)
                best_path_right = torch.sum(batch_right)
                d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
                best_d_right = torch.sum(d_ele_right)
                k_best_path_right.append(best_path_right.item())
                k_best_traj_right.append(best_traj_right.item())
                k_mrr.append(mrr)
        else:
            if args.topk > 0:
                preds = preds_ori[:, :, :args.topk]
            else:
                preds = preds_ori

            # (batch_size, topk, pre_len)
            gt = torch.unsqueeze(gt_ori, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
            # (batch_size, topk, pre_len)
            preds = preds.permute(0, 2, 1)
            # preds_unique, inverse_indices = torch.unique(preds[:2, ...], dim=1, return_inverse=True)

            # print(f'gt: {gt.shape}')
            # print(f'preds: {preds.shape}')

            equal = torch.all(torch.eq(gt, preds), dim=-1)
            cols = torch.argmax(equal * 1, dim=1) + 1
            rows = torch.unique(torch.where(equal == True)[0])
            # mrr = torch.sum(1 / cols[rows]) / gt.shape[0]
            mrr = torch.sum(1 / cols[rows])

            # assert 1==2

            # (batch_size, topk)
            ele_right = torch.sum((gt == preds) * 1, dim=-1)
            # (batch_size), find best in topk
            batch_right = torch.max(ele_right, dim=-1)[0]
            best_traj_right = torch.sum((batch_right == batch_long) * 1)
            best_path_right = torch.sum(batch_right)
            d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
            best_d_right = torch.sum(d_ele_right)
    else:
        preds, preds_d = get_predictions(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood,
                                         obs)
        ele_right = torch.sum((gt_ori == preds) * 1, dim=-1)
        best_traj_right = torch.sum((ele_right == batch_long) * 1)
        best_path_right = torch.sum(ele_right)
        d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
        best_d_right = torch.sum(d_ele_right)

    best_traj_total = args.batch_size
    best_path_total = args.batch_size * args.pre_len
    best_d_total = best_path_total

    if type != 'topk':
        return mrr, best_path_total, best_path_right, best_traj_total, best_traj_right, best_d_total, best_d_right, loss_ranking, preds_ori, \
               end_time, end_time-rerank_start, rerank_start-multi_start
    else:
        return k_mrr, best_path_total, k_best_path_right, best_traj_total, k_best_traj_right, best_d_total, best_d_right, loss_ranking, preds_ori, \
               end_time, end_time-rerank_start, rerank_start-multi_start


# This compute ranking_loss compare to the new_compute_acc function
def compute_acc(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood, batch_long,
                    obs=None, type=None, epoch=None, model=None, mlp_topk=None, preds_ori=None, preds_d=None):
    '''
    pred: edge predictions,                     shape: (batch_size, num_edges * pre_len),           value: 0 ~ num_edges - 1 (expected)
    pred_d: directions predictions,             shape: (batch_size, num_directions * pre_len),      value: 0 ~ num_directions - 1 (expected)
    gt: actual future edges,                    shape: (batch_size, pre_len),                       value: 0 ~ num_edges - 1
    direction_gt: actual future directions,     shape: (batch_size, pre_len),                       value: 0 ~ num_directions - 1
    obs: last edges in input sequences,         shape: (batch_size),                                value: 0 ~ num_edges - 1
    batch_long:  [pre_len, pre_len, ...]        shape: (batch_size)
    '''
    best_path_total, best_path_right, best_d_total, best_d_right, best_traj_total, best_traj_right, best_prediction = 0, 0, 1, 0, 0, 0, None
    topk_list = [1, 5, 10, 20, args.multi**args.pre_len]
    loss_ranking = 0
    if args.multi > 1:
        # preds: (batch_size, pre_len, topk), values: 0 ~ num_edges - 1
        # preds_d: (batch_size, pre_len)
        if preds_ori is None:
            preds_ori, preds_d = get_multi_predictions(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset,
                                                       padding_loglikelihood, obs)
            # preds_ori = torch.cat((preds_ori, mlp_topk[..., :args.mlpk]), dim=-1)
            loss_ranking, preds_ori, top1_pred = model.rerank(preds_ori, gt_ori)
            mlp_topk, _ = get_predictions(top1_pred, pred_d, gt_ori, direction_gt, args, graph_edges,
                                       all_node_adj_edges, ids, offset, padding_loglikelihood, obs)
            mlp_topk = torch.unsqueeze(mlp_topk, dim=-1)

            if mlp_topk is not None:
                # preds_ori[:, :, :args.mlpk] = mlp_topk[..., :args.mlpk]
                preds_ori = torch.cat((mlp_topk[..., :args.mlpk], preds_ori), dim=-1)
                preds_ori = preds_ori[:, :, :topk_list[-1]]
        if type == 'topk':
            k_best_path_right, k_best_traj_right, k_best_d_right = [], [], []
            for topk in topk_list:
                preds = preds_ori[:, :, :topk]
                # (batch_size, topk, pre_len)
                gt = torch.unsqueeze(gt_ori, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
                # (batch_size, topk, pre_len)
                preds = preds.permute(0, 2, 1)
                # (batch_size, topk)
                ele_right = torch.sum((gt == preds) * 1, dim=-1)
                # (batch_size), find best in topk
                batch_right = torch.max(ele_right, dim=-1)[0]
                best_traj_right = torch.sum((batch_right == batch_long) * 1)
                best_path_right = torch.sum(batch_right)
                d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
                best_d_right = torch.sum(d_ele_right)
                k_best_path_right.append(best_path_right.item())
                k_best_traj_right.append(best_traj_right.item())
        else:
            if args.topk > 0:
                preds = preds_ori[:, :, :args.topk]
            else:
                preds = preds_ori

            # (batch_size, topk, pre_len)
            gt = torch.unsqueeze(gt_ori, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
            # (batch_size, topk, pre_len)
            preds = preds.permute(0, 2, 1)
            preds_unique, inverse_indices = torch.unique(preds[:2, ...], dim=1, return_inverse=True)
            # (batch_size, topk)
            ele_right = torch.sum((gt == preds) * 1, dim=-1)
            # (batch_size), find best in topk
            batch_right = torch.max(ele_right, dim=-1)[0]
            best_traj_right = torch.sum((batch_right == batch_long) * 1)
            best_path_right = torch.sum(batch_right)
            d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
            best_d_right = torch.sum(d_ele_right)
    else:
        preds, preds_d = get_predictions(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood,
                                         obs)
        ele_right = torch.sum((gt_ori == preds) * 1, dim=-1)
        best_traj_right = torch.sum((ele_right == batch_long) * 1)
        best_path_right = torch.sum(ele_right)
        d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
        best_d_right = torch.sum(d_ele_right)

    best_traj_total = args.batch_size
    best_path_total = args.batch_size * args.pre_len
    best_d_total = best_path_total

    if type != 'topk':
        return best_path_total, best_path_right, best_traj_total, best_traj_right, best_d_total, best_d_right, loss_ranking, preds_ori
    else:
        return best_path_total, k_best_path_right, best_traj_total, k_best_traj_right, best_d_total, best_d_right, loss_ranking, preds_ori


def new_compute_acc(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood, batch_long,
                    obs=None, type=None, epoch=None):
    '''
    pred: edge predictions,                     shape: (batch_size, num_edges * pre_len),           value: 0 ~ num_edges - 1 (expected)
    pred_d: directions predictions,             shape: (batch_size, num_directions * pre_len),      value: 0 ~ num_directions - 1 (expected)
    gt: actual future edges,                    shape: (batch_size, pre_len),                       value: 0 ~ num_edges - 1
    direction_gt: actual future directions,     shape: (batch_size, pre_len),                       value: 0 ~ num_directions - 1
    obs: last edges in input sequences,         shape: (batch_size),                                value: 0 ~ num_edges - 1
    batch_long:  [pre_len, pre_len, ...]        shape: (batch_size)
    '''
    best_path_total, best_path_right, best_d_total, best_d_right, best_traj_total, best_traj_right, best_prediction = 0, 0, 1, 0, 0, 0, None
    topk_list = [1, 5, 10, 20, args.multi**args.pre_len]
    if args.multi > 1:
        # preds: (batch_size, pre_len, topk), values: 0 ~ num_edges - 1
        # preds_d: (batch_size, pre_len)
        multi_start = time.time()
        preds_ori, preds_d = get_multi_predictions(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset,
                                                   padding_loglikelihood, obs)
        end_time = time.time()
        if type == 'topk':
            k_best_path_right, k_best_traj_right, k_best_d_right, k_mrr = [], [], [], []
            for topk in topk_list:
                preds = preds_ori[:, :, :topk]
                # (batch_size, topk, pre_len)
                gt = torch.unsqueeze(gt_ori, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
                # (batch_size, topk, pre_len)
                preds = preds.permute(0, 2, 1)

                equal = torch.eq(gt, preds)
                # print(f'equal: {equal.shape}\n{equal}')
                equal = torch.all(equal, dim=-1)
                # print(f'equal: {equal.shape}\n{equal}')
                cols = torch.argmax(equal * 1, dim=1) + 1
                # print(f'cols: {cols.shape}\n{cols}')
                rows = torch.unique(torch.where(equal == True)[0])
                # print(f'rows: {rows.shape}\n{rows}')
                mrr = torch.sum(1 / cols[rows])

                # (batch_size, topk)
                ele_right = torch.sum((gt == preds) * 1, dim=-1)
                # (batch_size), find best in topk
                batch_right = torch.max(ele_right, dim=-1)[0]
                best_traj_right = torch.sum((batch_right == batch_long) * 1)
                best_path_right = torch.sum(batch_right)
                d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
                best_d_right = torch.sum(d_ele_right)
                k_best_path_right.append(best_path_right.item())
                k_best_traj_right.append(best_traj_right.item())
                k_mrr.append(mrr)
        else:
            if args.topk > 0:
                preds = preds_ori[:, :, :args.topk]
            else:
                preds = preds_ori
            # (batch_size, topk, pre_len)
            gt = torch.unsqueeze(gt_ori, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
            # (batch_size, topk, pre_len)
            preds = preds.permute(0, 2, 1)

            equal = torch.all(torch.eq(gt, preds), dim=-1)
            cols = torch.argmax(equal * 1, dim=1) + 1
            rows = torch.unique(torch.where(equal == True)[0])
            # mrr = torch.sum(1 / cols[rows]) / gt.shape[0]
            mrr = torch.sum(1 / cols[rows])

            # (batch_size, topk)
            ele_right = torch.sum((gt == preds) * 1, dim=-1)
            # (batch_size), find best in topk
            batch_right = torch.max(ele_right, dim=-1)[0]
            best_traj_right = torch.sum((batch_right == batch_long) * 1)
            best_path_right = torch.sum(batch_right)
            d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
            best_d_right = torch.sum(d_ele_right)
    else:
        preds, preds_d = get_predictions(pred, pred_d, gt_ori, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood,
                                         obs)
        ele_right = torch.sum((gt_ori == preds) * 1, dim=-1)
        best_traj_right = torch.sum((ele_right == batch_long) * 1)
        best_path_right = torch.sum(ele_right)
        d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
        best_d_right = torch.sum(d_ele_right)

    best_traj_total = args.batch_size
    best_path_total = args.batch_size * args.pre_len
    best_d_total = best_path_total

    if type != 'topk':
        return mrr, best_path_total, best_path_right, best_traj_total, best_traj_right, best_d_total, best_d_right, preds_ori, \
               end_time, end_time-multi_start
    else:
        return k_mrr, best_path_total, k_best_path_right, best_traj_total, k_best_traj_right, best_d_total, best_d_right, preds_ori, \
               end_time, end_time-multi_start
        
        
def get_predictions(pred, pred_d, gt, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood, obs=None):
    '''
    pred: edge predictions,                     shape: (batch_size, num_edges * pre_len),           value: 0 ~ num_edges - 1 (expected)
    pred: directions predictions,               shape: (batch_size, num_directions * pre_len),      value: 0 ~ num_directions - 1 (expected)
    gt: actual future edges,                    shape: (batch_size, pre_len),                       value: 0 ~ num_edges - 1
    direction_gt: actual future directions,     shape: (batch_size, pre_len),                       value: 0 ~ num_directions - 1
    obs: last edges in input sequences,         shape: (batch_size),                                value: 0 ~ num_edges - 1
    self.graph_edges:                           shape: (num_edges, 3),                              value: 1 ~ num_nodes,                          e.g.: [[start_node, end_node, 0], ...]
    self.node_adj_edges:                        shape: (num_nodes, max_adj_edges),                  value: 0 ~ num_edges - 1 with the padding_value == num_edges
    self.ids:                                   shape: (batch_size, 1)                              value: 0 ~ num_edges - 1
    '''

    # (batch_size), indicating the end_node of the last observed edge, which is also the start_node of the first gt edges. value: 0 ~ num_nodes-1
    end_node = graph_edges[gt[:, 0]][:, 0] - 1
    path_total, path_right, d_total, d_right, traj_total, traj_right, prediction, prediction_d, last_pred = 0, 0, 0, 0, 0, None, None, None, None
    # (batch_size), value: 0 ~ num_edges - 1
    last_pred = obs

    for dim in range(gt.shape[1]):
        # (batch_size, num_edges)
        cur_pred = F.log_softmax(pred[:, dim * args.num_edges: (dim + 1) * args.num_edges], dim=1)
        # (batch_size, num_edges + 1), padding_loglikelihood == -inf
        cur_pred = torch.cat((cur_pred, padding_loglikelihood), dim=-1)
        # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
        node_adj_edges = all_node_adj_edges[end_node]
        # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
        last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
        # mask out last_pred, no turning back
        node_adj_edges[torch.where(last_pred == node_adj_edges)] = args.num_edges
        # (batch_size, num_edges + 1) -> (batch_size, max_adj_edges), values: log_likelihood
        cur_pred = cur_pred[ids, node_adj_edges]
        # (batch_size, 1), value: idx of max likelihood
        cur_pred = cur_pred.max(1)[1][:, None]
        # (batch_size), values: 0 ~ num_edges - 1, predictions of current time step
        cur_pred = torch.squeeze(node_adj_edges[ids, cur_pred])
        # (batch_size), continue training when there is no edge
        cur_pred[torch.where(cur_pred == args.num_edges)[0]] -= offset
        # (batch_size),
        last_pred = cur_pred
        if dim != gt.shape[1] - 1:
            # (batch_size), indicating the end_node of the last observed edge. value: 0 ~ num_nodes-1
            end_node = graph_edges[cur_pred][:, 1] - 1
        # (batch_size, num_directions)
        cur_pred_d = F.log_softmax(pred_d[:, dim * args.direction: (dim + 1) * args.direction], dim=1)
        # (batch_size), values: 0 ~ num_directions - 1, predictions of directions.
        cur_pred_d = cur_pred_d.max(1)[1]
        if prediction == None:
            prediction = torch.unsqueeze(cur_pred, dim=-1)
            prediction_d = torch.unsqueeze(cur_pred_d, dim=-1)
        else:
            prediction = torch.cat((prediction, torch.unsqueeze(cur_pred, dim=-1)), dim=-1)
            prediction_d = torch.cat((prediction_d, torch.unsqueeze(cur_pred_d, dim=-1)), dim=-1)

    # prediction: shape: (batch_size, pre_len), values: 0 ~ num_edges - 1
    # prediction_d: shape: (batch_size, pre_len), values: 0 ~ num_directions - 1

    return prediction, prediction_d


def get_end_nodes(pred, dim, end_node, cur_parent, args, padding_loglikelihood, graph_edges, ids, offset, all_node_adj_edges):
    _start = time.time()
    # (batch_size, num_edges)
    # cur_pred = F.log_softmax(pred[:, dim * args.num_edges: (dim + 1) * args.num_edges], dim=1)
    cur_pred = F.softmax(pred[:, dim * args.num_edges: (dim + 1) * args.num_edges], dim=1)
    # (batch_size, num_edges + 1), self.padding_loglikelihood == -inf
    cur_pred = torch.cat((cur_pred, padding_loglikelihood), dim=-1)
    # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
    node_adj_edges = all_node_adj_edges[end_node]
    # (batch_size), value: 0 ~ num_edges - 1
    last_pred = cur_parent.pred
    # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
    last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
    # mask out last_pred, no turning back
    node_adj_edges[torch.where(last_pred == node_adj_edges)] = args.num_edges
    # (batch_size, num_edges + 1) -> (batch_size, max_adj_edges), values: log_likelihood
    cur_pred = cur_pred[ids, node_adj_edges]
    # cur_pred_value: (batch_size, args.multi), values: log likelihood
    cur_pred = cur_pred.topk(k=args.multi, dim=1, largest=True, sorted=True)
    # cur_pred_value: (batch_size, args.multi), values: idx
    cur_pred_value, cur_pred = cur_pred[0], cur_pred[1]
    # values: 0 ~ batch_size - 1
    row, col = torch.where(cur_pred_value == float('-inf'))
    # if num_adj < multi, pad with the fiTrst edge and corresponding value
    cur_pred.index_put_((row, col), cur_pred[row][:, 0])
    cur_pred_value.index_put_((row, col), cur_pred_value[row][:, 0])
    # (batch_size, args.multi), values: 0 ~ num_edges - 1, predictions of current time step
    cur_pred = torch.squeeze(node_adj_edges[ids, cur_pred])
    # continue training when there is no edge
    cur_pred[torch.where(cur_pred == args.num_edges)] -= offset

    for k in range(args.multi):
        # (batch_size)
        cur_pred_k = cur_pred[:, k]
        # (batch_size)
        end_node = graph_edges[cur_pred_k][:, 1] - 1
        # add leaves
        end_node_k = Node(name=k, parent=cur_parent, data=end_node, pred=cur_pred_k)


def get_multi_predictions(pred, pred_d, gt, direction_gt, args, graph_edges, all_node_adj_edges, ids, offset, padding_loglikelihood, obs=None):

    # (batch_size), indicating the end_node of the last observed edge, which is also the start_node of the first gt edges. value: 0 ~ num_nodes-1
    end_node = graph_edges[gt[:, 0]][:, 0] - 1
    prediction_d = None

    root = Node(name="root", data=end_node, pred=obs)
    for dim in range(args.pre_len):
        cur_pred_d = F.log_softmax(pred_d[:, dim * args.direction: (dim + 1) * args.direction], dim=1)
        # (batch_size), values: 0 ~ num_directions - 1, predictions of directions.
        cur_pred_d = cur_pred_d.max(1)[1]
        if prediction_d == None:
            prediction_d = torch.unsqueeze(cur_pred_d, dim=-1)
        else:
            prediction_d = torch.cat((prediction_d, torch.unsqueeze(cur_pred_d, dim=-1)), dim=-1)

        # if dim > 0:
        #     for pre, _, node in RenderTree(root):
        #         print(f"{pre}{node.name}")

        cur_leaves = root.leaves
        for node in cur_leaves:
            cur_parent = node
            end_node = node.data
            get_end_nodes(pred, dim, end_node, cur_parent, args, padding_loglikelihood, graph_edges, ids, offset, all_node_adj_edges)

        # if dim > 0:
        #     for pre, _, node in RenderTree(root):
        #         print(f"{pre}{node.name}")

    # assert 1==2

    walker = Walker()

    preds, k = None, 0
    if args.topk < 0:
        topk = np.power(args.multi, args.pre_len)
    else:
        topk = args.topk

    while k < topk:
        if root.leaves[k].depth != args.pre_len:
            continue
        upwards, common, downwards = walker.walk(root, root.leaves[k])
        pred_k = None
        for node in downwards:
            if pred_k == None:
                pred_k = torch.unsqueeze(node.pred, dim=-1)
            else:
                pred_k = torch.cat((pred_k, torch.unsqueeze(node.pred, dim=-1)), dim=-1)

        if preds == None:
            preds = torch.unsqueeze(pred_k, dim=-1)
        else:
            preds = torch.cat((preds, torch.unsqueeze(pred_k, dim=-1)), dim=-1)
        k += 1

    # (batch_size, pre_len, topk)
    return preds, prediction_d
