import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, Walker
from torch.nn.modules.rnn import RNNCellBase


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix, loc_dlabels_matrix, TM, edge_A):
        super(RNN, self).__init__()
        self.args = args
        self.node_adj_edges = node_adj_edges
        self.graph = graph
        self.edge_A = torch.LongTensor(edge_A).to(self.args.device)
        self.transition_matrix = torch.LongTensor(TM).to(self.args.device)
        self.direction_labels = torch.tensor(direction_labels).to(self.args.device)
        self.loc_direct_matrix = torch.tensor(loc_direct_matrix).to(self.args.device)
        self.loc_dist_matrix = torch.tensor(loc_dist_matrix).to(self.args.device)
        self.loc_dlabels_matrix = torch.LongTensor(loc_dlabels_matrix).to(self.args.device)
        # self.graph_edges: shape: (num_edges, 3), value: 1 ~ num_nodes, e.g.: [[start_node, end_node, 0], ...]
        self.graph_edges = torch.LongTensor(np.array(self.graph.edges)).to(self.args.device)
        self.batch_long = torch.ones(self.args.batch_size).to(self.args.device) * self.args.pre_len

        node_adj_edges_list = []
        for i in range(self.node_adj_edges.shape[0]):
            node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
        # (num_nodes, max_adj_edges), values: 0 ~ num_edges - 1 with the padding_value == num_edges
        self.node_adj_edges = pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)
        self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * float('-inf'), dim=-1).to(self.args.device)
        self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
        self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)

        self.link_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
        self.direction_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt, pred_d, gt_d):

        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
            # loss += self.criterion(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], gt_d[:, dim])

        return loss


    def get_end_nodes(self, pred, gt, dim, end_node, cur_parent):
        # (batch_size, num_edges)
        cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
        # (batch_size, num_edges + 1), self.padding_loglikelihood == -inf
        cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
        # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
        node_adj_edges = self.node_adj_edges[end_node]
        # (batch_size), value: 0 ~ num_edges - 1
        last_pred = cur_parent.pred
        # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
        last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
        # mask out last_pred, no turning back
        node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges
        # (batch_size, num_edges + 1) -> (batch_size, max_adj_edges), values: log_likelihood
        cur_pred = cur_pred[self.ids, node_adj_edges]
        # cur_pred_value: (batch_size, self.args.multi), values: log likelihood
        cur_pred = cur_pred.topk(k=self.args.multi, dim=1, largest=True, sorted=True)
        # cur_pred_value: (batch_size, self.args.multi), values: idx
        cur_pred_value, cur_pred = cur_pred[0], cur_pred[1]
        # values: 0 ~ batch_size - 1
        row, col = torch.where(cur_pred_value == float('-inf'))
        # if num_adj < multi, pad with the first edge and corresponding value
        cur_pred.index_put_((row, col), cur_pred[row][:, 0])
        cur_pred_value.index_put_((row, col), cur_pred_value[row][:, 0])
        # (batch_size, self.args.multi), values: 0 ~ num_edges - 1, predictions of current time step
        cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
        # continue training when there is no edge
        cur_pred[torch.where(cur_pred == self.args.num_edges)] -= self.offset
        for k in range(self.args.multi):
            # (batch_size)
            cur_pred_k = cur_pred[:, k]
            # (batch_size)
            end_node = self.graph_edges[cur_pred_k][:, 1] - 1
            # add leaves
            end_node_k = Node(name=k, parent=cur_parent, data=end_node, pred=cur_pred_k)

    def get_multi_predictions(self, pred, pred_d, gt, direction_gt, obs=None):

        # (batch_size), indicating the end_node of the last observed edge, which is also the start_node of the first gt edges. value: 0 ~ num_nodes-1
        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1

        root = Node(name="root", data=end_node, pred=obs)
        for dim in range(self.args.pre_len):
            cur_leaves = root.leaves
            for node in cur_leaves:
                cur_parent = node
                end_node = node.data
                self.get_end_nodes(pred, gt, dim, end_node, cur_parent)
        walker = Walker()

        # print('='*100)
        preds, k = None, 0
        if self.args.topk < 0:
            topk = np.power(self.args.multi, self.args.pre_len)
        else:
            topk = self.args.topk

        while k < topk:
            if root.leaves[k].depth != self.args.pre_len:
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
        return preds

    def get_predictions(self, pred, pred_d, gt, direction_gt, obs=None):
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
        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1
        path_total, path_right, d_total, d_right, traj_total, traj_right, prediction, prediction_d, last_pred = 0, 0, 0, 0, 0, None, None, None, None
        # (batch_size), value: 0 ~ num_edges - 1
        last_pred = obs

        for dim in range(gt.shape[1]):
            # (batch_size, num_edges)
            cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
            # (batch_size, num_edges + 1), self.padding_loglikelihood == -inf
            cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
            # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
            node_adj_edges = self.node_adj_edges[end_node]
            # (batch_size, max_adj_edges), values: 0 ~ num_edges - 1
            last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
            # mask out last_pred, no turning back
            node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges
            # (batch_size, num_edges + 1) -> (batch_size, max_adj_edges), values: log_likelihood
            cur_pred = cur_pred[self.ids, node_adj_edges]
            # (batch_size, 1), value: idx of max likelihood
            cur_pred = cur_pred.max(1)[1][:, None]
            # (batch_size), values: 0 ~ num_edges - 1, predictions of current time step
            cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
            # (batch_size), continue training when there is no edge
            cur_pred[torch.where(cur_pred == self.args.num_edges)[0]] -= self.offset
            # (batch_size),
            last_pred = cur_pred
            if dim != gt.shape[1] - 1:
                # (batch_size), indicating the end_node of the last observed edge. value: 0 ~ num_nodes-1
                end_node = self.graph_edges[cur_pred][:, 1] - 1
            # (batch_size, num_directions)
            cur_pred_d = F.log_softmax(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], dim=1)
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

    def new_compute_acc(self, pred, pred_d, gt, direction_gt, obs=None, type=None, epoch=None):
        '''
        pred: edge predictions,                     shape: (batch_size, num_edges * pre_len),           value: 0 ~ num_edges - 1 (expected)
        pred: directions predictions,               shape: (batch_size, num_directions * pre_len),      value: 0 ~ num_directions - 1 (expected)
        gt: actual future edges,                    shape: (batch_size, pre_len),                       value: 0 ~ num_edges - 1
        direction_gt: actual future directions,     shape: (batch_size, pre_len),                       value: 0 ~ num_directions - 1
        obs: last edges in input sequences,         shape: (batch_size),                                value: 0 ~ num_edges - 1
        '''

        best_path_total, best_path_right, best_d_total, best_d_right, best_traj_total, best_traj_right, best_prediction = 0, 0, 1, 0, 0, 0, None
        preds, preds_d = self.get_predictions(pred, pred_d, gt, direction_gt, obs)

        if self.args.multi > 1:
            preds = self.get_multi_predictions(pred, pred_d, gt, direction_gt, obs=obs)
            if self.args.topk > 0:
                preds = preds[:, :, :self.args.topk]
            gt = torch.unsqueeze(gt, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
            preds = preds.permute(0, 2, 1)
            ele_right = torch.sum((gt == preds) * 1, dim=-1)
            batch_right = torch.max(ele_right, dim=-1)[0]
            best_traj_right = torch.sum((batch_right == self.batch_long) * 1)
            best_path_right = torch.sum(batch_right)
            d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
            best_d_right = torch.sum(d_ele_right)
        else:
            ele_right = torch.sum((gt == preds) * 1, dim=-1)
            best_traj_right = torch.sum((ele_right == self.batch_long) * 1)
            best_path_right = torch.sum(ele_right)
            d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
            best_d_right = torch.sum(d_ele_right)

        best_traj_total = self.args.batch_size
        best_path_total = self.args.batch_size * self.args.pre_len
        best_d_total = best_path_total

        return best_path_total, best_path_right, best_traj_total, best_traj_right, best_d_total, best_d_right, best_prediction, preds

    # Knowledge Graph (CE)
    def forward(self, inputs, directions, mask):
        '''
        inputs:      edge sequences,         shape: (batch_size, seq_len),    value: 1 ~ num_edges
        directions:  direction sequences,    shape: (batch_size, seq_len),    value: 1 ~ num_directions
        mask:        list of valid sequence length for each sequence in the batch, len(mask) == batch_size
        '''

        loss = 0
        if self.args.rand:
            pred = torch.rand((self.args.batch_size, self.args.num_edges * self.args.pre_len)).to(self.args.device)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
            return pred, pred_d, loss
        else:
            # (batch_size, edge_dim)
            link_embs = self.link_emblayer(inputs[:, -1])
            # (batch_size, edge_dim)
            direction_embs = self.direction_emblayer(directions[:, -1])
            # (batch_size, num_edges), this indicate each embedded sequence's similarity to all edges
            sim_score_link = torch.matmul((link_embs+direction_embs), self.link_emblayer.weight[1:, :].T)
            # (batch_size, num_edges * pre_len)
            pred = sim_score_link.repeat(1, self.args.pre_len)
            # (batch_size, num_directions * pre_len)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)

            return pred, pred_d, loss