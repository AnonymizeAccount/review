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
        self.transition_matrix = torch.FloatTensor(TM).to(self.args.device)
        self.direction_labels = torch.tensor(direction_labels).to(self.args.device)
        self.loc_direct_matrix = torch.tensor(loc_direct_matrix).to(self.args.device)
        self.loc_dist_matrix = torch.tensor(loc_dist_matrix).to(self.args.device)
        self.loc_dlabels_matrix = torch.LongTensor(loc_dlabels_matrix).to(self.args.device)
        self.graph_edges = torch.LongTensor(np.array(self.graph.edges)).to(self.args.device)
        self.batch_long = torch.ones(self.args.batch_size).to(self.args.device) * self.args.pre_len

        node_adj_edges_list = []
        for i in range(self.node_adj_edges.shape[0]):
            node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
        self.node_adj_edges = \
            pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)
        self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * float('-inf'), dim=-1).to(self.args.device)
        self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
        self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)

        self.link_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
        self.direction_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.sim_criterion = nn.KLDivLoss(reduction='batchmean')

    def compute_loss(self, pred, gt, pred_d, gt_d):
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])

        return loss

    def get_end_nodes(self, pred, gt, dim, end_node, cur_parent):
        cur_gt = gt[:, dim]
        cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
        cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
        node_adj_edges = self.node_adj_edges[end_node]
        last_pred = cur_parent.pred
        last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
        node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges
        cur_pred = cur_pred[self.ids, node_adj_edges]
        cur_pred = cur_pred.topk(k=self.args.multi, dim=1, largest=True, sorted=True)
        cur_pred_value, cur_pred = cur_pred[0], cur_pred[1]
        row, col = torch.where(cur_pred_value == float('-inf'))
        cur_pred.index_put_((row, col), cur_pred[row][:, 0])
        cur_pred_value.index_put_((row, col), cur_pred_value[row][:, 0])
        cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
        wrong_pred_idx = torch.where(cur_pred == self.args.num_edges)

        cur_pred[torch.where(cur_pred == self.args.num_edges)] -= self.offset
        for k in range(self.args.multi):
            cur_pred_k = cur_pred[:, k]
            end_node = self.graph_edges[cur_pred_k][:, 1] - 1
            end_node_k = Node(name=k, parent=cur_parent, data=end_node, pred=cur_pred_k)

    def get_multi_predictions(self, pred, pred_d, gt, direction_gt, obs=None):

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
        # print(self.args.topk)
        while k < topk:
            if root.leaves[k].depth != self.args.pre_len:
                continue
            upwards, common, downwards = walker.walk(root, root.leaves[k])
            # print(f'node: {common.data.shape} {common.data}')
            # pred_k = torch.unsqueeze(common.data, dim=-1)
            pred_k = None
            for node in downwards:
                # print(f'node: {node.data.shape} {node.data}')
                if pred_k == None:
                    pred_k = torch.unsqueeze(node.pred, dim=-1)
                else:
                    pred_k = torch.cat((pred_k, torch.unsqueeze(node.pred, dim=-1)), dim=-1)
            # print(f'pred_{k}: {pred_k.shape}\n{pred_k}')
            if preds == None:
                preds = torch.unsqueeze(pred_k, dim=-1)
            else:
                preds = torch.cat((preds, torch.unsqueeze(pred_k, dim=-1)), dim=-1)
        # print(list(walker.walk(root, root.leaves[0])))
            k += 1
        return preds

    def get_predictions(self, pred, pred_d, gt, direction_gt, obs=None):
        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1
        path_total, path_right, d_total, d_right, traj_total, traj_right, prediction, prediction_d, last_pred = 0, 0, 0, 0, 0, None, None, None, None
        for dim in range(gt.shape[1]):
            cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
            cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
            node_adj_edges = self.node_adj_edges[end_node]
            if dim == 0:
                last_pred = obs
            last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
            node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges
            cur_pred = cur_pred[self.ids, node_adj_edges]
            cur_pred = cur_pred.max(1)[1][:, None]
            cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
            cur_pred[torch.where(cur_pred == self.args.num_edges)[0]] -= self.offset
            last_pred = cur_pred
            if dim != gt.shape[1] - 1:
                end_node = self.graph_edges[cur_pred][:, 1] - 1
            cur_pred_d = F.log_softmax(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], dim=1)
            cur_pred_d = cur_pred_d.max(1)[1]

            if prediction == None:
                prediction = torch.unsqueeze(cur_pred, dim=-1)
                prediction_d = torch.unsqueeze(cur_pred_d, dim=-1)
            else:
                prediction = torch.cat((prediction, torch.unsqueeze(cur_pred, dim=-1)), dim=-1)
                prediction_d = torch.cat((prediction_d, torch.unsqueeze(cur_pred_d, dim=-1)), dim=-1)

        return prediction, prediction_d

    def new_compute_acc(self, pred, pred_d, gt, direction_gt, obs=None, type=None, epoch=None):
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

    # Knowledge Graph (CE) + sim
    def forward(self, inputs, directions, mask):
        link_embedding = self.link_emblayer.weight[1:, :]
        reconstructed_transition = torch.matmul(link_embedding, link_embedding.T)
        # print(F.softmax(reconstructed_transition, dim=1))
        # print(F.softmax(self.transition_matrix, dim=1) * self.edge_A + torch.eye(link_embedding.shape[0]).to(self.args.device))
        loss = self.sim_criterion(F.softmax(reconstructed_transition, dim=1),
                                  F.softmax(self.transition_matrix, dim=1) * self.edge_A + torch.eye(link_embedding.shape[0]).to(self.args.device))
        # print(f'loss: {loss}')
        # assert 1==2
        link_embs = self.link_emblayer(inputs[:, -1])
        direction_embs = self.direction_emblayer(directions[:, -1])

        sim_score_link = torch.matmul((link_embs + direction_embs), self.link_emblayer.weight[1:, :].T)
        pred = sim_score_link.repeat(1, self.args.pre_len)
        pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)

        return pred, pred_d, loss