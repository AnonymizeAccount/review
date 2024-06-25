import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.nn.modules.rnn import RNNCellBase


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, loc_dlabels_matrix):
        super(RNN, self).__init__()
        self.args = args
        self.node_adj_edges = node_adj_edges
        self.loc_dlabels_matrix = torch.LongTensor(loc_dlabels_matrix).to(self.args.device)
        self.graph = graph
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

        self.enc = nn.LSTM(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1, batch_first=True,
                           dropout=self.args.dropout).to(self.args.device)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.out_link = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)
        self.out_direction = nn.Linear(self.args.hidden_dim, self.args.direction * self.args.pre_len).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt, pred_d, gt_d, mask=None):

        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
            loss += self.criterion(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], gt_d[:, dim])

        return loss

    def forward(self, inputs, directions, mask, goal, epoch=None, type=None, y=None):
        if self.args.rand:
            pred = torch.rand((self.args.batch_size, self.args.num_edges * self.args.pre_len)).to(self.args.device)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        else:

            last_obs = inputs[:, -1]
            # (batch_size), value: 1 ~ num_edges
            self.goal_directions = self.loc_dlabels_matrix[last_obs - 1, goal] + 1

            link_embs = self.link_emblayer(inputs)
            direction_embs = self.direction_emblayer(directions)
            embs = torch.cat((link_embs, direction_embs), dim=-1)

            l_pack = pack_padded_sequence(embs, mask, batch_first=True, enforce_sorted=False)
            _, (h_n, c_n) = self.enc(l_pack)
            out = torch.squeeze(h_n)

            # (pred_len * edges)
            pred = self.out_link(out)

            # (pred_len * directions)
            pred_d = self.out_direction(out)

        return pred, pred_d, 0, torch.tensor([0])






# class RNN(nn.Module):
#     def __init__(self, args, graph, node_adj_edges, loc_dlabels_matrix):
#         super(RNN, self).__init__()
#         self.args = args
#         self.node_adj_edges = node_adj_edges
#         self.loc_dlabels_matrix = torch.LongTensor(loc_dlabels_matrix).to(self.args.device)
#         self.graph = graph
#         self.graph_edges = torch.LongTensor(np.array(self.graph.edges)).to(self.args.device)
#         self.batch_long = torch.ones(self.args.batch_size).to(self.args.device) * self.args.pre_len
#
#         node_adj_edges_list = []
#         for i in range(self.node_adj_edges.shape[0]):
#             node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
#         self.node_adj_edges = \
#             pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)
#         self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * float('-inf'), dim=-1).to(self.args.device)
#         self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
#         self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)
#
#         self.link_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
#         self.direction_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
#         # self.direction_emblayer = DirectionEmbedding(self.args).to(self.args.device)
#         self.enc = nn.LSTM(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
#                            batch_first=True, dropout=self.args.dropout).to(self.args.device)
#         self.dropout = nn.Dropout(p=self.args.dropout)
#         self.out_direction = nn.Linear(self.args.hidden_dim, self.args.direction * self.args.pre_len).to(self.args.device)
#         self.pad = torch.unsqueeze(torch.tensor([self.args.num_edges for i in range(self.args.batch_size)]).to(self.args.device), dim=-1)
#
#         self.criterion = nn.CrossEntropyLoss()
#
#     def compute_loss(self, pred_d, gt_d, last_obs, gt):
#
#         loss = 0
#         for dim in range(gt_d.shape[1]):
#
#             if dim == 0:
#                 label = self.loc_dlabels_matrix[last_obs - 1, gt[:, dim]]
#             else:
#                 label = self.loc_dlabels_matrix[gt[:, dim-1], gt[:, dim]]
#
#             # print(f'label: {label.shape}\n{label}')
#             # assert 1==2
#             # loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
#             loss += self.criterion(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], label)
#
#         return loss
#
#     def compute_metrics(self, pred_d, gt, gt_d, last_obs):
#
#         pred = None
#         for dim in range(gt.shape[1]):
#             if dim == 0:
#                 end_node = self.graph_edges[last_obs-1][:, 1] - 1
#             else:
#                 end_node = self.graph_edges[torch.squeeze(edge_predictions)][:, 1] - 1
#             # print(f'end_node: {end_node}')
#             batch_adj_edges = self.node_adj_edges[end_node, :]
#             batch_adj_edges = torch.cat((batch_adj_edges, self.pad), dim=1)
#             # print(f'batch_adj_edges: {batch_adj_edges.shape}\n{batch_adj_edges}')
#             batch_mask = (batch_adj_edges != self.args.num_edges) * 1
#             # print(f'batch_mask: {batch_mask.shape}\n{batch_mask}')
#             batch_mask_adj_edges = batch_adj_edges * batch_mask
#             # print(f'batch_mask_adj_edges: {batch_mask_adj_edges.shape}\n{batch_mask_adj_edges}')
#             edge_adj_directions = self.loc_dlabels_matrix[last_obs - 1, :] + 1
#             # print(f'edge_adj_directions: {edge_adj_directions.shape}\n{edge_adj_directions}')
#             batch_mask_adj_edges_directions = torch.gather(edge_adj_directions, 1, batch_mask_adj_edges) * batch_mask
#             # print(f'batch_mask_adj_edges_directions: {batch_mask_adj_edges_directions.shape}\n{batch_mask_adj_edges_directions}')
#
#             # print(f'pred_d: {pred_d.shape}\n{pred_d}')
#             batch_pred = F.log_softmax(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], dim=-1)
#             # print(f'batch_pred: {batch_pred.shape}\n{batch_pred}')
#             _, predicted_directions = torch.max(batch_pred, 1)
#             # print(f'predicted_directions: {predicted_directions.shape}\n{predicted_directions}')
#             # add a column for padding, make it easier to get predictions
#             predicted_directions = torch.unsqueeze(predicted_directions, dim=-1).expand(self.args.batch_size, batch_mask_adj_edges_directions.shape[1])
#             # batch_mask_adj_edges_directions = torch.cat((batch_mask_adj_edges_directions, torch.unsqueeze(predicted_directions[:, -1], dim=-1)), dim=-1)
#             batch_mask_adj_edges_directions[:, -1] = predicted_directions[:, -1]
#             # print(f'predicted_directions: {predicted_directions.shape}\n{predicted_directions}')
#             edge_predictions = (predicted_directions == batch_mask_adj_edges_directions) * 1
#             edge_predictions = torch.gather(batch_adj_edges, 1, torch.unsqueeze(torch.argmax(edge_predictions, dim=1), dim=-1))
#             # print(f'edge_predictions: {edge_predictions.shape}\n{edge_predictions}')
#             rows = torch.where(edge_predictions == self.args.num_edges)[0]
#             edge_predictions[rows] -= self.offset
#             # print(f'edge_predictions: {edge_predictions.shape}, max: {torch.max(edge_predictions)}, min: {torch.min(edge_predictions)}\n{edge_predictions}')
#             if dim == 0:
#                 pred = edge_predictions
#             else:
#                 pred = torch.cat((pred, edge_predictions), dim=1)
#             # assert 1==2
#
#         # print(f'pred: {pred.shape}\n{pred}')
#         # print(f'gt: {gt.shape}\n{gt}')
#
#         ele_right = torch.sum((gt == pred) * 1, dim=-1)
#         best_traj_right = torch.sum((ele_right == self.batch_long) * 1)
#         best_path_right = torch.sum(ele_right)
#
#         # print(f'best_path_right: {best_path_right}')
#         # print(f'best_traj_right: {best_traj_right}')
#
#         return best_traj_right, best_path_right
#
#     def forward(self, inputs, directions, mask, epoch=None, type=None, y=None):
#
#         # last_obs = inputs[:, -1]
#         # (batch_size), value: 1 ~ num_edges
#         # self.goal_directions = self.loc_dlabels_matrix[last_obs - 1, goal] + 1
#
#         # print(f'loc_dlabels_matrix: {self.loc_dlabels_matrix.shape}\n{self.loc_dlabels_matrix}')
#         # print(f'node_adj_edges: {self.node_adj_edges.shape}\n{self.node_adj_edges}')
#         # print(f'self.graph_edges: {self.graph_edges.shape}\n{self.graph_edges}')
#         # print(f'y: {y.shape}')
#
#
#         link_embs = self.link_emblayer(inputs)
#         direction_embs = self.direction_emblayer(directions)
#
#         embs = torch.cat((link_embs, direction_embs), dim=-1)
#
#         l_pack = pack_padded_sequence(embs, mask, batch_first=True, enforce_sorted=False)
#         # if self.args.model == 'lstm':
#         _, (h_n, c_n) = self.enc(l_pack)
#         # else:
#         #     _, h_n = self.enc(l_pack)
#         out = torch.squeeze(h_n)
#
#         # (pred_len * directions)
#         pred_d = self.out_direction(out)
#
#         return pred_d