import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, nodes_links_directions):
        super(RNN, self).__init__()
        self.args = args
        self.node_adj_edges = node_adj_edges
        self.graph = graph
        self.graph_edges = torch.LongTensor(np.array(self.graph.edges)).to(self.args.device)
        self.batch_long = torch.ones(self.args.batch_size).to(self.args.device) * self.args.pre_len

        node_adj_edges_list = []
        for i in range(self.node_adj_edges.shape[0]):
            node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
        self.node_adj_edges = \
            pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)
        self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * -100, dim=-1).to(self.args.device)
        self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
        self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)

        self.link_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
        self.direction_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)

        if self.args.model == 'rnn':
            self.enc = nn.RNN(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'gru':
            self.enc = nn.GRU(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'lstm':
            self.enc = nn.LSTM(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
                               batch_first=True, dropout=self.args.dropout).to(self.args.device)
        else:
            raise ValueError

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.out_link = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)
        self.out_direction = nn.Linear(self.args.hidden_dim, self.args.direction * self.args.pre_len).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt, pred_d, gt_d):

        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
            loss += self.criterion(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], gt_d[:, dim])

        return loss

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
        path_total, path_right, traj_total, traj_right, d_total, d_right, prediction, prediction_d = \
            self.args.batch_size * self.args.pre_len, 0, self.args.batch_size, 0, self.args.batch_size * self.args.pre_len, 0, None, None

        preds, preds_d = self.get_predictions(pred, pred_d, gt, direction_gt, obs)
        ele_right = torch.sum((gt == preds) * 1, dim=-1)
        d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
        traj_right = torch.sum((ele_right == self.batch_long) * 1)
        path_right = torch.sum(ele_right)
        d_right = torch.sum(d_ele_right)

        return path_total, path_right, traj_total, traj_right, d_total, d_right, prediction, preds_d

    def forward(self, inputs, directions, mask):
        if self.args.rand:
            pred = torch.rand((self.args.batch_size, self.args.num_edges * self.args.pre_len)).to(self.args.device)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        else:
            link_embs = self.link_emblayer(inputs)
            direction_embs = self.direction_emblayer(directions)
            embs = torch.cat((link_embs, direction_embs), dim=-1)
            l_pack = pack_padded_sequence(embs, mask, batch_first=True, enforce_sorted=False)
            _, (h_n, c_n) = self.enc(l_pack)
            out = torch.squeeze(h_n)
            pred = self.out_link(out)
            pred_d = self.out_direction(out)

        return pred, pred_d


# def new_compute_acc(self, pred, pred_d, gt, direction_gt, obs=None, type=None, epoch=None):
#     # print(f'pred: {pred.shape}')
#     # print(f'gt: {gt.shape}')
#     # assert 1==2
#
#     end_node = self.graph_edges[gt[:, 0]][:, 0] - 1
#
#     path_total, path_right, d_total, d_right, traj_total, traj_right, prediction, last_pred = 0, 0, 0, 0, 0, None, None, None
#     for dim in range(gt.shape[1]):
#         cur_gt = gt[:, dim]
#         cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
#         cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
#         node_adj_edges = self.node_adj_edges[end_node]
#
#         if dim == 0:
#             last_pred = obs
#
#         last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
#         # print(f'last_pred: {last_pred.shape}\n{last_pred}')
#
#         node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges
#
#
#         cur_pred = cur_pred[self.ids, node_adj_edges]
#         cur_pred = cur_pred.max(1)[1][:, None]
#         cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
#         cur_pred[torch.where(cur_pred == self.args.num_edges)[0]] -= self.offset
#         last_pred = cur_pred
#
#         if dim != gt.shape[1] - 1:
#             end_node = self.graph_edges[cur_pred][:, 1] - 1
#
#         # print(f'pred_d: {pred_d.shape}')
#         # print(f'direction_gt: {direction_gt.shape}')
#         cur_pred_d = F.log_softmax(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], dim=1)
#         cur_pred_d = cur_pred_d.max(1)[1]
#         # print(f'cur_pred_d: {cur_pred_d.shape}')
#         cur_gt_d = direction_gt[:, dim]
#         # print(f'cur_gt_d: {cur_gt_d.shape}')
#
#         d_total += cur_gt_d.shape[0]
#         d_right += torch.sum((cur_pred_d == cur_gt_d) * 1)
#
#         path_total += cur_gt.shape[0]
#         path_right += torch.sum((cur_gt == cur_pred) * 1)
#
#         if traj_right is None:
#             traj_right = (cur_gt == cur_pred)
#         else:
#             traj_right *= (cur_gt == cur_pred)
#
#         if prediction == None:
#             prediction = torch.unsqueeze(cur_pred, dim=-1)
#         else:
#             prediction = torch.cat((prediction, torch.unsqueeze(cur_pred, dim=-1)), dim=-1)
#
#     traj_total = gt.shape[0]
#     traj_right = torch.sum(traj_right * 1)
#
#     return path_total, path_right, traj_total, traj_right, d_total, d_right, prediction, prediction