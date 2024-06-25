import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.nn.modules.rnn import RNNCellBase


class LSTMCell(RNNCellBase):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__(input_dim, hidden_dim, bias=True, num_chunks=4)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_ih = nn.Parameter(torch.Tensor(4 * self.hidden_dim, self.input_dim))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * self.hidden_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * self.hidden_dim, self.hidden_dim))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(64)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_feature, hidden_states, cell_states):
        # (batch_size, rnn_dim * 4)
        gates = F.linear(input_feature.float(), self.weight_ih, self.bias_ih) + \
                F.linear(hidden_states.float(), self.weight_hh, self.bias_hh)
        # (batch_size, rnn_dim)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)
        input_gate = F.sigmoid(input_gate)
        forget_gate = F.sigmoid(forget_gate)
        cell_gate = F.tanh(cell_gate)
        output_gate = F.sigmoid(output_gate)
        output_cell_state = forget_gate * cell_states + input_gate * cell_gate
        output_hidden_state = output_gate * F.tanh(output_cell_state)

        return output_gate, output_hidden_state, output_cell_state


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix, loc_dlabels_matrix, TM, edge_A,
                 length_shortest_paths):
        super(RNN, self).__init__()
        print(f'Model: CE_Goal_Rerank')
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
        self.length_shortest_paths = torch.LongTensor(length_shortest_paths).to(self.args.device)
        self.distance_ids = torch.unsqueeze(torch.LongTensor([i for i in range(self.args.pre_len)]), dim=0)\
            .expand(self.args.batch_size, -1).to(self.args.device)

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
        # self.distance_emblayer = nn.Embedding(self.args.pre_len, self.args.distance_dim).to(self.args.device)
        self.connect_emblayer = nn.Embedding(1, self.args.connection_dim).to(self.args.device)
        # self.combine = nn.Linear(self.args.edge_dim * 2, self.args.edge_dim)
        # self.decoder = nn.Linear(self.args.num_edges, self.args.num_edges * self.args.pre_len).to(self.args.device)
        # self.direction_predictor = nn.Linear((self.args.edge_dim + self.args.direction_dim) * self.args.seq_len, self.args.direction)
        # self.direction_predictor = nn.Sequential(
        #     nn.Linear((self.args.edge_dim + self.args.direction_dim) * self.args.seq_len, self.args.hidden_dim),
            # nn.ReLU(),
            # nn.Linear(self.args.hidden_dim, self.args.direction),
        # )
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.criterion = nn.CrossEntropyLoss()

        self.rank = LSTMCell(input_dim=self.args.edge_dim + self.args.direction_dim, hidden_dim=self.args.hidden_dim).to(self.args.device)
        self.rank_predictor = nn.Sequential(
            nn.Linear(self.args.multi**self.args.pre_len * self.args.hidden_dim, self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, self.args.multi**self.args.pre_len),
        )
        self.rank_link_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
        self.rank_direction_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        self.ids_rank = torch.arange(self.args.pre_len)[:, None].to(self.args.device)

    def compute_loss(self, pred, gt, pred_d, gt_d):
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.args.d * self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
        return loss

    def rerank(self, preds_topk, gt):
        '''
        preds_topk:     shape: (batch_size, pre_len, topk),     value: 0 ~ num_edges - 1
        gt:             shape: (batch_size, pre_len),           value: 0 ~ num_edges - 1
        '''

        gt_expanded = torch.unsqueeze(gt, dim=-1).expand(-1, -1, self.args.multi**self.args.pre_len)
        # print(f'gt_expanded: {gt_expanded.shape}')
        value, idx = torch.max(torch.sum((gt_expanded.permute((0, 2, 1)) == preds_topk.permute((0, 2, 1))) * 1, dim=-1), dim=-1)
        idx_gt_in_topk = torch.where(value == 5)[0]

        # (batch_size, topk, pre_len), value: 1 ~ num_edges
        preds_topk = preds_topk.permute((0, 2, 1)) + 1
        # (batch_size, topk, pre_len, edge_dim)
        preds_topk_embs = self.rank_link_emblayer(preds_topk)
        # (batch_size, topk, pre_len), value: 1 ~ num_directions
        preds_topk_d_labels = self.direction_labels[preds_topk] + 1
        # (batch_size, topk, pre_len, direction_dim)
        preds_topk_d_embs = self.rank_direction_emblayer(preds_topk_d_labels)
        # (batch_size * topk, pre_len, edge_dim + direction_dim)
        preds_topk_embs = torch.cat((preds_topk_embs, preds_topk_d_embs), dim=-1)\
            .reshape(-1, self.args.pre_len, self.args.edge_dim + self.args.direction_dim)
        # print(f'preds_topk_embs: {preds_topk_embs.shape}')
        hidden_states = torch.zeros(preds_topk_embs.shape[0], self.args.hidden_dim).to(self.args.device)
        cell_states = torch.zeros(preds_topk_embs.shape[0], self.args.hidden_dim).to(self.args.device)
        for i in range(preds_topk_embs.shape[1]):
            input_i = preds_topk_embs[:, i, :]
            _, hidden_states, cell_states = self.rank(input_i, hidden_states, cell_states)
        out = self.dropout(hidden_states)
        # print(f'out: {out.shape}')
        logits = out.reshape(self.args.batch_size, -1)
        # print(f'logits: {logits.shape}')
        rank_pred = self.rank_predictor(logits)
        # print(f'rank_pred: {rank_pred.shape}')

        rank_pred = torch.log_softmax(rank_pred, dim=1)
        rank_pred_sorted, idx_sorted = torch.sort(rank_pred, dim=1, descending=True)
        # print(f'rank_pred_sorted: {rank_pred_sorted.shape}\n{rank_pred_sorted}')
        # print(f'idx_sorted: {idx_sorted.shape}\n{idx_sorted}')
        # print(f'idx:\n{rank_pred[self.ids, idx_sorted]}')

        # print(f'preds_topk: {preds_topk.shape}\n{preds_topk}')
        preds_topk = preds_topk[self.ids, idx_sorted] - 1
        # print(f'preds_topk: {preds_topk.shape}\n{preds_topk}')

        rank_pred, gt, idx_gt_k_in_topk = rank_pred[idx_gt_in_topk, :], gt[idx_gt_in_topk], idx[idx_gt_in_topk]
        # print(f'rank_pred: {rank_pred.shape}')
        # print(f'gt: {gt.shape}')
        # print(f'idx_gt_k_in_topk: {idx_gt_k_in_topk.shape}')

        loss_ranking = self.args.e * self.criterion(rank_pred, idx_gt_k_in_topk)

        return loss_ranking, preds_topk.permute((0, 2, 1))

    # Knowledge Graph (CE)
    def forward(self, inputs, directions, mask, goal, epoch=None, type=None, y=None):
        '''
        inputs:      edge sequences,         shape: (batch_size, seq_len),    value: 1 ~ num_edges
        directions:  direction sequences,    shape: (batch_size, seq_len),    value: 1 ~ num_directions
        mask:        list of valid sequence length for each sequence in the batch, len(mask) == batch_size
        goal:        final edge,             shape: (batch_size),             value: 0 ~ num_edges - 1
        y:           Ground Truth,           shape: (batch_size, pre_len),    value: 0 ~ num_edges - 1
        '''

        last_obs = inputs[:, -1]
        # (batch_size), value: 1 ~ num_edges
        goal_directions = self.loc_dlabels_matrix[last_obs-1, goal] + 1
        loss = 0

        # (batch_size, edge_dim)
        link_embs = self.link_emblayer(inputs[:, -1])
        # (batch_size, edge_dim)
        direction_embs = self.direction_emblayer(goal_directions)

        if type == 'train':
            # connection
            rows, cols = torch.where(self.edge_A == 1)
            ids = torch.randperm(rows.shape[0])[:self.args.batch_size].to(self.args.device)
            rows, cols = rows[ids], cols[ids]
            connection_s_link_embs = self.link_emblayer(rows + 1)
            logits = torch.matmul((connection_s_link_embs + self.connect_emblayer.weight), self.link_emblayer.weight[1:, :].T)
            # print(f'logits: {logits.shape}')
            loss += self.args.a * self.criterion(torch.log_softmax(logits, dim=1), cols)

            # direction
            # rows, cols = torch.where(self.length_shortest_paths <= self.args.pre_len)
            # rows, cols = rows[:self.args.batch_size], cols[:self.args.batch_size]
            rows, cols = torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device), \
                         torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device)
            sampled_directions_embs = self.direction_emblayer(self.loc_dlabels_matrix[rows, cols] + 1)
            # sampled_distance = torch.unsqueeze(self.length_shortest_paths[rows, cols], dim=-1)
            # print(f'sampled_directions_embs: {sampled_directions_embs.shape}')
            # print(f'sampled_distance: {sampled_distance.shape}')
            direction_s_link_embs = self.link_emblayer(rows+1)
            logits = torch.matmul((direction_s_link_embs + sampled_directions_embs), self.link_emblayer.weight[1:, :].T)
            # print((direction_s_link_embs + sampled_directions_embs).shape, self.link_emblayer.weight[1:, :].T.shape, logits.shape)
            # assert 1==2
            loss += self.args.b * self.criterion(torch.log_softmax(logits, dim=1), cols)


        pred_hard = None
        for i in range(self.args.pre_len):
            # distance_embs = self.distance_emblayer(self.distance_ids[:, i])

            # if type == 'train':
                # rows, cols = torch.where(self.length_shortest_paths == i+1)
                # ids_i = torch.randperm(rows.shape[0])[:self.args.batch_size].to(self.args.device)
                # rows, cols = rows[ids_i], cols[ids_i]
                # dist_s_link_embs = self.link_emblayer(rows+1)
                # logits = torch.matmul((dist_s_link_embs + distance_embs), self.link_emblayer.weight[1:, :].T)
                # # distance
                # loss += self.args.c * self.criterion(torch.log_softmax(logits, dim=1), cols) / self.args.pre_len
                # logits = torch.matmul((link_embs + distance_embs), self.link_emblayer.weight[1:, :].T)
                # loss += self.args.c * self.criterion(logits, y[:, i]) / self.args.pre_len

            if i == 0:
                # pred_hard = torch.matmul((link_embs + direction_embs * distance_embs), self.link_emblayer.weight[1:, :].T)
                pred_hard = torch.matmul((link_embs + direction_embs * (i+1)) + self.connect_emblayer.weight, self.link_emblayer.weight[1:, :].T)
            else:
                # pred_hard = torch.cat((pred_hard, torch.matmul((link_embs + direction_embs * distance_embs),
                #                                                self.link_emblayer.weight[1:, :].T)), dim=-1)
                pred_hard = torch.cat((pred_hard, torch.matmul((link_embs + direction_embs * (i+1)) + self.connect_emblayer.weight,
                                                               self.link_emblayer.weight[1:, :].T)), dim=-1)

        pred_d_rand = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)

        direction_correct = torch.sum((goal_directions == goal_directions) * 1)

        return pred_hard, pred_d_rand, loss, direction_correct