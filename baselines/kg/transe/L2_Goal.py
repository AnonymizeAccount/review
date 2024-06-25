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

        self.distfn = nn.PairwiseDistance(p=1)
        self.margin = 1

    def normalizeEmbedding(self):
        embedWeight = self.link_emblayer.weight.detach().cpu().numpy()
        embedWeight = embedWeight / np.sqrt(np.sum(np.square(embedWeight), axis=1, keepdims=True))
        self.link_emblayer.weight.data.copy_(torch.from_numpy(embedWeight))

    def compute_loss(self, pred, gt, pred_d, gt_d):
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
        return loss

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

        # rows, cols = torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device), \
        #              torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device)

        # pos_dlabels = self.loc_dlabels_matrix[rows, cols]
        # neg_dlabels = pos_dlabels + 4
        # neg_dlabels[torch.where(neg_dlabels >= 8)[0]] -= self.args.direction
        #
        # sampled_posd_embs = self.direction_emblayer(pos_dlabels + 1)
        # sampled_negd_embs = self.direction_emblayer(neg_dlabels + 1)
        #
        # direction_s_link_embs = self.link_emblayer(rows + 1)
        # direction_e_link_embs = self.link_emblayer(cols + 1)
        #
        # loss += torch.sum(
        #     F.relu(input=self.distfn(direction_s_link_embs + sampled_posd_embs, direction_e_link_embs) -
        #                  self.distfn(direction_s_link_embs + sampled_negd_embs, direction_e_link_embs) + self.margin)) \
        #     / self.args.batch_size

        # (batch_size, edge_dim)
        link_embs = self.link_emblayer(inputs[:, -1])
        # link_embs = self.combine(torch.cat((link_embs, goal_embs), dim=1))
        # (batch_size, edge_dim)
        # direction_embs = self.direction_emblayer(directions[:, -1])
        direction_embs = self.direction_emblayer(goal_directions)
        # (batch_size, num_edges), this indicate each embedded sequence's similarity to all edges
        sim_score_link = torch.matmul((link_embs+direction_embs), self.link_emblayer.weight[1:, :].T)

        sim_score_link = torch.log_softmax(sim_score_link, dim=1)

        # (batch_size, num_edges * pre_len)
        pred_hard = sim_score_link.repeat(1, self.args.pre_len)
        # (batch_size, num_directions * pre_len)
        pred_d_rand = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        direction_correct = torch.sum((goal_directions == goal_directions) * 1)

        return pred_hard, pred_d_rand, loss, direction_correct