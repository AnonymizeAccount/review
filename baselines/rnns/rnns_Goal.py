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
        # self.direction_emblayer = DirectionEmbedding(self.args).to(self.args.device)


        # self.enc = LSTMCell(input_dim=self.args.edge_dim * 2 + self.args.direction_dim, hidden_dim=self.args.hidden_dim).to(self.args.device)
        if self.args.model == 'rnn':
            self.enc = nn.RNN(self.args.edge_dim * 2 + self.args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'gru':
            self.enc = nn.GRU(self.args.edge_dim * 2 + self.args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'lstm':
            self.enc = nn.LSTM(self.args.edge_dim * 2 + self.args.direction_dim, self.args.hidden_dim, 1,
                               batch_first=True, dropout=self.args.dropout).to(self.args.device)
        else:
            raise ValueError

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.out_link = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)
        self.out_direction = nn.Linear(self.args.hidden_dim, self.args.direction * self.args.pre_len).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt, pred_d, gt_d, mask=None):

        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
            # loss += self.criterion(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], gt_d[:, dim])

        return loss

    def forward(self, inputs, directions, mask, goal, epoch=None, type=None, y=None):
        if self.args.rand:
            pred = torch.rand((self.args.batch_size, self.args.num_edges * self.args.pre_len)).to(self.args.device)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        else:

            hidden_states = torch.zeros(self.args.batch_size, self.args.hidden_dim).to(inputs.device)
            cell_states = torch.zeros(self.args.batch_size, self.args.hidden_dim).to(inputs.device)

            last_obs = inputs[:, -1]
            # (batch_size), value: 1 ~ num_edges
            self.goal_directions = self.loc_dlabels_matrix[last_obs - 1, goal] + 1

            link_embs = self.link_emblayer(inputs)
            direction_embs = self.direction_emblayer(directions)
            goal_direction_embs = self.direction_emblayer(self.goal_directions)

            goal_direction_embs = torch.unsqueeze(goal_direction_embs, dim=1).expand(self.args.batch_size, self.args.seq_len, -1)
            embs = torch.cat((link_embs, direction_embs, goal_direction_embs), dim=-1)

            l_pack = pack_padded_sequence(embs, mask, batch_first=True, enforce_sorted=False)
            if self.args.model == 'lstm':
                _, (h_n, c_n) = self.enc(l_pack)
            else:
                _, h_n = self.enc(l_pack)
            out = torch.squeeze(h_n)

            # (pred_len * edges)
            pred = self.out_link(out)

            # (pred_len * directions)
            pred_d = self.out_direction(out)

        return pred, pred_d, 0, torch.tensor([0])