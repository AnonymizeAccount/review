import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
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

    # Knowledge Graph + Random Sample from X (CE) + sim (KL)
    def forward(self, inputs, directions, mask):
        loss = 0
        if self.args.rand:
            pred = torch.rand((self.args.batch_size, self.args.num_edges * self.args.pre_len)).to(self.args.device)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
            return pred, pred_d, loss
        else:
            link_embedding = self.link_emblayer.weight[1:, :]
            reconstructed_transition = torch.matmul(link_embedding, link_embedding.T)
            loss = self.sim_criterion(F.softmax(reconstructed_transition, dim=1),
                                      F.softmax(self.transition_matrix, dim=1) *
                                      self.edge_A + torch.eye(link_embedding.shape[0]).to(self.args.device)) * 10

            link_embs = self.link_emblayer(inputs[:, -1])
            direction_embs = self.direction_emblayer(directions[:, -1])

            sample_o, sample_d = torch.randint(high=5, size=(self.args.batch_size,)), torch.randint(low=5, high=10, size=(self.args.batch_size,))
            sample_o, sample_d = torch.squeeze(inputs[self.ids, sample_o[:, None]]), torch.squeeze(inputs[self.ids, sample_d[:, None]])

            head_entities, tail_entities = self.link_emblayer(sample_o), self.link_emblayer(sample_d)
            relations = self.direction_emblayer(self.loc_dlabels_matrix[(sample_o-1, sample_d-1)]+1)
            kg_sim_score_link = torch.matmul((head_entities+relations), self.link_emblayer.weight[1:, :].T)
            loss += self.criterion(kg_sim_score_link, sample_d-1)

            sim_score_link = torch.matmul((link_embs+direction_embs), self.link_emblayer.weight[1:, :].T)
            pred = sim_score_link.repeat(1, self.args.pre_len)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)

            return pred, pred_d, loss