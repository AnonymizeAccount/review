import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.nn.modules.rnn import RNNCellBase


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix, loc_dlabels_matrix, TM, edge_A,
                 length_shortest_paths):
        super(RNN, self).__init__()
        print(f'Model: KG2E_CE_Goal')
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
        self.pre_len_ids = torch.arange(1, self.args.pre_len + 1).to(self.args.device)

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
        # self.distance_emblayer = nn.Embedding(self.args.pre_len + 1, self.args.distance_dim, padding_idx=0).to(self.args.device)

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.criterion = nn.CrossEntropyLoss()
        # self.distfn = nn.PairwiseDistance(p=1)
        self.margin = 1

    # def normalizeEmbedding(self):
    #     embedWeight = self.link_emblayer.weight.detach().cpu().numpy()
    #     embedWeight = embedWeight / np.sqrt(np.sum(np.square(embedWeight), axis=1, keepdims=True))
    #     self.link_emblayer.weight.data.copy_(torch.from_numpy(embedWeight))

    def normalizeEmbedding(self, embedding):
        return embedding / torch.sqrt(torch.sum(torch.square(embedding), dim=1, keepdim=True))

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
        loss, pred_hard = 0, None

        # rows, cols = torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device), \
        #              torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device)
        #
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
        # loss += torch.sum(input=self.distfn(direction_s_link_embs - direction_e_link_embs, sampled_posd_embs) -
        #                         self.distfn(direction_s_link_embs - direction_e_link_embs, sampled_negd_embs) + self.margin) \
        #     / self.args.batch_size

        # print(f'loss: {loss}')
        # loss += torch.sum(
        #     F.relu(input=self.distfn(direction_s_link_embs + sampled_posd_embs, direction_e_link_embs) -
        #                  self.distfn(direction_s_link_embs + sampled_negd_embs, direction_e_link_embs) + self.margin)) \
        #     / self.args.batch_size

        # (batch_size, edge_dim)
        link_embs = self.link_emblayer(inputs[:, -1])
        # (batch_size, edge_dim)
        direction_embs = self.direction_emblayer(goal_directions)

        for i in range(self.args.pre_len):
            # (batch_size, num_edges), this indicate each embedded sequence's similarity to all edges
            # sim_score_link = torch.matmul((link_embs + self.normalizeEmbedding(direction_embs) * self.distance_emblayer(self.pre_len_ids[i])),
                                          # self.link_emblayer.weight[1:, :].T)
            sim_score_link = torch.matmul(link_embs + direction_embs, self.link_emblayer.weight[1:, :].T)
            if pred_hard == None:
                pred_hard = sim_score_link
            else:
                pred_hard = torch.cat((pred_hard, sim_score_link), dim=1)

        # (batch_size, num_directions * pre_len)
        pred_d_rand = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        direction_correct = torch.sum((goal_directions == goal_directions) * 1)

        return pred_hard, pred_d_rand, loss, direction_correct