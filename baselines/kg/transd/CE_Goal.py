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
        print(f'Model: CE_Goal')
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
        self.direction_ids = torch.LongTensor([i for i in range(1, self.args.direction + 1)]).to(self.args.device)

        node_adj_edges_list = []
        for i in range(self.node_adj_edges.shape[0]):
            node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
        # (num_nodes, max_adj_edges), values: 0 ~ num_edges - 1 with the padding_value == num_edges
        self.node_adj_edges = pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)
        self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * float('-inf'), dim=-1).to(self.args.device)
        self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
        self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)

        self.linkEmblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
        self.linkMapEmblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
        self.directionEmblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        self.directionMapEmblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt, pred_d, gt_d):
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
        return loss


    def normalizeEmbedding(self):
        self.linkEmblayer.weight.data.copy_(torch.renorm(input=self.linkEmblayer.weight.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1))
        self.directionEmblayer.weight.data.copy_(torch.renorm(input=self.directionEmblayer.weight.detach().cpu(),
                                                              p=2,
                                                              dim=0,
                                                              maxnorm=1))

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
        linkEmbs = self.linkEmblayer(inputs[:, -1])
        linkEmbsP = self.linkMapEmblayer(inputs[:, -1])
        # (batch_size, edge_dim)
        directionEmbs = self.directionEmblayer(goal_directions)
        directionEmbsP = self.directionMapEmblayer(goal_directions)

        # print(f'linkEmbs: {linkEmbs.shape}')
        # print(f'linkEmbsP: {linkEmbsP.shape}')
        # print(f'directionEmbs: {directionEmbs.shape}')
        # print(f'directionEmbsP: {directionEmbsP.shape}')

        linkEmbsP = torch.unsqueeze(linkEmbsP, dim=1)
        directionEmbsP = torch.unsqueeze(directionEmbsP, dim=2)
        # print(f'linkEmbsP: {linkEmbsP.shape}')
        # print(f'directionEmbsP: {directionEmbsP.shape}')

        all_tailEmbs = self.linkEmblayer.weight[1:, :]
        all_tailMapEmbs = self.linkMapEmblayer.weight[1:, :]
        # print(f'all_tailEmbs: {all_tailEmbs.shape}')
        # print(f'all_tailMapEmbs: {all_tailMapEmbs.shape}')

        Mrh = torch.matmul(directionEmbsP, linkEmbsP) + torch.eye(self.args.direction_dim, self.args.edge_dim).to(linkEmbs.device)
        # print(f'Mrh: {Mrh.shape}')
        linkEmbs = torch.unsqueeze(linkEmbs, dim=2)
        head = torch.squeeze(torch.matmul(Mrh, linkEmbs), dim=2)
        # print(f'head: {head.shape}')
        all_tailMapEmbs = torch.unsqueeze(all_tailMapEmbs, dim=1)
        all_tailEmbs = torch.unsqueeze(all_tailEmbs, dim=2)
        # print(f'all_tailEmbs: {all_tailEmbs.shape}')
        # print(f'all_tailMapEmbs: {all_tailMapEmbs.shape}')

        all_tail = None
        for i, ids in enumerate(self.direction_ids):
            cur_directionEmbsP = self.directionMapEmblayer(ids)
            cur_directionEmbsP = torch.unsqueeze(torch.unsqueeze(cur_directionEmbsP, dim=0), dim=2)
            curMrt = torch.matmul(cur_directionEmbsP, all_tailMapEmbs) + torch.eye(self.args.direction_dim, self.args.edge_dim).to(linkEmbs.device)

            tail = torch.squeeze(torch.matmul(curMrt, all_tailEmbs), dim=2)
            if all_tail == None:
                all_tail = torch.unsqueeze(tail, dim=-1)
            else:
                all_tail = torch.cat((all_tail, torch.unsqueeze(tail, dim=-1)), dim=-1)

        # print(f'all_tail: {all_tail.shape}')
        tail = all_tail[..., goal_directions-1].permute(2, 0, 1)
        # print(f'tail: {tail.shape}')
        head = torch.unsqueeze(head + directionEmbs, dim=1)
        # print(f'head: {head.shape}')
        sim_score_link = torch.sum(head * tail, dim=-1)
        # print(f'sim_score_link: {sim_score_link.shape}')

        # (batch_size, num_edges * pre_len)
        pred_hard = sim_score_link.repeat(1, self.args.pre_len)
        # pred_soft = self.decoder(sim_score_link)
        # (batch_size, num_directions * pre_len)
        pred_d_rand = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)

        direction_correct = torch.sum((goal_directions == goal_directions) * 1)


        return pred_hard, pred_d_rand, loss, direction_correct