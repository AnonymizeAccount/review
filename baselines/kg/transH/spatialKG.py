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
        # self.distance_ids = torch.unsqueeze(torch.LongTensor([i for i in range(self.args.pre_len)]), dim=0)\
        #     .expand(self.args.batch_size, -1).to(self.args.device)
        self.direction_ids = torch.LongTensor([i for i in range(1, self.args.direction + 1)]).to(self.args.device)
        self.pre_len_ids = torch.LongTensor([i for i in range(self.args.pre_len)]).to(self.args.device)

        node_adj_edges_list = []
        for i in range(self.node_adj_edges.shape[0]):
            node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
        # (num_nodes, max_adj_edges), values: 0 ~ num_edges - 1 with the padding_value == num_edges
        self.node_adj_edges = pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)
        self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * float('-inf'), dim=-1).to(self.args.device)
        self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
        self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)

        # entities
        self.link_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)

        # relations
        self.direction_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        self.direction_hyper = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        self.connection_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.connection_dim, padding_idx=0).to(self.args.device)
        self.connection_hyper = nn.Embedding(self.args.num_edges + 1, self.args.connection_dim, padding_idx=0).to(self.args.device)
        self.consistent_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.consistent_dim, padding_idx=0).to(self.args.device)
        self.consistent_hyper = nn.Embedding(self.args.num_edges + 1, self.args.consistent_dim, padding_idx=0).to(self.args.device)
        # self.lambda_distance = nn.Parameter(torch.tensor([1.])).to(self.args.device)
        self.lambda_distance = nn.Embedding(self.args.pre_len, 1, _weight=torch.tensor(
            [[float(i+1) / self.args.pre_len] for i in range(self.args.pre_len)]).to(self.args.device)).to(self.args.device)

        self.margin = 1
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.distfn = nn.PairwiseDistance(2)

    def normalizeEmbedding(self):
        self.direction_hyper.weight.data.copy_(self.direction_hyper.weight /
                                               torch.sqrt(torch.sum(torch.square(self.direction_hyper.weight), dim=1, keepdim=True)))
        self.connection_emblayer.weight.data.copy_(self.connection_emblayer.weight /
                                                   torch.sqrt(torch.sum(torch.square(self.connection_emblayer.weight), dim=1, keepdim=True)))
        self.consistent_hyper.weight.data.copy_(self.consistent_hyper.weight /
                                                torch.sqrt(torch.sum(torch.square(self.consistent_hyper.weight), dim=1, keepdim=True)))

    def compute_loss(self, pred, gt, pred_d, gt_d):
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.args.d * self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
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

        pred_hard = None

        last_obs = inputs[:, -1]
        # (batch_size), value: 1 ~ num_edges
        goal_directions = self.loc_dlabels_matrix[last_obs-1, goal] + 1
        loss = 0

        # (batch_size, edge_dim)
        link_embs = self.link_emblayer(inputs[:, -1])
        # (batch_size, edge_dim)
        direction_embs = self.direction_emblayer(goal_directions)
        direction_hyper = self.direction_hyper(goal_directions)

        link_embs = link_embs - direction_hyper * torch.sum(link_embs * direction_hyper, dim=1, keepdim=True)

        # print(f'link_embs: {link_embs.shape}')
        all_tail_embs = self.link_emblayer.weight[1:, :]
        # print(f'all_tail_embs: {all_tail_embs.shape}')

        # (direction, num_edges, direction_dim)
        direction_tail_embs = None
        for i, ids in enumerate(self.direction_ids):
            direction_hyper = self.direction_hyper(ids)

            cur_direction_tail_embs = all_tail_embs - direction_hyper * torch.sum(all_tail_embs * direction_hyper, dim=1, keepdim=True)
            cur_direction_tail_embs = torch.unsqueeze(cur_direction_tail_embs, dim=0)

            if i == 0:
                direction_tail_embs = cur_direction_tail_embs
            else:
                direction_tail_embs = torch.cat((direction_tail_embs, cur_direction_tail_embs), dim=0)

        if type == 'train':
            # Connection Relation
            # Positive connection
            rows, cols = torch.where(self.edge_A == 1)
            ids = torch.randperm(rows.shape[0])[:self.args.batch_size].to(self.args.device)
            heads, tails = rows[ids], cols[ids]
            connection_heads = self.link_emblayer(heads + 1)
            connection_tails = self.link_emblayer(tails + 1)
            connection_embs = self.connection_emblayer(heads + 1)
            connection_hyper = self.connection_hyper(heads + 1)
            connection_heads = connection_heads - connection_hyper * torch.sum(connection_heads * connection_hyper, dim=1, keepdim=True)
            connection_tails = connection_tails - connection_hyper * torch.sum(connection_tails * connection_hyper, dim=1, keepdim=True)
            connection_Pos_score = torch.sum(F.relu(self.distfn(connection_heads + connection_embs, connection_tails)))
            # Negative connection
            connection_tails = self.link_emblayer(tails) # wrong one, should be tails + 1, act as a negative sample here
            connection_tails = connection_tails - connection_hyper * torch.sum(connection_tails * connection_hyper, dim=1, keepdim=True)
            connection_Neg_score = torch.sum(F.relu(self.distfn(connection_heads + connection_embs, connection_tails)))
            loss += self.args.a * torch.sum(F.relu(input=connection_Pos_score - connection_Neg_score + self.margin)) / self.args.batch_size

            # # Direction Relation
            # # Positive Direction
            # sample_o, sample_d = torch.randint(high=5, size=(self.args.batch_size,)), torch.randint(low=5, high=10, size=(self.args.batch_size,))
            # sample_o, sample_d = torch.squeeze(inputs[self.ids, sample_o[:, None]]), torch.squeeze(inputs[self.ids, sample_d[:, None]])
            # sampled_directions = self.loc_dlabels_matrix[sample_o, sample_d] + 1
            # sampled_directions_embs = self.direction_emblayer(sampled_directions)
            # sampled_directions_hyper = self.direction_hyper(sampled_directions)
            # direction_heads = self.link_emblayer(sample_o + 1)
            # direction_heads = direction_heads - sampled_directions_hyper * torch.sum(direction_heads * sampled_directions_hyper, dim=1, keepdim=True)
            # direction_tail_embs = direction_tail_embs[sampled_directions-1, :, :]
            # # print(f'tail_embs: {tail_embs.shape}')
            # head_embs = torch.unsqueeze(direction_heads + sampled_directions_embs, dim=1)
            # # print(f'head_embs: {head_embs.shape}')
            # sim_score_link = torch.sum(head_embs * direction_tail_embs, dim=-1)
            # loss += self.args.b * self.criterion(sim_score_link, sample_d)

            # Direction Relation
            # Positive Direction
            rows, cols = torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device), \
                         torch.randperm(self.args.num_edges)[:self.args.batch_size].to(self.args.device)
            sampled_directions = self.loc_dlabels_matrix[rows, cols] + 1
            sampled_directions_embs = self.direction_emblayer(sampled_directions)
            sampled_directions_hyper = self.direction_hyper(sampled_directions)
            direction_heads = self.link_emblayer(rows + 1)
            direction_tails = self.link_emblayer(cols + 1)
            direction_heads = direction_heads - sampled_directions_hyper * torch.sum(direction_heads * sampled_directions_hyper, dim=1, keepdim=True)
            direction_tails = direction_tails - sampled_directions_hyper * torch.sum(direction_tails * sampled_directions_hyper, dim=1, keepdim=True)
            direction_Pos_score = torch.sum(F.relu(self.distfn(direction_heads + sampled_directions_embs, direction_tails)))
            # Negative connection
            direction_tails = self.link_emblayer(cols)  # wrong one, should be cols + 1, act as a negative sample here
            direction_tails = direction_tails - sampled_directions_hyper * torch.sum(direction_tails * sampled_directions_hyper, dim=1, keepdim=True)
            direction_Neg_score = torch.sum(F.relu(self.distfn(direction_heads + sampled_directions_embs, direction_tails)))
            loss += self.args.b * torch.sum(F.relu(input=direction_Pos_score - direction_Neg_score + self.margin)) / self.args.batch_size

            # Consistent Relation
            # Positive Direction
            sample_o, sample_d = torch.randint(high=5, size=(self.args.batch_size,)), torch.randint(low=5, high=10, size=(self.args.batch_size,))
            sample_o, sample_d = torch.squeeze(inputs[self.ids, sample_o[:, None]]), torch.squeeze(inputs[self.ids, sample_d[:, None]])
            sampled_consistent_embs = self.consistent_emblayer(sample_o + 1)
            sampled_consistent_hyper = self.consistent_hyper(sample_o + 1)
            consistent_heads = self.link_emblayer(sample_o + 1)
            consistent_tails = self.link_emblayer(sample_d + 1)
            consistent_heads = consistent_heads - sampled_consistent_hyper * \
                               torch.sum(consistent_heads * sampled_consistent_hyper, dim=1, keepdim=True)
            consistent_tails = consistent_tails - sampled_consistent_hyper * \
                               torch.sum(consistent_tails * sampled_consistent_hyper, dim=1, keepdim=True)
            consistent_Pos_score = torch.sum(F.relu(self.distfn(consistent_heads + sampled_consistent_embs, consistent_tails)))
            # Negative connection
            consistent_tails = self.link_emblayer(sample_d)  # wrong one, should be cols + 1, act as a negative sample here
            consistent_tails = consistent_tails - sampled_consistent_hyper * \
                               torch.sum(consistent_tails * sampled_consistent_hyper, dim=1, keepdim=True)
            consistent_Neg_score = torch.sum(F.relu(self.distfn(consistent_heads + sampled_consistent_embs, consistent_tails)))
            loss += self.args.c * torch.sum(F.relu(input=consistent_Pos_score - consistent_Neg_score + self.margin)) / self.args.batch_size

        # (batch_size, num_edges, direction_dim)
        direction_tail_embs = direction_tail_embs[goal_directions-1, :, :]
        # # (batch_size, 1, edge_dim)
        # head_embs = torch.unsqueeze(link_embs + direction_embs, dim=1)
        # # (batch_size, num_edges)
        # sim_score_link = torch.sum(head_embs * direction_tail_embs, dim=-1)
        # # (batch_size, num_edges * pre_len)
        # pred_hard = sim_score_link.repeat(1, self.args.pre_len)
        # print(self.lambda_distance.weight, self.lambda_distance.weight.shape)

        for i, id in enumerate(self.pre_len_ids):
            head_embs = torch.unsqueeze((link_embs + direction_embs) * self.lambda_distance(id), dim=1)
            sim_score_link = torch.sum(head_embs * direction_tail_embs, dim=-1)
            if i == 0:
                pred_hard = sim_score_link
            else:
                pred_hard = torch.cat((pred_hard, sim_score_link), dim=1)

        pred_d_rand = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        direction_correct = torch.sum((goal_directions == goal_directions) * 1)

        return pred_hard, pred_d_rand, loss, direction_correct