import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.nn.modules.rnn import RNNCellBase


class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, args):
        super(MLP, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.fc1 = nn.Linear(self.input_dim, self.args.hidden_dim)
        self.fc2 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.fc3 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.out = nn.Linear(self.args.hidden_dim, self.out_dim).to(self.args.device)

    def forward(self, input_feature):
        x = F.relu(self.fc1(input_feature))
        x = F.relu(self.fc2(x))
        output_hidden_state = self.fc3(x)

        return self.out(output_hidden_state)


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix, loc_dlabels_matrix, TM, edge_A,
                 length_shortest_paths):
        super(RNN, self).__init__()
        print(f'Model: spatialKG_rerank_1_best_0514_SpKG_EGoalD')
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
        self.connection_emblayer = nn.Embedding(1, self.args.connection_dim).to(self.args.device)
        self.connection_hyper = nn.Embedding(1, self.args.connection_dim).to(self.args.device)
        self.consistent_emblayer = nn.Embedding(1, self.args.consistent_dim).to(self.args.device)
        self.consistent_hyper = nn.Embedding(1, self.args.consistent_dim).to(self.args.device)

        self.lambda_distance = nn.Embedding(self.args.pre_len, 1, _weight=torch.tensor(
            [[float(i+1) / self.args.pre_len] for i in range(self.args.pre_len)]).to(self.args.device)).to(self.args.device)

        self.margin = self.args.margin
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.distfn = nn.PairwiseDistance(self.args.L)

        self.link_ED_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
        self.direction_ED_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
        # self.d_predictor = MLP(input_dim=self.args.seq_len * 2 + self.args.direction_dim, out_dim=self.args.direction, args=self.args).to(args.device)
        # self.d_predictor = MLP(input_dim=self.args.seq_len, out_dim=self.args.direction, args=self.args).to(args.device)
        self.d_predictor = MLP(input_dim=self.args.obs_len * (self.args.edge_dim + self.args.direction_dim),
                               out_dim=self.args.direction, args=self.args).to(args.device)
        self.mlp = MLP(input_dim=(self.args.edge_dim + self.args.direction_dim * 2 + self.args.hidden_dim), args=self.args,
                       out_dim=self.args.num_edges * self.args.pre_len).to(args.device)
        self.topk_compress = nn.Linear(self.args.multi ** self.args.pre_len * self.args.hidden_dim, self.args.hidden_dim).to(args.device)
        self.rank = MLP(input_dim=self.args.multi ** self.args.pre_len * self.args.pre_len * (self.args.edge_dim + self.args.direction_dim),
                        out_dim=self.args.hidden_dim * self.args.multi ** self.args.pre_len, args=self.args).to(self.args.device)

        self.rank_predictor = nn.Sequential(
            nn.Linear((self.args.multi ** self.args.pre_len) *
                      (self.args.hidden_dim + self.args.connection_dim + self.args.consistent_dim), self.args.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.args.hidden_dim, (self.args.multi ** self.args.pre_len)),
        )
        self.ids_rank = torch.arange(self.args.pre_len)[:, None].to(self.args.device)

    def normalizeEmbedding(self):
        self.direction_hyper.weight.data.copy_(self.direction_hyper.weight /
                                               torch.sqrt(torch.sum(torch.square(self.direction_hyper.weight), dim=1, keepdim=True)))
        self.connection_emblayer.weight.data.copy_(self.connection_emblayer.weight /
                                                   torch.sqrt(torch.sum(torch.square(self.connection_emblayer.weight), dim=1, keepdim=True)))
        self.consistent_hyper.weight.data.copy_(self.consistent_hyper.weight /
                                                torch.sqrt(torch.sum(torch.square(self.consistent_hyper.weight), dim=1, keepdim=True)))

    def compute_loss(self, pred, gt, pred_d=None, gt_d=None):
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.args.g * self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
        return loss

    def rerank(self, preds_topk, gt, type=None):
        '''
        preds_topk:     shape: (batch_size, pre_len, topk),     value: 0 ~ num_edges - 1
        gt:             shape: (batch_size, pre_len),           value: 0 ~ num_edges - 1
        '''
        loss_ranking = 0
        # (batch_size, pre_len, topk)
        gt_expanded = torch.unsqueeze(gt, dim=-1).expand(-1, -1, preds_topk.shape[-1])
        # value: shape (batch_size), 0 ~ pre_len
        # idx: shape (batch_size), 0 ~ topk-1
        value, idx = torch.max(torch.sum((gt_expanded.permute((0, 2, 1)) == preds_topk.permute((0, 2, 1))) * 1, dim=-1), dim=-1)
        # (num), num is the with full correct id, 0 ~ batch_size-1
        idx_gt_in_topk = torch.where(value == self.args.pre_len)[0]
        # (batch_size, topk, pre_len), value: 1 ~ num_edges
        preds_topk = preds_topk.permute((0, 2, 1)) + 1

        # print(f'preds_topk | max: {torch.max(preds_topk)} | min: {torch.min(preds_topk)}')

        # (batch_size, topk, pre_len, edge_dim)
        # preds_topk_embs = self.rank_link_emblayer(preds_topk)
        preds_topk_embs = self.link_emblayer(preds_topk).detach()
        # (1, connection_dim)
        connection_embs = self.connection_emblayer.weight
        # (1, connection_dim)
        connection_hyper = self.connection_hyper.weight
        # (batch_size, topk, pre_len, edge_dim)
        connection_preds_topk_embs = preds_topk_embs - connection_hyper * torch.sum(preds_topk_embs * connection_hyper, dim=1, keepdim=True)
        # (batch_size, topk, edge_dim)
        connection_margin = torch.mean(connection_preds_topk_embs[:, :, :-1, :] + connection_embs - connection_preds_topk_embs[:, :, 1:, :], dim=-2)
        # (1, consistent_dim)
        consistent_embs = self.consistent_emblayer.weight
        # (1, consistent_dim)
        consistent_hyper = self.consistent_hyper.weight
        # (batch_size, topk, pre_len, edge_dim)
        consistent_preds_topk_embs = preds_topk_embs - consistent_hyper * torch.sum(preds_topk_embs * consistent_hyper, dim=1, keepdim=True)
        # (batch_size, topk, edge_dim)
        consistent_margin = torch.mean(consistent_preds_topk_embs[:, :, :-1, :] + consistent_embs - consistent_preds_topk_embs[:, :, 1:, :], dim=-2)

        # (batch_size, topk, pre_len), value: 1 ~ num_directions
        preds_topk_d_labels = self.direction_labels[preds_topk-1] + 1

        # (batch_size, topk, pre_len, direction_dim)
        # preds_topk_d_embs = self.rank_direction_emblayer(preds_topk_d_labels)
        preds_topk_d_embs = self.direction_emblayer(preds_topk_d_labels).detach()
        # (batch_size * topk, pre_len, edge_dim + direction_dim)

        # preds_topk_embs = torch.cat((preds_topk_embs, preds_topk_d_embs), dim=-1)\
        #     .reshape(-1, self.args.pre_len, self.args.edge_dim + self.args.direction_dim)
        preds_topk_embs_return = torch.cat((preds_topk_embs, preds_topk_d_embs), dim=-1)
        # preds_topk_embs = preds_topk_embs_return.reshape(-1, self.args.pre_len, self.args.edge_dim + self.args.direction_dim)
        preds_topk_embs = preds_topk_embs_return.reshape(self.args.batch_size, -1)
        # print(f'preds_topk_embs: {preds_topk_embs.shape}')

        # # (batch_size * topk, hidden_dim)
        # hidden_states = torch.zeros(preds_topk_embs.shape[0], self.args.hidden_dim).to(self.args.device)
        # # (batch_size * topk, hidden_dim)
        # cell_states = torch.zeros(preds_topk_embs.shape[0], self.args.hidden_dim).to(self.args.device)
        connection_margin = connection_margin.reshape(self.args.batch_size, -1)
        consistent_margin = consistent_margin.reshape(self.args.batch_size, -1)
        # for i in range(preds_topk_embs.shape[1]):
        #     input_i = preds_topk_embs[:, i, :]
        #     _, hidden_states, cell_states = self.rank(input_i, hidden_states, cell_states)
        # # (batch_size * topk, hidden_dim)
        # out = self.dropout(hidden_states)
        # # (batch_size, hidden_dim * topk)
        # logits = out.reshape(self.args.batch_size, -1)

        logits = self.rank(preds_topk_embs)

        top1_logits = self.topk_compress(logits.detach())
        self.embs = torch.cat((self.embs, top1_logits), dim=-1)
        top1_pred = self.mlp(self.embs)
        loss_ranking += self.args.f * self.compute_loss(top1_pred, gt)

        # (batch_size, hidden_dim * topk * 3)
        logits = torch.cat((logits, consistent_margin, connection_margin), dim=-1)
        # (batch_size, topk)
        rank_pred = self.rank_predictor(logits)

        self.rank_prob = torch.softmax(rank_pred.detach(), dim=1)

        rank_pred = torch.log_softmax(rank_pred, dim=1)

        # rank_pred_sorted, idx_sorted: (batch_size, topk)
        rank_pred_sorted, idx_sorted = torch.sort(rank_pred, dim=1, descending=True)
        # (batch_size, topk, pre_len)
        preds_topk = preds_topk[self.ids, idx_sorted] - 1
        # rank_pred:            (num, topk)
        # gt:                   (num, pre_len)
        # idx_gt_k_in_topk:     (num)
        rank_pred, gt, idx_gt_k_in_topk = rank_pred[idx_gt_in_topk, :], gt[idx_gt_in_topk], idx[idx_gt_in_topk]
        loss_ranking += self.args.e * self.criterion(rank_pred, idx_gt_k_in_topk)

        return loss_ranking, preds_topk.permute((0, 2, 1)), top1_pred

    # def fine_rank(self, embs, topk_embs):
    #     # print(f'embs: {embs.shape}')
    #     # print(f'topk_embs: {topk_embs.shape}')
    #
    #     topk_embs = self.topk_compress(topk_embs)
    #     pred_fine = self.out_link(self.mlp(torch.cat((embs, topk_embs), -1)))
    #
    #     # print(f'pred_fine: {pred_fine.shape}')
    #     # assert 1==2
    #
    #     return pred_fine

    # Knowledge Graph (CE)
    def forward(self, inputs, directions, mask, goal, epoch=None, type=None, y=None):
        '''
        inputs:      edge sequences,         shape: (batch_size, seq_len),    value: 1 ~ num_edges
        directions:  direction sequences,    shape: (batch_size, seq_len),    value: 1 ~ num_directions
        mask:        list of valid sequence length for each sequence in the batch, len(mask) == batch_size
        goal:        final edge,             shape: (batch_size),             value: 0 ~ num_edges - 1
        y:           Ground Truth,           shape: (batch_size, pre_len),    value: 0 ~ num_edges - 1
        '''

        pred_hard, pred_hard_detached, loss = None, None, 0
        # pred_goald_logits = self.d_predictor(torch.cat((inputs, directions), dim=1).float())
        ED_direction_embs = self.direction_ED_emblayer(directions[:, (self.args.seq_len - self.args.obs_len):]).reshape(directions.shape[0], -1)
        ED_link_embs = self.link_ED_emblayer(inputs[:, (self.args.seq_len - self.args.obs_len):]).reshape(inputs.shape[0], -1)
        pred_goald_logits = self.d_predictor(torch.cat((ED_link_embs, ED_direction_embs), dim=-1))

        # input_d_embs = self.direction_ED_emblayer(self.loc_dlabels_matrix[inputs[:, 0]-1, inputs[:, -1]-1] + 1)
        # pred_goald_logits = self.d_predictor(torch.cat((inputs, directions, input_d_embs), dim=1).float())


        last_obs = inputs[:, -1]
        # (batch_size), value: 1 ~ num_edges
        self.goal_directions = self.loc_dlabels_matrix[last_obs-1, goal] + 1
        goal_directions_label = self.goal_directions - 1
        # print(f'goal_directions: {self.goal_directions.shape}')

        if type == 'train':
            loss += self.args.h * self.criterion(pred_goald_logits, goal_directions_label)

        pred_goald_prob = F.softmax(pred_goald_logits, dim=1)
        _, predicted = torch.max(pred_goald_prob, 1)  # Get the predicted labels
        correct = (predicted == goal_directions_label).sum()  # Count the number of correct predictions
        # total = goal_directions_label.size(0)  # Get the total number of samples


        # (batch_size, edge_dim)
        link_embs = self.link_emblayer(inputs[:, -1])

        self.goal_directions = predicted + 1

        # (batch_size, edge_dim)
        direction_embs = self.direction_emblayer(self.goal_directions)
        direction_hyper = self.direction_hyper(self.goal_directions)
        link_embs = link_embs - direction_hyper * torch.sum(link_embs * direction_hyper, dim=1, keepdim=True)

        d_embs = self.direction_emblayer(directions[:, -1])

        self.embs = torch.cat((link_embs, direction_embs, d_embs), dim=-1)

        all_tail_embs = self.link_emblayer.weight[1:, :]

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

        # (batch_size, num_edges, direction_dim)
        direction_tail_embs = direction_tail_embs[self.goal_directions-1, :, :]

        if type == 'train':
            # Connection Relation
            # Positive connection
            rows, cols = torch.where(self.edge_A == 1)
            ids = torch.randperm(rows.shape[0])[:self.args.batch_size].to(self.args.device)
            heads, tails = rows[ids], cols[ids]
            connection_heads = self.link_emblayer(heads + 1)
            connection_tails = self.link_emblayer(tails + 1)
            # connection_embs = self.connection_emblayer(heads + 1)
            # connection_hyper = self.connection_hyper(heads + 1)
            connection_embs = self.connection_emblayer.weight
            connection_hyper = self.connection_hyper.weight
            connection_heads = connection_heads - connection_hyper * torch.sum(connection_heads * connection_hyper, dim=1, keepdim=True)
            connection_tails = connection_tails - connection_hyper * torch.sum(connection_tails * connection_hyper, dim=1, keepdim=True)
            connection_Pos_score = torch.sum(F.relu(self.distfn(connection_heads + connection_embs, connection_tails)))
            # Negative connection
            connection_tails = self.link_emblayer(tails) # wrong one, should be tails + 1, act as a negative sample here
            connection_tails = connection_tails - connection_hyper * torch.sum(connection_tails * connection_hyper, dim=1, keepdim=True)
            connection_Neg_score = torch.sum(F.relu(self.distfn(connection_heads + connection_embs, connection_tails)))
            loss += self.args.a * torch.sum(F.relu(input=connection_Pos_score - connection_Neg_score + self.margin)) / self.args.batch_size

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

            # distance
            for i, id in enumerate(self.pre_len_ids):
                # detached for training the distance relation
                head_embs_detached = torch.unsqueeze((link_embs.detach() + direction_embs.detach()) * self.lambda_distance(id), dim=1)
                sim_score_link_detached = torch.sum(head_embs_detached * direction_tail_embs.detach(), dim=-1)
                if i == 0:
                    pred_hard_detached = sim_score_link_detached
                else:
                    pred_hard_detached = torch.cat((pred_hard_detached, sim_score_link_detached), dim=1)

            for dim in range(y.shape[1]):
                loss += self.args.d * self.criterion(pred_hard_detached[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], y[:, dim])

            # Consistent Relation
            # Positive Direction
            sample_o, sample_d = torch.randint(high=int(self.args.seq_len / 2), size=(self.args.batch_size,)), \
                                 torch.randint(low=int(self.args.seq_len / 2), high=self.args.seq_len, size=(self.args.batch_size,))
            sample_o, sample_d = torch.squeeze(inputs[self.ids, sample_o[:, None]]), torch.squeeze(inputs[self.ids, sample_d[:, None]])

            # print(f'sample_o: max: {torch.max(sample_o)} | min: {torch.min(sample_o)}')
            # print(f'sample_d: max: {torch.max(sample_d)} | min: {torch.min(sample_d)}')

            # sampled_consistent_embs = self.consistent_emblayer(sample_o + 1)
            # sampled_consistent_hyper = self.consistent_hyper(sample_o + 1)
            sampled_consistent_embs = self.consistent_emblayer.weight
            sampled_consistent_hyper = self.consistent_hyper.weight
            # 0405 change sample_o+1, sample_d+1 to sample_o, sample_d
            consistent_heads = self.link_emblayer(sample_o)
            consistent_tails = self.link_emblayer(sample_d)
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

        for i, id in enumerate(self.pre_len_ids):
            head_embs = torch.unsqueeze((link_embs + direction_embs) * self.lambda_distance(id), dim=1)
            # head_embs = torch.unsqueeze(link_embs + direction_embs * self.lambda_distance(id), dim=1)


            # sim_score_link = torch.sum((head_embs + torch.unsqueeze(goal_embs, dim=1)) * direction_tail_embs, dim=-1)
            sim_score_link = torch.sum(head_embs * direction_tail_embs, dim=-1)

            if i == 0:
                pred_hard = sim_score_link
            else:
                pred_hard = torch.cat((pred_hard, sim_score_link), dim=1)

        pred_d_rand = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        # direction_correct = torch.sum((self.goal_directions == self.goal_directions) * 1)


        return pred_hard, pred_d_rand, loss, correct