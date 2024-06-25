import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np


class NaiveEmbedding(nn.Module):
    def __init__(self, args):
        super(NaiveEmbedding, self).__init__()

        self.args = args
        self.emb_edges = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0)

    def forward(self, inputs):
        embeddings = self.emb_edges(inputs)

        return embeddings


class DirectionEmbedding(nn.Module):
    def __init__(self, args):
        super(DirectionEmbedding, self).__init__()

        self.args = args
        self.emb_directions = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0)

    def forward(self, inputs):
        embeddings = self.emb_directions(inputs)

        return embeddings


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, nodes_links_directions, ):
        super(RNN, self).__init__()
        self.args = args
        self.node_adj_edges = node_adj_edges
        self.nodes_links_directions = nodes_links_directions
        self.graph = graph
        self.graph_edges = torch.LongTensor(np.array(self.graph.edges)).to(self.args.device)

        node_adj_edges_list = []
        for i in range(self.node_adj_edges.shape[0]):
            node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
        self.node_adj_edges = \
            pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)

        nodes_links_directions_list = []
        for i in range(self.nodes_links_directions.shape[0]):
            nodes_links_directions_list.append(torch.from_numpy(self.nodes_links_directions[i]))
        self.nodes_links_directions = \
            pad_sequence(nodes_links_directions_list, batch_first=True, padding_value=self.args.direction).to(self.args.device)

        self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * -100, dim=-1).to(self.args.device)
        self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
        # offset used for continue training when predicted trajectory has not other route to select
        self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)

        self.link_emblayer = NaiveEmbedding(self.args).to(self.args.device)
        self.direction_emblayer = DirectionEmbedding(self.args).to(self.args.device)

        if self.args.model in ['rnn']:
            self.enc = nn.RNN(self.args.edge_dim + args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
            # self.dec = nn.RNN(self.args.hidden_dim, self.args.hidden_dim, 1,
            #                   batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model in ['gru']:
            self.enc = nn.GRU(self.args.edge_dim + args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
            # self.dec = nn.GRU(self.args.hidden_dim, self.args.hidden_dim, 1,
            #                   batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model in ['lstm', 'nettraj', ]:
            self.enc = nn.LSTM(self.args.edge_dim + args.direction_dim, self.args.hidden_dim, 1,
                               batch_first=True, dropout=self.args.dropout).to(self.args.device)
            # self.dec = nn.LSTM(self.args.hidden_dim, self.args.hidden_dim, 1,
            #                    batch_first=True, dropout=self.args.dropout).to(self.args.device)
        else:
            raise ValueError

        # self.linear = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)
        self.linear = nn.Linear(self.args.hidden_dim, self.args.direction * self.args.pre_len).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_mask(self, node_adj_edges):
        new_node_adj_edges = np.ones((node_adj_edges.shape[0], self.args.num_edges))
        for i in range(node_adj_edges.shape[0]):
            new_node_adj_edges[i][node_adj_edges[i]] = 0

        return torch.LongTensor(new_node_adj_edges).to(self.args.device)

    def compute_loss(self, pred, gt):
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.direction: (dim + 1) * self.args.direction], gt[:, dim])

        return loss

    def new_compute_acc(self, pred, gt, gt_direction):

        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1
        path_total, path_right = 0, 0
        traj_total, traj_right = 0, None
        prediction = None

        for dim in range(gt.shape[1]):
            cur_gt = gt[:, dim]
            cur_pred = F.log_softmax(pred[:, dim * self.args.direction: (dim + 1) * self.args.direction], dim=1)
            print(f'cur_pred: {cur_pred.shape}')
            print(f'cur_gt: {cur_gt.shape}')
            print(f'gt: {gt.shape}')
            print(f'gt_direction: {gt_direction.shape}')
            cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)

            print(f'cur_pred: {cur_pred.shape}\n{cur_pred}')
            node_adj_edges = self.node_adj_edges[end_node]
            nodes_links_directions = self.nodes_links_directions[end_node]
            print(f'node_adj_edges: {node_adj_edges.shape}\n{node_adj_edges}')
            print(f'nodes_links_directions: {nodes_links_directions.shape}\n{nodes_links_directions}')
            cur_pred = cur_pred[self.ids, nodes_links_directions]
            print(f'cur_pred: {cur_pred.shape}\n{cur_pred}')
            print(f'max: {cur_pred.max(1)}')
            cur_pred = cur_pred.max(1)[1][:, None]

            print(f'cur_pred: {cur_pred.shape}\n{cur_pred}')
            assert 1==2

            # cur_pred = cur_pred[self.ids, node_adj_edges]
            # cur_pred = cur_pred.max(1)[1][:, None]
            cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
            cur_pred[torch.where(cur_pred == self.args.num_edges)[0]] -= self.offset

            if dim != gt.shape[1] - 1:
                end_node = self.graph_edges[cur_pred][:, 1] - 1

            path_total += cur_gt.shape[0]
            path_right += torch.sum((cur_gt == cur_pred) * 1)
            if traj_right is None:
                traj_right = (cur_gt == cur_pred)
            else:
                traj_right *= (cur_gt == cur_pred)

            if prediction == None:
                prediction = torch.unsqueeze(cur_pred, dim=-1)
            else:
                prediction = torch.cat((prediction, torch.unsqueeze(cur_pred, dim=-1)), dim=-1)

        traj_total = gt.shape[0]
        traj_right = torch.sum(traj_right * 1)

        return path_total, path_right, traj_total, traj_right, prediction

    def forward(self, inputs, directions, mask):
        link_embs = self.link_emblayer(inputs)
        direction_embs = self.direction_emblayer(directions)
        embs = torch.cat((direction_embs, link_embs), dim=-1)
        l_pack = pack_padded_sequence(embs, mask, batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.enc(l_pack)
        out = torch.squeeze(h_n)
        preds = self.linear(out)

        return preds