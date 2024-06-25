import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


class NaiveEmbedding(nn.Module):
    def __init__(self, args):
        super(NaiveEmbedding, self).__init__()

        self.num_nodes = args.num_nodes
        self.num_edges = args.num_edges
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.mask_node = args.mask_node
        self.mask_edge = args.mask_edge
        self.emb_edges = nn.Embedding(self.num_edges+1, self.edge_dim, padding_idx=0)

    def forward(self, inputs):

        embeddings = self.emb_edges(inputs)

        # assert nodes_embeddings is not None and edges_embeddings is not None
        return embeddings


class RnnFactory():
    ''' Creates the desired RNN unit. '''

    def __init__(self, args):
        self.args = args
        self.rnn_type = args.model

    def create(self, input_dim, hidden_dim, num_layer, dropout=0):
        if self.rnn_type == 'rnn':
            return nn.RNN(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).to(self.args.device)
        if self.rnn_type == 'gru':
            return nn.GRU(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).to(self.args.device)
        if self.rnn_type == 'lstm':
            return nn.LSTM(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).to(self.args.device)


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, nodes_links_directions):
        super(RNN, self).__init__()
        self.args = args
        self.graph = graph
        self.embeddinglayer = NaiveEmbedding(args)
        self.model = RnnFactory(args).create(input_dim=1, hidden_dim=args.hidden_dim, num_layer=1, dropout=0)
        self.linear = nn.Linear(self.args.hidden_dim, self.args.num_edges * args.pre_len)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt):

        # print(f'pred: {pred.shape}\n{pred}')
        # print(f'gt: {gt.shape}\n{gt}')
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim+1) * self.args.num_edges], gt[:, dim])

        return loss

    def new_compute_acc(self, pred, gt):

        # print(f'pred: {pred.shape}\n{pred}')
        # print(f'gt: {gt.shape}\n{gt}')
        # assert 1==2

        path_total, path_right = 0, 0
        traj_total, traj_right = 0, None
        prediction = None
        for dim in range(gt.shape[1]):
            cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim+1) * self.args.num_edges], dim=1)
            cur_pred = cur_pred.max(1)[1]
            cur_gt = gt[:, dim]
            
            if prediction == None:
                prediction = torch.unsqueeze(cur_pred, dim=-1)
            else:
                prediction = torch.cat((prediction, torch.unsqueeze(cur_pred, dim=-1)), dim=-1)

            # print(f'cur_pred: {cur_pred.shape}\n{cur_pred}')
            # print(f'cur_gt: {cur_gt.shape}\n{cur_gt}')
            # print(cur_gt == cur_pred)
            # print(f'right: {torch.sum((cur_gt == cur_pred) * 1)}')
            # print(f'prediction: {prediction.shape}')
            # assert 1==2

            path_total += cur_gt.shape[0]
            path_right += torch.sum((cur_gt == cur_pred) * 1)
            if traj_right is None:
                traj_right = (cur_gt == cur_pred)
            else:
                traj_right *= (cur_gt == cur_pred)

        traj_total = gt.shape[0]
        traj_right = torch.sum(traj_right * 1)

        return path_total, path_right, traj_total, traj_right, prediction

    def forward(self, inputs, mask):

        # print(f'inputs: {inputs.shape}')
        emb_inputs = self.embeddinglayer(inputs)
        # print(f'emb_inputs: {emb_inputs.shape}')
        l_pack = pack_padded_sequence(emb_inputs, mask, batch_first=True, enforce_sorted=False)
        _, (out, _) = self.model(l_pack)
        # print(f'out: {out.shape}')
        out = torch.squeeze(out)
        # print(f'out: {out.shape}')
        pred = self.linear(out)

        return pred
