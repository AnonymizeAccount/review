import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.nn.modules.rnn import RNNCellBase


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_ih = nn.Parameter(torch.Tensor(3 * self.hidden_dim, self.input_dim))
        self.bias_ih = nn.Parameter(torch.Tensor(3 * self.hidden_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * self.hidden_dim, self.hidden_dim))
        self.bias_hh = nn.Parameter(torch.Tensor(3 * self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_feature, hidden_states):
        gates = F.linear(input_feature, self.weight_ih, self.bias_ih) + \
                F.linear(hidden_states, self.weight_hh, self.bias_hh)

        reset_gate, update_gate, candidate_gate = gates.chunk(3, 1)

        reset_gate = F.sigmoid(reset_gate)
        update_gate = F.sigmoid(update_gate)
        candidate_gate = F.tanh(candidate_gate * reset_gate)

        output_hidden_state = (1 - update_gate) * hidden_states + update_gate * candidate_gate

        return output_hidden_state


class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.weight_ih = nn.Parameter(torch.Tensor(self.hidden_dim, self.input_dim))
        self.bias_ih = nn.Parameter(torch.Tensor(self.hidden_dim))
        self.weight_hh = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.bias_hh = nn.Parameter(torch.Tensor(self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_feature, hidden_states):
        output_hidden_state = F.tanh(F.linear(input_feature, self.weight_ih, self.bias_ih) + \
                                     F.linear(hidden_states, self.weight_hh, self.bias_hh))
        return output_hidden_state


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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, input_feature):
        # print(f'input_feature: {input_feature.shape}')
        # print(f'input_dim: {self.input_dim}')
        x = F.tanh(self.fc1(input_feature))
        x = F.tanh(self.fc2(x))
        output_hidden_state = self.fc3(x)

        return output_hidden_state

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

        if self.args.model == 'mlp':
            self.enc = MLP(input_dim=(self.args.edge_dim + self.args.direction_dim * 2) * self.args.seq_len,
                           hidden_dim=self.args.hidden_dim).to(self.args.device)
        elif self.args.model == 'rnn':
            self.enc = RNNCell(input_dim=self.args.edge_dim + self.args.direction_dim * 2, hidden_dim=self.args.hidden_dim).to(self.args.device)
        elif self.args.model == 'gru':
            self.enc = GRUCell(input_dim=self.args.edge_dim + self.args.direction_dim * 2, hidden_dim=self.args.hidden_dim).to(self.args.device)
        elif self.args.model in ['lstm', 'nettraj']:
            self.enc = LSTMCell(input_dim=self.args.edge_dim + self.args.direction_dim * 2, hidden_dim=self.args.hidden_dim).to(self.args.device)
        else:
            raise ValueError

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.out_link = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)
        if self.args.model in ['nettraj']:
            self.out_direction = nn.Linear(self.args.hidden_dim, self.args.direction * self.args.pre_len).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt, pred_d, gt_d, mask=None):

        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
            if self.args.model in ['nettraj']:
                loss += self.criterion(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], gt_d[:, dim])

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

            if self.args.model in ['mlp']:
                embs = embs.reshape((self.args.batch_size, -1))
                out = self.enc(embs)
            else:
                for i in range(embs.shape[1]):
                    input_i = embs[:, i, :]
                    if self.args.model in ['lstm', 'nettraj']:
                        _, hidden_states, cell_states = self.enc(input_i, hidden_states, cell_states)
                    else:
                        hidden_states = self.enc(input_i, hidden_states)
                out = hidden_states

            # (pred_len * edges)
            pred = self.out_link(out)
            if self.args.model in ['nettraj']:
                pred_d = self.out_direction(out)
            else:
                pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)

        return pred, pred_d, 0, torch.tensor([0])