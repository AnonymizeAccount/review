import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
