import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP_model(nn.Module):
    def __init__(self, input_dim, args):
        super(MLP_model, self).__init__()

        self.args = args
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, self.args.hidden_dim)
        self.fc2 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.fc3 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
        self.out = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)

    def forward(self, input_feature):
        # x = F.tanh(self.fc1(input_feature))
        # x = F.tanh(self.fc2(x))
        x = F.relu(self.fc1(input_feature))
        x = F.relu(self.fc2(x))
        output_hidden_state = self.fc3(x)

        return self.out(output_hidden_state)


# class MLP_model(nn.Module):
#     def __init__(self, input_dim, args):
#         super(MLP_model, self).__init__()
#
#         self.args = args
#         self.input_dim = input_dim
#
#         self.link_emblayer = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0).to(self.args.device)
#         self.direction_emblayer = nn.Embedding(self.args.direction + 1, self.args.direction_dim, padding_idx=0).to(self.args.device)
#
#         # self.fc1 = nn.Linear(self.args.edge_dim * 2 + self.args.direction_dim * 2, self.args.hidden_dim)
#         self.fc1 = nn.Linear(self.input_dim, self.args.hidden_dim)
#         self.fc2 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
#         self.fc3 = nn.Linear(self.args.hidden_dim, self.args.hidden_dim)
#         self.out = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)
#
#     def forward(self, inputs, directions, mask, goal):
#
#         # last_obs = inputs[:, -1]
#         # (batch_size), value: 1 ~ num_edges
#         # self.goal_directions = self.loc_dlabels_matrix[last_obs-1, goal] + 1
#         link_embs = self.link_emblayer(inputs)
#         direction_embs = self.direction_emblayer(directions)
#         goal_embs = self.link_emblayer(goal + 1)
#
#         link_embs = link_embs[:, :self.args.obs_len, :]
#         direction_embs = direction_embs[:, :self.args.obs_len, :]
#         # link_embs = link_embs[:, -1, :]
#         # direction_embs = direction_embs[:, -1, :]
#         # goald_embs = self.direction_emblayer(self.goal_directions)
#
#         # print(f'link_embs: {link_embs.shape}')
#         # print(f'direction_embs: {direction_embs.shape}')
#         # print(f'goald_embs: {goald_embs.shape}')
#         # print(f'goal_embs: {goal_embs.shape}')
#         # assert 1==2
#
#         # embs = torch.cat((link_embs, direction_embs, goal_embs, goald_embs), dim=-1)
#         goal_embs = torch.unsqueeze(goal_embs, dim=1).expand(self.args.batch_size, link_embs.shape[1], -1)
#         embs = torch.cat((link_embs, direction_embs, goal_embs), dim=-1)
#         embs = embs.reshape((self.args.batch_size, -1))
#
#         x = F.tanh(self.fc1(embs))
#         x = F.tanh(self.fc2(x))
#         output_hidden_state = self.fc3(x)
#
#         return self.out(output_hidden_state)