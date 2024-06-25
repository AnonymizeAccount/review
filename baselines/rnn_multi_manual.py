import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, Walker
from torch.nn.modules.rnn import RNNCellBase


# class DirectionEmbedding(nn.Module):
#     def __init__(self, args):
#         super(DirectionEmbedding, self).__init__()
#
#         self.args = args
#         angle = torch.tensor([-np.pi + 2 * np.pi / (2 * self.args.direction) + ((2 * np.pi) / self.args.direction) * i
#                               for i in range(self.args.direction)]).to(self.args.device)
#         self.raw_emb = torch.cat((torch.unsqueeze(torch.cos(angle), dim=1), torch.unsqueeze(torch.sin(angle), dim=1)), dim=1)
#         self.raw_emb /= torch.max(torch.abs(self.raw_emb))
#         self.transform = nn.Linear(2, self.args.direction_dim).to(self.args.device)
#
#     def forward(self, directions):
#         self.direction_embedding = self.transform(self.raw_emb)
#         return F.embedding(directions, self.direction_embedding)


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


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges):
        super(RNN, self).__init__()
        self.args = args
        self.node_adj_edges = node_adj_edges
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
        # self.direction_emblayer = DirectionEmbedding(self.args).to(self.args.device)
        self.enc = LSTMCell(input_dim=self.args.edge_dim + self.args.direction_dim, hidden_dim=self.args.hidden_dim).to(self.args.device)
        # self.enc = LSTMCell(input_dim=self.args.edge_dim, hidden_dim=self.args.hidden_dim).to(self.args.device)
        self.dropout = nn.Dropout(p=self.args.dropout)
        self.out_link = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).to(self.args.device)
        self.out_direction = nn.Linear(self.args.hidden_dim, self.args.direction * self.args.pre_len).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, pred, gt, pred_d, gt_d):

        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])
            loss += self.criterion(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], gt_d[:, dim])

        return loss


    def get_end_nodes(self, pred, gt, dim, end_node, cur_parent):
        cur_gt = gt[:, dim]
        cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
        cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
        node_adj_edges = self.node_adj_edges[end_node]
        last_pred = cur_parent.pred
        last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
        node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges
        cur_pred = cur_pred[self.ids, node_adj_edges]
        cur_pred = cur_pred.topk(k=self.args.multi, dim=1, largest=True, sorted=True)
        cur_pred_value, cur_pred = cur_pred[0], cur_pred[1]
        row, col = torch.where(cur_pred_value == float('-inf'))
        cur_pred.index_put_((row, col), cur_pred[row][:, 0])
        cur_pred_value.index_put_((row, col), cur_pred_value[row][:, 0])
        cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
        wrong_pred_idx = torch.where(cur_pred == self.args.num_edges)

        cur_pred[torch.where(cur_pred == self.args.num_edges)] -= self.offset
        for k in range(self.args.multi):
            cur_pred_k = cur_pred[:, k]
            end_node = self.graph_edges[cur_pred_k][:, 1] - 1
            end_node_k = Node(name=k, parent=cur_parent, data=end_node, pred=cur_pred_k)

    def get_multi_predictions(self, pred, pred_d, gt, direction_gt, obs=None):

        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1

        root = Node(name="root", data=end_node, pred=obs)
        for dim in range(self.args.pre_len):
            cur_leaves = root.leaves
            for node in cur_leaves:
                cur_parent = node
                end_node = node.data
                self.get_end_nodes(pred, gt, dim, end_node, cur_parent)
        walker = Walker()

        # print('='*100)
        preds, k = None, 0
        if self.args.topk < 0:
            topk = np.power(self.args.multi, self.args.pre_len)
        else:
            topk = self.args.topk
        # print(self.args.topk)
        while k < topk:
            if root.leaves[k].depth != self.args.pre_len:
                continue
            upwards, common, downwards = walker.walk(root, root.leaves[k])
            # print(f'node: {common.data.shape} {common.data}')
            # pred_k = torch.unsqueeze(common.data, dim=-1)
            pred_k = None
            for node in downwards:
                # print(f'node: {node.data.shape} {node.data}')
                if pred_k == None:
                    pred_k = torch.unsqueeze(node.pred, dim=-1)
                else:
                    pred_k = torch.cat((pred_k, torch.unsqueeze(node.pred, dim=-1)), dim=-1)
            # print(f'pred_{k}: {pred_k.shape}\n{pred_k}')
            if preds == None:
                preds = torch.unsqueeze(pred_k, dim=-1)
            else:
                preds = torch.cat((preds, torch.unsqueeze(pred_k, dim=-1)), dim=-1)
        # print(list(walker.walk(root, root.leaves[0])))
            k += 1
        return preds

    def get_predictions(self, pred, pred_d, gt, direction_gt, obs=None):
        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1
        path_total, path_right, d_total, d_right, traj_total, traj_right, prediction, prediction_d, last_pred = 0, 0, 0, 0, 0, None, None, None, None
        for dim in range(gt.shape[1]):
            cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
            cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
            node_adj_edges = self.node_adj_edges[end_node]
            if dim == 0:
                last_pred = obs
            last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
            node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges
            cur_pred = cur_pred[self.ids, node_adj_edges]
            cur_pred = cur_pred.max(1)[1][:, None]
            cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
            cur_pred[torch.where(cur_pred == self.args.num_edges)[0]] -= self.offset
            last_pred = cur_pred
            if dim != gt.shape[1] - 1:
                end_node = self.graph_edges[cur_pred][:, 1] - 1
            cur_pred_d = F.log_softmax(pred_d[:, dim * self.args.direction: (dim + 1) * self.args.direction], dim=1)
            cur_pred_d = cur_pred_d.max(1)[1]

            if prediction == None:
                prediction = torch.unsqueeze(cur_pred, dim=-1)
                prediction_d = torch.unsqueeze(cur_pred_d, dim=-1)
            else:
                prediction = torch.cat((prediction, torch.unsqueeze(cur_pred, dim=-1)), dim=-1)
                prediction_d = torch.cat((prediction_d, torch.unsqueeze(cur_pred_d, dim=-1)), dim=-1)

        return prediction, prediction_d

    def new_compute_acc(self, pred, pred_d, gt, direction_gt, obs=None, type=None, epoch=None):
        best_path_total, best_path_right, best_d_total, best_d_right, best_traj_total, best_traj_right, best_prediction = 0, 0, 1, 0, 0, 0, None
        preds, preds_d = self.get_predictions(pred, pred_d, gt, direction_gt, obs)

        if self.args.multi > 1:
            preds = self.get_multi_predictions(pred, pred_d, gt, direction_gt, obs=obs)
            if self.args.topk > 0:
                preds = preds[:, :, :self.args.topk]
            gt = torch.unsqueeze(gt, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
            preds = preds.permute(0, 2, 1)
            ele_right = torch.sum((gt == preds) * 1, dim=-1)
            batch_right = torch.max(ele_right, dim=-1)[0]
            best_traj_right = torch.sum((batch_right == self.batch_long) * 1)
            best_path_right = torch.sum(batch_right)
            d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
            best_d_right = torch.sum(d_ele_right)
        else:
            ele_right = torch.sum((gt == preds) * 1, dim=-1)
            best_traj_right = torch.sum((ele_right == self.batch_long) * 1)
            best_path_right = torch.sum(ele_right)
            d_ele_right = torch.sum((direction_gt == preds_d) * 1, dim=-1)
            best_d_right = torch.sum(d_ele_right)

        best_traj_total = self.args.batch_size
        best_path_total = self.args.batch_size * self.args.pre_len
        best_d_total = best_path_total

        return best_path_total, best_path_right, best_traj_total, best_traj_right, best_d_total, best_d_right, best_prediction, preds

    def forward(self, inputs, directions, mask):
        if self.args.rand:
            pred = torch.rand((self.args.batch_size, self.args.num_edges * self.args.pre_len)).to(self.args.device)
            pred_d = torch.rand((self.args.batch_size, self.args.direction * self.args.pre_len)).to(self.args.device)
        else:
            hidden_states = torch.zeros(self.args.batch_size, self.args.hidden_dim).to(inputs.device)
            cell_states = torch.zeros(self.args.batch_size, self.args.hidden_dim).to(inputs.device)

            link_embs = self.link_emblayer(inputs)
            direction_embs = self.direction_emblayer(directions - 1)
            # direction_embs = self.direction_emblayer(directions)
            embs = torch.cat((link_embs, direction_embs), dim=-1)
            # embs = link_embs + direction_embs
            for i in range(embs.shape[1]):
                input_i = embs[:, i, :]
                _, hidden_states, cell_states = self.enc(input_i, hidden_states, cell_states)
            out = hidden_states

            # (pred_len * edges)
            pred = self.out_link(out)
            # (pred_len * directions)
            pred_d = self.out_direction(out)

        return pred, pred_d