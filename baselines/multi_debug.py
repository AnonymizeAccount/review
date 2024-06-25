import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, Walker


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, nodes_links_directions):
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

        if self.args.model == 'rnn':
            self.enc = nn.RNN(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'gru':
            self.enc = nn.GRU(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
                              batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'lstm':
            self.enc = nn.LSTM(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
                               batch_first=True, dropout=self.args.dropout).to(self.args.device)
        else:
            raise ValueError

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
        # print(f'cur_pred0: {cur_pred.shape}\n{cur_pred}')
        cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
        # print(f'cur_pred1: {cur_pred.shape}\n{cur_pred}')
        # print(f'self.node_adj_edges: {self.node_adj_edges.shape}')
        node_adj_edges = self.node_adj_edges[end_node]
        # print(f'node_adj_edges: {node_adj_edges.shape}\n{node_adj_edges[:, :6]}')


        last_pred = cur_parent.pred
        # print(f'last_pred: {last_pred.shape}\n{last_pred}')
        last_pred = torch.unsqueeze(last_pred, dim=-1).expand(-1, node_adj_edges.shape[1])
        # print(f'last_pred: {last_pred.shape}\n{last_pred}')

        node_adj_edges[torch.where(last_pred == node_adj_edges)] = self.args.num_edges

        # print(torch.where(last_pred == node_adj_edges[:, 0]))
        # print(torch.where(last_pred == node_adj_edges[:, 1]))
        # print(torch.where(last_pred == node_adj_edges[:, 2]))
        # print(torch.where(last_pred == node_adj_edges[:, 3]))
        # print(torch.where(last_pred == node_adj_edges[:, 4]))
        # print(torch.where(last_pred == node_adj_edges[:, 5]))
        # print(torch.where(last_pred == node_adj_edges[:, 6]))
        # print(torch.where(last_pred == node_adj_edges[:, 7]))


        # assert 1==2
        cur_pred = cur_pred[self.ids, node_adj_edges]
        # print(f'cur_pred2: {cur_pred.shape}\n{cur_pred}')
        # cur_pred = cur_pred.max(1)[1][:, None]
        cur_pred = cur_pred.topk(k=self.args.multi, dim=1, largest=True, sorted=True)
        cur_pred_value, cur_pred = cur_pred[0], cur_pred[1]
        # print(f'cur_pred_value: {cur_pred_value.shape}\n{cur_pred_value}')
        # print(f'cur_pred3: {cur_pred.shape}\n{cur_pred}')
        # print(f'cur_gt: {cur_gt.shape}\n{cur_gt}')

        # print(torch.where(cur_pred_value == float('-inf')))
        # print(cur_pred[torch.where(cur_pred_value == float('-inf'))[0]])
        # print(cur_pred_value[torch.where(cur_pred_value == float('-inf'))[0]])

        row, col = torch.where(cur_pred_value == float('-inf'))
        # print(torch.where(cur_pred_value == float('-inf')))
        # idx_with_pad = torch.where(cur_pred_value == float('-inf'))[0]
        # print(f'cur_pred[idx_with_pad]: \n{cur_pred[row]}')
        # print(f'cur_pred_value[idx_with_pad]: \n{cur_pred_value[row]}')
        # cur_pred[idx_with_pad][:, -1], cur_pred_value[idx_with_pad][:, -1] = cur_pred[idx_with_pad][:, 0], cur_pred_value[idx_with_pad][:, 0]

        cur_pred.index_put_((row, col), cur_pred[row][:, 0])
        cur_pred_value.index_put_((row, col), cur_pred_value[row][:, 0])

        # print(f'cur_pred[idx_with_pad]: \n{cur_pred[row]}')
        # print(f'cur_pred_value[idx_with_pad]: \n{cur_pred_value[row]}')
        # print(f'cur_pred3: {cur_pred.shape}\n{cur_pred[idx_with_pad]}')

        cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
        # print(f'cur_pred4: {cur_pred.shape}{torch.max(cur_pred)}\n{cur_pred}')
        # if torch.max(cur_pred) == self.args.num_edges:
        #     return
        wrong_pred_idx = torch.where(cur_pred == self.args.num_edges)
        # print(f'wrong_pred_idx: \n{wrong_pred_idx}')
        # print(f'idx_with_pad: {row}')
        # print(f'wrong_pred_idx: \n{wrong_pred_idx}')
        # if len(wrong_pred_idx) > 0:
        # assert 1==2

        cur_pred[torch.where(cur_pred == self.args.num_edges)] -= self.offset

        # if dim != gt.shape[1] - 1:
        for k in range(self.args.multi):
            cur_pred_k = cur_pred[:, k]
            end_node = self.graph_edges[cur_pred_k][:, 1] - 1
            end_node_k = Node(name=k, parent=cur_parent, data=end_node, pred=cur_pred_k)

    def get_multi_predictions(self, pred, pred_d, gt, direction_gt, obs=None):
        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1
        root = Node(name="root", data=end_node, pred=obs)

        for dim in tqdm(range(self.args.pre_len)):

            cur_leaves = root.leaves
            for node in cur_leaves:
                cur_parent = node
                end_node = node.data
                self.get_end_nodes(pred, gt, dim, end_node, cur_parent)

        walker = Walker()

        # print('='*100)
        preds, k = None, 0
        if self.args.topk < 0:
            self.args.topk = np.power(self.args.multi, self.args.pre_len)
        # print(self.args.topk)
        for k in tqdm(range(self.args.topk)):
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
        return preds

    def new_compute_acc(self, pred, pred_d, gt, direction_gt, obs=None, type=None, epoch=None):


        # print(f'obs: {obs}, gt: {gt}')
        # end_node = self.graph_edges[obs][:, 1] - 1
        # node_adj_edges = self.node_adj_edges[end_node]
        # print(f'node_adj_edges: \n{node_adj_edges}')


        preds = self.get_multi_predictions(pred, pred_d, gt, direction_gt, obs=obs)
        best_path_total, best_path_right, best_d_total, best_d_right, best_traj_total, best_traj_right, best_prediction = 0, 0, 1, 0, 0, 0, None
        # print(f'preds: {preds.shape}, gt: {gt.shape}')
        # for i in range(gt.shape[1]):
        #     cur_gt = gt[:, i]
        #     # print(f'cur_pred: {cur_pred.shape}, cur_gt: {cur_gt.shape}')
        #     end_node = self.graph_edges[cur_gt][:, 1] - 1
        #     node_adj_edges = self.node_adj_edges[end_node]
        #     print(f'node_adj_edges: \n{node_adj_edges}')

        gt = torch.unsqueeze(gt, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
        preds = preds.permute(0, 2, 1)

        ele_right = torch.sum((gt == preds) * 1, dim=-1)
        batch_right = torch.max(ele_right, dim=-1)[0]
        traj_right = torch.sum((batch_right == self.batch_long) * 1)
        path_right = torch.sum(batch_right)
        best_traj_total = self.args.batch_size
        best_path_total = self.args.batch_size * self.args.pre_len


        # for k in tqdm(range(preds.shape[2])):
        #     pred_k = preds[:, :, k]
        #     print(pred_k)
        #     path_total, path_right, d_total, d_right, traj_total, traj_right, prediction = 0, 0, 0, 0, 0, None, None
        #     for i in range(pred_k.shape[1]):
        #         cur_pred = pred_k[:, i]
        #         cur_gt = gt[:, i]
        #         # print(f'cur_pred: {cur_pred.shape}, cur_gt: {cur_gt.shape}')
        #         # end_node = self.graph_edges[cur_gt][:, 1] - 1
        #         # node_adj_edges = self.node_adj_edges[end_node]
        #         # print(f'node_adj_edges: \n{node_adj_edges}')
        #         path_total += cur_gt.shape[0]
        #         path_right += torch.sum((cur_gt == cur_pred) * 1)
        #         if traj_right is None:
        #             traj_right = (cur_gt == cur_pred)
        #         else:
        #             traj_right *= (cur_gt == cur_pred)
        #
        #         if prediction == None:
        #             prediction = torch.unsqueeze(cur_pred, dim=-1)
        #         else:
        #             prediction = torch.cat((prediction, torch.unsqueeze(cur_pred, dim=-1)), dim=-1)
        #
        #     # print(f'pred: {pred_k}, gt: {gt}')
        #
        #     traj_right = torch.sum(traj_right * 1)
        #     # print(f'traj_right: {traj_right}, best: {best_traj_right}')
        #     # assert 1==2
        if traj_right > best_traj_right:
            best_traj_right = traj_right
            best_path_right = path_right
            # best_prediction = prediction
            # best_path_total = path_total

        # best_traj_total = gt.shape[0]

        # print(f'path: {best_path_right}/{best_path_total} | traj: {best_traj_right}/{best_traj_total}')
        # assert 1==2
        return best_path_total, best_path_right, best_traj_total, best_traj_right, best_d_total, best_d_right, best_prediction, preds

    def forward(self, inputs, directions, mask):
        link_embs = self.link_emblayer(inputs)
        direction_embs = self.direction_emblayer(directions)
        embs = torch.cat((link_embs, direction_embs), dim=-1)
        l_pack = pack_padded_sequence(embs, mask, batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.enc(l_pack)
        out = torch.squeeze(h_n, dim=0)
        pred = self.out_link(out)
        pred_d = self.out_direction(out)

        return pred, pred_d