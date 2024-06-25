import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, Walker

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
           其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
            相当于一个Residual block

            :param n_inputs: int, 输入通道数
            :param n_outputs: int, 输出通道数
            :param kernel_size: int, 卷积核尺寸
            :param stride: int, 步长，一般为1
            :param dilation: int, 膨胀系数
            :param padding: int, 填充系数
            :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding) # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding) #  裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
            参数初始化

            :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
            :param x: size of (Batch, input_channel, seq_len)
            :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
            TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
            对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
            对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

            :param num_inputs: int， 输入通道数
            :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
            :param kernel_size: int, 卷积核尺寸
            :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1] # 确定每一层的输入通道数
            out_channels = num_channels[i] # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
            输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
            这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
            很巧妙的设计。

            :param x: size of (Batch, input_channel, seq_len)
            :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)



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

        num_channels = [self.args.hidden_dim] * 2
        print(f'num_channels: {num_channels}')
        self.enc = TemporalConvNet(self.args.edge_dim + self.args.direction_dim, num_channels, 3, dropout=self.args.dropout).to(self.args.device)

        # if self.args.model == 'rnn':
        #     self.enc = nn.RNN(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
        #                       batch_first=True, dropout=self.args.dropout).to(self.args.device)
        # elif self.args.model == 'gru':
        #     self.enc = nn.GRU(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
        #                       batch_first=True, dropout=self.args.dropout).to(self.args.device)
        # elif self.args.model == 'lstm':
        #     self.enc = nn.LSTM(self.args.edge_dim + self.args.direction_dim, self.args.hidden_dim, 1,
        #                        batch_first=True, dropout=self.args.dropout).to(self.args.device)
        # else:
        #     raise ValueError

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
            self.args.topk = np.power(self.args.multi, self.args.pre_len)
        # print(self.args.topk)
        while k < self.args.topk:
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

    def new_compute_acc(self, pred, pred_d, gt, direction_gt, obs=None, type=None, epoch=None):
        preds = self.get_multi_predictions(pred, pred_d, gt, direction_gt, obs=obs)
        best_path_total, best_path_right, best_d_total, best_d_right, best_traj_total, best_traj_right, best_prediction = 0, 0, 1, 0, 0, 0, None

        gt = torch.unsqueeze(gt, dim=-1).expand(-1, -1, preds.shape[-1]).permute(0, 2, 1)
        preds = preds.permute(0, 2, 1)

        ele_right = torch.sum((gt == preds) * 1, dim=-1)
        batch_right = torch.max(ele_right, dim=-1)[0]
        best_traj_right = torch.sum((batch_right == self.batch_long) * 1)
        best_path_right = torch.sum(batch_right)
        best_traj_total = self.args.batch_size
        best_path_total = self.args.batch_size * self.args.pre_len

        return best_path_total, best_path_right, best_traj_total, best_traj_right, best_d_total, best_d_right, best_prediction, preds

    def forward(self, inputs, directions, mask):

        link_embs = self.link_emblayer(inputs)
        direction_embs = self.direction_emblayer(directions)
        embs = torch.cat((link_embs, direction_embs), dim=-1)
        # print(f'embs: {embs.shape}')
        out = self.enc(embs.permute(0, 2, 1))[:, :, -1]
        # print(f'out: {out.shape}')
        # assert 1==2
        pred = self.out_link(out)
        pred_d = self.out_direction(out)

        return pred, pred_d
