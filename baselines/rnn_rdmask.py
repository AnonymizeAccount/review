import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
import numpy as np


class NaiveEmbedding(nn.Module):
    def __init__(self, args):
        super(NaiveEmbedding, self).__init__()

        self.args = args
        # self.num_nodes = args.num_nodes
        # self.num_edges = args.num_edges
        # self.node_dim = args.node_dim
        # self.edge_dim = args.edge_dim
        # self.mask_node = args.mask_node
        # self.mask_edge = args.mask_edge
        self.emb_edges = nn.Embedding(self.args.num_edges + 1, self.args.edge_dim, padding_idx=0)

    def forward(self, inputs):
        embeddings = self.emb_edges(inputs)

        # assert nodes_embeddings is not None and edges_embeddings is not None
        return embeddings


# class RnnFactory(nn.Module):
#     def __init__(self, args):
#         self.args = args
#         self.rnn_type = args.model
#
#     def create(self, input_dim, hidden_dim, num_layer, dropout=0):
#         if self.args.mgpu:
#             if self.rnn_type == 'RNN':
#                 return nn.RNN(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).cuda()
#             if self.rnn_type == 'GRU':
#                 return nn.GRU(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).cuda()
#             if self.rnn_type == 'LSTM':
#                 return nn.LSTM(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).cuda()
#         else:
#             if self.rnn_type == 'RNN':
#                 return nn.RNN(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).to(self.args.device)
#             if self.rnn_type == 'GRU':
#                 return nn.GRU(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).to(self.args.device)
#             if self.rnn_type == 'LSTM':
#                 return nn.LSTM(self.args.edge_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout).to(self.args.device)


class RNN(nn.Module):
    def __init__(self, args, graph, node_adj_edges, nodes_links_directions):
        super(RNN, self).__init__()
        self.args = args
        self.node_adj_edges = node_adj_edges
        self.graph = graph
        self.graph_edges = torch.LongTensor(np.array(self.graph.edges)).to(self.args.device)

        # if self.args.efficient:
            # if self.args.dataset in ['shanghai']:
        node_adj_edges_list = []
        for i in range(self.node_adj_edges.shape[0]):
            node_adj_edges_list.append(torch.from_numpy(self.node_adj_edges[i]))
        self.node_adj_edges = \
            pad_sequence(node_adj_edges_list, batch_first=True, padding_value=self.args.num_edges).to(self.args.device)
        self.padding_loglikelihood = torch.unsqueeze(torch.ones(self.args.batch_size) * -100, dim=-1).to(self.args.device)
        self.ids = torch.arange(self.args.batch_size)[:, None].to(self.args.device)
        # offset used for continue training when predicted trajectory has not other route to select
        self.offset = torch.randint(1, self.args.num_edges, (1,)).to(self.args.device)
        # else:
        #     self.node_adj_edges = torch.LongTensor(self.node_adj_edges).to(self.args.device)

        self.embeddinglayer = NaiveEmbedding(self.args).cuda()
        # self.model = RnnFactory(args).create(input_dim=1, hidden_dim=args.hidden_dim, num_layer=1, dropout=0)

        if self.args.model == 'rnn':
            self.model = nn.RNN(self.args.edge_dim, self.args.hidden_dim, 1, batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'gru':
            self.model = nn.GRU(self.args.edge_dim, self.args.hidden_dim, 1, batch_first=True, dropout=self.args.dropout).to(self.args.device)
        elif self.args.model == 'lstm':
            self.model = nn.LSTM(self.args.edge_dim, self.args.hidden_dim, 1, batch_first=True, dropout=self.args.dropout).to(self.args.device)
        else:
            raise ValueError

        self.dropout = nn.Dropout(p=self.args.dropout)
        self.linear = nn.Linear(self.args.hidden_dim, self.args.num_edges * self.args.pre_len).cuda()
        self.criterion = nn.CrossEntropyLoss()

    def get_mask(self, node_adj_edges):
        # print(node_adj_edges.shape)
        # assert 1==2
        new_node_adj_edges = np.ones((node_adj_edges.shape[0], self.args.num_edges))
        for i in range(node_adj_edges.shape[0]):
            new_node_adj_edges[i][node_adj_edges[i]] = 0

        if self.args.mgpu:
            return torch.LongTensor(new_node_adj_edges).cuda()
        else:
            return torch.LongTensor(new_node_adj_edges).to(self.args.device)

    def compute_loss(self, pred, gt):

        # print(f'pred: {pred.shape}\n{pred}')
        # print(f'gt: {gt.shape}\n{gt}')
        loss = 0
        for dim in range(gt.shape[1]):
            loss += self.criterion(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], gt[:, dim])

        return loss

    def new_compute_acc(self, pred, gt):

        end_node = self.graph_edges[gt[:, 0]][:, 0] - 1

        path_total, path_right = 0, 0
        traj_total, traj_right = 0, None
        prediction = None
        # print(gt.shape)
        for dim in range(gt.shape[1]):
            cur_gt = gt[:, dim]
            cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
            # print(cur_pred.shape)
            # assert 1==2
            cur_pred = torch.cat((cur_pred, self.padding_loglikelihood), dim=-1)
            # print(cur_pred.shape)
            # print(cur_pred)
            # print(torch.min(cur_pred))
            node_adj_edges = self.node_adj_edges[end_node]
            cur_pred = cur_pred[self.ids, node_adj_edges]

            # for i in range(len(cur_gt)):
            #     # print(f'cur_gt: {cur_gt[i]}')
            #     if cur_gt[i] not in node_adj_edges[i, :] or cur_gt[i] == self.args.num_edges:
            #         print(f'Bug here!!')
            #         print(f'i: {i} / {len(cur_gt)}')
            #         print(f'cur_gt: {cur_gt[i]}')
            #         print(f'node_adj_edges: {node_adj_edges[i, :]}')
            #         assert 1==2

            # print(f'cur_gt: {cur_gt.shape}\n{cur_gt}')
            # print(f'node_adj_edges: {node_adj_edges.shape}\n{node_adj_edges}')
            # print(f'cur_pred: {cur_pred.shape}\n{cur_pred}')
            cur_pred = cur_pred.max(1)[1][:, None]
            # print(cur_pred.shape)
            # print(cur_pred)
            # print(f'idX:{node_adj_edges[self.ids, cur_pred].shape}\n{node_adj_edges[self.ids, cur_pred]}')
            cur_pred = torch.squeeze(node_adj_edges[self.ids, cur_pred])
            cur_pred[torch.where(cur_pred == self.args.num_edges)[0]] -= self.offset
            # cur_pred = cur_gt

            # print(cur_pred.shape)
            # print(cur_pred)
            # assert 1==2
            # print(cur_pred, torch.max(cur_pred))
            # print(cur_pred[torch.where(cur_pred==self.args.num_edges)[0]] -= torch.randint(self.args.num_edges, (1,)))
            # print(curr[torch.where(cur_pred==self.args.num_edges)[0], :])
            #
            # assert 1==2
            # else:
            #     cur_pred = cur_pred - cur_pred * self.node_adj_edges[end_node] - 100 * self.node_adj_edges[end_node]
            #     cur_pred = cur_pred.max(1)[1]

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

    def forward(self, inputs, mask):
        emb_inputs = self.embeddinglayer(inputs)
        l_pack = pack_padded_sequence(emb_inputs, mask, batch_first=True, enforce_sorted=False)
        _, (out, _) = self.model(l_pack)
        out = torch.squeeze(out)
        out = self.dropout(out)
        pred = self.linear(out)

        return pred





























    # def compute_acc(self, pred, gt):
    #
    #     # print(f'pred: {pred.shape}\n{pred}')
    #     # print(f'gt: {gt.shape}\n{gt}')
    #
    #     # print(self.graph.edges)
    #     # print()
    #     # print(list(self.graph.edges))
    #
    #     end_node = self.graph_edges[gt[:, 0]][:, 0] - 1
    #
    #     path_total, path_right = 0, 0
    #     traj_total, traj_right = 0, None
    #     count = 0
    #     prediction = None
    #     for dim in range(gt.shape[1]):
    #         cur_gt = gt[:, dim]
    #         cur_pred = F.log_softmax(pred[:, dim * self.args.num_edges: (dim + 1) * self.args.num_edges], dim=1)
    #         # cur_pred = cur_pred.detach().cpu().numpy()
    #
    #         # for i in range(cur_pred.shape[0]):
    #         #     cur_sample_pred = cur_pred[i, :]
    #         #     cur_sample_pred[torch.unique(self.node_adj_edges[end_node][i])] = -100
    #         #     cur_pred[i, :] = cur_sample_pred
    #         #     if len(self.node_adj_edges[end_node][i]) == self.args.num_edges and dim != gt.shape[1]-1:
    #         #         count += 1
    #         # print(end_node.device)
    #         # print(self.node_adj_edges.device)
    #         # if self.args.efficient:
    #         node_adj_edges = self.get_mask(self.node_adj_edges[end_node.detach().cpu().numpy()])
    #         cur_pred = cur_pred - cur_pred * node_adj_edges - 100 * node_adj_edges
    #         # else:
    #         #     if self.args.dataset in ['shanghai']:
    #         #         # cur_pred = cur_pred - cur_pred * torch.LongTensor(self.node_adj_edges[end_node.detach().cpu().numpy()]).to(self.args.device) \
    #         #         #            - 100 * torch.LongTensor(self.node_adj_edges[end_node.detach().cpu().numpy()]).to(self.args.device)
    #         #         if self.args.mgpu:
    #         #             batch_node_adj_edges = torch.LongTensor(self.node_adj_edges[end_node.detach().cpu().numpy()]).cuda()
    #         #         else:
    #         #             batch_node_adj_edges = torch.LongTensor(self.node_adj_edges[end_node.detach().cpu().numpy()]).to(self.args.device)
    #         #
    #         #         cur_pred = cur_pred - cur_pred * batch_node_adj_edges - 100 * batch_node_adj_edges
    #         #     else:
    #         #         cur_pred = cur_pred - cur_pred * self.node_adj_edges[end_node] - 100 * self.node_adj_edges[end_node]
    #
    #         # print(f'cur_pred: {cur_pred.shape}\n{cur_pred}')
    #         # cur_pred = torch.tensor(cur_pred).to(cur_gt.device)
    #         cur_pred = cur_pred.max(1)[1]
    #
    #         # end_node = np.array(self.graph.edges)[cur_pred.detach().cpu().numpy()][:, 1] - 1
    #         end_node = self.graph_edges[cur_pred][:, 1] - 1
    #
    #         # print(f'cur_pred: {cur_pred.shape}\n{cur_pred}')
    #         # cur_pred = torch.ShortTensor(cur_pred).to(cur_gt.device)
    #         # print(f'cur_gt: {cur_gt.shape}\n{cur_gt}')
    #
    #         # assert 1==2
    #
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
    #     # print(total, right, count)
    #     traj_total = gt.shape[0]
    #     traj_right = torch.sum(traj_right * 1)
    #
    #     return path_total, path_right, traj_total, traj_right, prediction

























# self.args = args
# self.graph = graph
# self.graph_edges = torch.LongTensor(np.array(self.graph.edges)).to(self.args.device)
#
# # 1 ..
# self.out_edges = {}

# for node in self.graph.nodes:
#     # print(f'node: {node}')
#     for edge in self.graph.out_edges(node):
#         # print(f'edge: {edge}')
#         start_node, end_node = edge
#
#         if start_node not in self.out_edges.keys():
#             self.out_edges[start_node] = [end_node]
#         else:
#             if end_node not in self.out_edges[start_node]:
#                 self.out_edges[start_node].append(end_node)
#
#     for edge in self.graph.in_edges(node):
#         # print(f'edge: {edge}')
#         start_node, end_node = edge
#         if end_node not in self.out_edges.keys():
#             self.out_edges[end_node] = [start_node]
#         else:
#             if start_node not in self.out_edges[end_node]:
#                 self.out_edges[end_node].append(start_node)

#     print(f'self.out_edges: {self.out_edges}')
#     assert 1==2

# for node in self.graph.nodes:
#     # print(f'node: {node}')
#     for edge in self.graph.out_edges(node):
#         # print(f'edge: {edge}')
#         start_node, end_node = edge
#
#         # mark = False
#         # for path in self.graph[start_node][end_node].keys():
#         #     # print(self.graph[start_node][end_node][path]["oneway"])
#         #     if self.graph[start_node][end_node][path]["oneway"] == "False":
#         #         mark = True
#         #         break
#
#         if start_node not in self.out_edges.keys():
#             self.out_edges[start_node] = [end_node]
#         else:
#             if end_node not in self.out_edges[start_node]:
#                 self.out_edges[start_node].append(end_node)
#
#         # if mark:
#         #     # print(f'yes')
#         #     if end_node not in self.out_edges.keys():
#         #         self.out_edges[end_node] = [start_node]
#         #     else:
#         #         if start_node not in self.out_edges[end_node]:
#         #             self.out_edges[end_node].append(start_node)
#
#     # print(f'self.out_edges: {self.out_edges}')
#     # assert 1==2
#
# for i in range(1, len(self.graph.nodes)+1):
#     if i not in self.out_edges.keys():
#         # print(f'yes')
#         self.out_edges[i] = []
#
# self.node_adj_edges = []
# # for start_node in self.out_edges.keys():
# #     adj_edge_list = [i for i in range(len(self.graph.edges))]
# #     for end_node in self.out_edges[start_node]:
# #         if self.graph.has_edge(start_node, end_node):
# #             for path in self.graph[start_node][end_node].keys():
# #                 # print(self.graph[start_node][end_node][path])
# #                 # idx = self.graph[start_node][end_node][path]["idx"]
# #                 # print(f'idx: {idx}')
# #                 # assert 1==2
# #
# #                 # adj_edge_list.append(self.graph[start_node][end_node][path]["idx"])
# #                 # print(self.graph[start_node][end_node][path]["idx"])
# #                 if self.graph[start_node][end_node][path]["idx"] in adj_edge_list:
# #                     adj_edge_list.remove(self.graph[start_node][end_node][path]["idx"])
# #
# #         if self.graph.has_edge(end_node, start_node):
# #             for path in self.graph[end_node][start_node].keys():
# #                 if self.graph[end_node][start_node][path]["idx"] in adj_edge_list:
# #                     adj_edge_list.remove(self.graph[end_node][start_node][path]["idx"])
#
# # for start_node in self.out_edges.keys():
# #     adj_edge_list = [i for i in range(len(self.graph.edges))]
# #     for end_node in self.out_edges[start_node]:
# #         if self.graph.has_edge(start_node, end_node):
# #             for path in self.graph[start_node][end_node].keys():
# #                 if self.graph[start_node][end_node][path]["idx"] in adj_edge_list:
# #                     adj_edge_list.remove(self.graph[start_node][end_node][path]["idx"])
# #
# #         if self.graph.has_edge(end_node, start_node):
# #             for path in self.graph[end_node][start_node].keys():
# #                 if self.graph[end_node][start_node][path]["idx"] in adj_edge_list:
# #                     adj_edge_list.remove(self.graph[end_node][start_node][path]["idx"])
# #     cur_len = len(adj_edge_list)
# #     for i in range(len(self.graph.edges) - cur_len):
# #         adj_edge_list.append(adj_edge_list[-1])
#
# self.out_edges = dict(sorted(self.out_edges.items()))
# # print(f'self.out_edges.keys(): \n{self.out_edges.keys()}')
# for start_node in self.out_edges.keys():
#     adj_edge_list = [True for i in range(len(self.graph.edges))]
#     for end_node in self.out_edges[start_node]:
#         if self.graph.has_edge(start_node, end_node):
#             for path in self.graph[start_node][end_node].keys():
#                 if adj_edge_list[self.graph[start_node][end_node][path]["idx"]] is True:
#                     adj_edge_list[self.graph[start_node][end_node][path]["idx"]] = False
#
#         if self.graph.has_edge(end_node, start_node):
#             for path in self.graph[end_node][start_node].keys():
#                 if adj_edge_list[self.graph[end_node][start_node][path]["idx"]] is True:
#                     adj_edge_list[self.graph[end_node][start_node][path]["idx"]] = False
#
#     # cur_len = len(adj_edge_list)
#     # for i in range(len(self.graph.edges) - cur_len):
#     #     adj_edge_list.append(adj_edge_list[-1])
#
#     self.node_adj_edges.append(np.array(adj_edge_list, dtype=np.int))
# self.node_adj_edges = np.array(self.node_adj_edges)
# self.node_adj_edges = torch.LongTensor(self.node_adj_edges).to(self.args.device)
# # print(torch.sum(self.node_adj_edges))
# # assert 1==2
#
# self.embeddinglayer = NaiveEmbedding(args)
# self.model = RnnFactory(args).create(input_dim=1, hidden_dim=args.hidden_dim, num_layer=1, dropout=0)
# self.linear = nn.Linear(self.args.hidden_dim, self.args.num_edges * args.pre_len)
# self.criterion = nn.CrossEntropyLoss()