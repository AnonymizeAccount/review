import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


class NaiveEmbedding(nn.Module):
    def __init__(self, num_nodes, num_edges, node_dim, edge_dim, mask_node, mask_edge):
        super(NaiveEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.mask_node = mask_node
        self.mask_edge = mask_edge

        if not self.mask_node:
            self.emb_nodes = nn.Embedding(self.num_nodes, self.node_dim)
        if not self.mask_edge:
            self.emb_edges = nn.Embedding(self.num_edges, self.edge_dim)

    def forward(self, nodes, edges):
        nodes_embeddings, edges_embeddings = None, None
        if not self.mask_node:
            nodes_embeddings = self.emb_nodes(nodes)
        if not self.mask_edge:
            edges_embeddings = self.emb_edges(edges)

        assert nodes_embeddings is not None and edges_embeddings is not None

        return nodes_embeddings, edges_embeddings
