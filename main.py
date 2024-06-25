import os
import copy
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import setproctitle

from utils.setup import setup_seed, arg_parse, get_model, new_get_model
from utils.data import load_data
# from flow.compute_flow import compute_flow
# from trainer import train
# from utils.trainer_rn import train
from utils.new_trainer import train

# from baselines.hard.shortest_paths import train
# from baselines.markovs.markov import train
# from utils.nettraj_epoch import train


def main():
    args = arg_parse(argparse.ArgumentParser())
    setup_seed(args)
    loader, graph, node_adj_edges, direction_labels, loc_direct_matrix, \
        loc_dist_matrix, loc_dlabels_matrix, TM, edge_A, length_shortest_paths, = load_data(args)
    model, optimizer = get_model(args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix, loc_dlabels_matrix,
                                 TM, edge_A, length_shortest_paths)
    # MLP, opt_mlp, model, opt_model = new_get_model(args, graph, node_adj_edges, direction_labels, loc_direct_matrix, loc_dist_matrix,
    #                                                loc_dlabels_matrix, TM, edge_A, length_shortest_paths)
    print(f'Arguments: \n\n{args}\n\n')
    # train(loader, args, MLP, opt_mlp, model, opt_model)
    train(loader, args, model, optimizer)


if __name__ == '__main__':
    setproctitle.setproctitle("Yihong's Process")
    main()