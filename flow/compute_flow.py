import os
import gc
import copy
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
import torch.nn.functional as F

from tqdm import tqdm
from .test import eval_one_epoch
from .funcs import compute_recall_mrr


def compute_flow(args, loaders, model, opt_model):

    train_loaders, val_loaders, test_loaders = loaders

    # flow_counts = [0 for _ in range(args.num_edges)]
    # # data = None
    # for loader in test_loaders:
    #     for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
    #         print(x.shape)
    #         for traj in y:
    #             for link in traj:
    #                 flow_counts[link] += 1


    all_x, all_y, all_preds_topk, all_goald = None, None, None, None

    model.eval()
    with torch.no_grad():
        for loader in test_loaders:
            for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
                x = torch.LongTensor(x).to(args.device)
                y = torch.LongTensor(y).to(args.device)
                goal = y[:, -1]
                direction_x = torch.LongTensor(direction_x).to(args.device)
                direction_y = torch.LongTensor(direction_y).to(args.device)

                model.normalizeEmbedding()
                pred, pred_d, loss_kg, direction_correct = model(x, direction_x, length, goal)

                batch_mrr, batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, _, preds_topk, end_time, rank_time, b_multi_time = \
                    compute_recall_mrr(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                       model.padding_loglikelihood, model.batch_long, obs=x[:, -1] - 1, type=type, model=model)
                goal_directions = model.loc_dlabels_matrix[x[:, -1] - 1, goal]


                if all_x is None:
                    all_x = x.cpu()
                else:
                    all_x = torch.cat((all_x, x.cpu()), dim=0)

                if all_y is None:
                    all_y = y.cpu()
                else:
                    all_y = torch.cat((all_y, y.cpu()), dim=0)

                if all_goald is None:
                    all_goald = goal_directions.cpu()
                else:
                    all_goald = torch.cat((all_goald, goal_directions.cpu()), dim=0)

                if all_preds_topk is None:
                    all_preds_topk = preds_topk.cpu()
                else:
                    all_preds_topk = torch.cat((all_preds_topk, preds_topk.cpu()), dim=0)


    if args.goal:
        save_dir = os.path.join(args.base_dir, 'checkpoints', args.dataset, 'predictions', f'goal_')
    elif args.ed:
        save_dir = os.path.join(args.base_dir, 'checkpoints', args.dataset, 'predictions', f'estimate_goald_')
    else:
        save_dir = os.path.join(args.base_dir, 'checkpoints', args.dataset, 'predictions', f'goald_')

    print(f'all_x: {all_x.shape}, all_y: {all_y.shape}')
    print(f'all_preds_topk: {all_preds_topk.shape}')
    print(f'all_goald: {all_goald.shape}')
    torch.save(all_x, save_dir+f'all_x.pt')
    torch.save(all_y, save_dir+f'all_y.pt')
    torch.save(all_goald, save_dir+f'all_goald.pt')
    torch.save(all_preds_topk, save_dir+f'all_preds_topk.pt')











    test_acc, test_traj_acc, _, _, test_mrr, preds_topk = eval_one_epoch(test_loaders, args, model, opt_model, type='topk', epoch=99999)

    print(f'a: {args.a} | b: {args.b} | c: {args.c} | d: {args.d} | e: {args.e} | f: {args.f} | g: {args.g} | '
          f'[{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}], '
          f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}], '
          f'm: [{test_mrr[0]:.3f}, {test_mrr[1]:.3f}, {test_mrr[2]:.3f}, {test_mrr[3]:.3f}, {test_mrr[4]:.3f}]')

    assert 1==2
    preds_topk = preds_topk[..., :10]
    print(preds_topk.shape)

    reverse_ranks = 11 - np.arange(1, 11)
    temperature = args.temp

    # Apply temperature scaling
    scaled_reverse_ranks = reverse_ranks ** (1 / temperature)

    # Calculate the sum of temperature-scaled reversed ranks
    sum_scaled_reverse_ranks = np.sum(scaled_reverse_ranks)

    # Normalize to get probabilities
    probabilities = scaled_reverse_ranks / sum_scaled_reverse_ranks

    print(f'probabilities: {probabilities}')

    flow_counts = np.array(flow_counts)
    mae_list, rmse_list, r2_list = [], [], []

    for _ in range(5):
        pred_flow_counts = [0 for _ in range(args.num_edges)]
        for traj in preds_topk:
            sampled = np.random.choice(range(10), p=probabilities)
            traj = traj[:, sampled]

            for link in traj:
                pred_flow_counts[link] += 1
        pred_flow_counts = np.array(pred_flow_counts)


        mae = np.mean(np.abs(pred_flow_counts - flow_counts))

        # Calculate the Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((pred_flow_counts - flow_counts) ** 2))

        # Calculate the coefficient of determination (R^2)
        mean_ground_truth = np.mean(flow_counts)
        total_sum_of_squares = np.sum((flow_counts - mean_ground_truth) ** 2)
        residual_sum_of_squares = np.sum((flow_counts - pred_flow_counts) ** 2)
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)

        # print("MAE:", mae)
        # print("RMSE:", rmse)
        # print("R^2:", r_squared)
        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r_squared)


    print(f'MAE: {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f} | '
          f'RMSE: {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f} | '
          f'R2: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}')


    # fo = open(f"{args.result_dir}", "a")
    # fo.write(f'MAE: {np.mean(mae_list):.3f} ± {np.std(mae_list):.3f} | '
    #          f'RMSE: {np.mean(rmse_list):.3f} ± {np.std(rmse_list):.3f} | '
    #          f'R2: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}\n')
    # fo.close()

    assert 1==2