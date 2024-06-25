import os
import copy
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import geopandas as gpd

from tqdm import tqdm


def train(loader, args, model, optimizer):
    start_time = time.time()
    TM = model.transition_matrix.float()
    print(f'TM: {TM.shape}, sum: {torch.sum(TM)}, min: {torch.min(TM)}, max: {torch.max(TM)}')
    train_loaders, val_loaders, test_loaders = loader
    best_acc, best_traj_acc, best_d_acc, best_goal_d_acc, count_patience, state, dur = -1, -1, -1, 0, 0, None, []
    total, traj_total = 0, 0
    right, traj_right, all_mrr = [0 for _ in range(5)], [0 for _ in range(5)], [0 for _ in range(5)]

    all_time_list = []
    # for _ in range(20):
    all_time, traj_num = 0, 0

    for loader in test_loaders:
        for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):

            batch_start_time = time.time()
            # print(f'x: {x.shape}, y: {y.shape}')
            x = torch.LongTensor(x).to(args.device)
            y = torch.LongTensor(y).to(args.device)

            traj_num += x.shape[0]

            all_topk_prob = None
            for _ in range(args.multi ** args.pre_len):

                batch_prob = TM[x[:, -1]-1]
                # print(f'batch_prob: {batch_prob.shape}')

                all_prob = None
                for _ in range(args.pre_len):
                    batch_prob = torch.squeeze(torch.multinomial(F.softmax(batch_prob, dim=1), 1))
                    # print(f'batch_prob: {batch_prob.shape}, max: {torch.max(batch_prob)}, min: {torch.min(batch_prob)}')

                    if all_prob is None:
                        all_prob = torch.unsqueeze(batch_prob, dim=-1)
                    else:
                        all_prob = torch.cat((all_prob, torch.unsqueeze(batch_prob, dim=-1)), dim=-1)
                    # print(f'all_prob: {all_prob.shape}')

                    batch_prob = TM[batch_prob]

                if all_topk_prob == None:
                    all_topk_prob = torch.unsqueeze(all_prob, dim=1)
                else:
                    all_topk_prob = torch.cat((all_topk_prob, torch.unsqueeze(all_prob, dim=1)), dim=1)

                # print(f'all_topk_prob: {all_topk_prob.shape}')

            all_time += time.time() - batch_start_time

            topk_list = [1, 5, 10, 20, args.multi ** args.pre_len]
            k_best_path_right, k_best_traj_right, k_best_d_right, k_mrr = [], [], [], []
            for topk in topk_list:
                preds = all_topk_prob[:, :topk, :]
                gt = torch.unsqueeze(y, dim=1).expand(-1, preds.shape[1], -1)
                # print(f'preds: {preds.shape}')
                equal = torch.eq(gt, preds)
                # print(f'equal: {equal.shape}\n{equal}')
                equal = torch.all(equal, dim=-1)
                # print(f'equal: {equal.shape}\n{equal}')
                cols = torch.argmax(equal * 1, dim=1) + 1
                # print(f'cols: {cols.shape}\n{cols}')
                rows = torch.unique(torch.where(equal == True)[0])
                # print(f'rows: {rows.shape}\n{rows}')
                mrr = torch.sum(1 / cols[rows])

                ele_right = torch.sum((gt == preds) * 1, dim=-1)
                # (batch_size), find best in topk
                batch_right = torch.max(ele_right, dim=-1)[0]
                best_traj_right = torch.sum((batch_right == model.batch_long) * 1)
                best_path_right = torch.sum(batch_right)
                k_best_path_right.append(best_path_right.item())
                k_best_traj_right.append(best_traj_right.item())
                k_mrr.append(mrr.item())

            total += args.batch_size * args.pre_len
            traj_total += args.batch_size
            for k in range(len(topk_list)):
                right[k] += k_best_path_right[k]
                traj_right[k] += k_best_traj_right[k]
                all_mrr[k] += k_mrr[k]

        # print(all_time / traj_num * 1e4 * 1e3)
        all_time_list.append(all_time / traj_num * 1e4 * 1e3)
    # print(f'mean: {np.mean(all_time_list)}, std: {np.std(all_time_list)}')
    # assert 1==2

    right = np.array(right)
    traj_right = np.array(traj_right)
    all_mrr = np.array(all_mrr)

    right, traj_right, all_mrr = right / total, traj_right / traj_total, all_mrr / traj_total

    print(f'T: [{right[0]:.3f}, {right[1]:.3f}, {right[2]:.3f}, {right[3]:.3f}, {right[4]:.3f}] '
          f'[{traj_right[0]:.3f}, {traj_right[1]:.3f}, {traj_right[2]:.3f}, {traj_right[3]:.3f}, {traj_right[4]:.3f}], '
          f'm: [{all_mrr[0]:.3f}, {all_mrr[1]:.3f}, {all_mrr[2]:.3f}, {all_mrr[3]:.3f}, {all_mrr[4]:.3f}] | Time: {time.time()-start_time:.2f}')