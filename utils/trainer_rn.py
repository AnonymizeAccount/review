import os
import copy
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
# from utils.trainer_epoch import train_one_epoch, eval_one_epoch
from .kg_epoch_rn import train_one_epoch, eval_one_epoch


def train(loader, args, model, optimizer):
    train_loaders, val_loaders, test_loaders = loader
    best_acc, best_traj_acc, best_d_acc, best_goal_d_acc, count_patience, state, dur = -1, -1, -1, 0, 0, None, []
    loss, train_acc, train_traj_acc, train_d_acc, val_acc, val_traj_acc, val_d_acc, test_acc, test_traj_acc, test_d_acc, inf_time_list = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []
    for ep in range(args.epoch):
        if count_patience > args.patience:
            break
        start_time = time.time()
        # if not args.rand:
        loss, train_acc, train_traj_acc, train_d_acc, train_goal_d_acc = train_one_epoch(train_loaders, args, model, optimizer, epoch=ep)
        val_acc, val_traj_acc, val_d_acc, val_goal_d_acc, val_mrr = eval_one_epoch(val_loaders, args, model, epoch=ep)
        # test_acc, test_traj_acc, test_d_acc, test_goal_d_acc = eval_one_epoch(test_loaders, args, model, type='test', epoch=ep)
        test_acc, test_traj_acc, _, test_goal_d_acc, test_mrr = eval_one_epoch(test_loaders, args, model, type='topk', epoch=ep,
                                                                               inf_time_list=inf_time_list)
        dur.append(time.time() - start_time)

        print(f'Ep {ep + 1} | p:{count_patience} | Loss:{loss:.2f} | T: {train_acc:.3f}, {train_traj_acc:.3f} | '
              f'V: {val_acc:.3f}, {val_traj_acc:.3f}, m: {val_mrr:.3f} | '
              f'T: [{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}] '
              f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}], '
              f'm: [{test_mrr[0]:.3f}, {test_mrr[1]:.3f}, {test_mrr[2]:.3f}, {test_mrr[3]:.3f}, {test_mrr[4]:.3f}] | '
              f'Time: {dur[-1]:.2f}')

        # better = np.sum(np.array([val_traj_acc >= best_traj_acc, val_acc >= best_acc, val_d_acc >= best_d_acc]) * 1)
        better = np.sum(np.array([val_traj_acc > best_traj_acc, val_acc > best_acc]) * 1)

        if better >= 1:
            best_acc = val_acc
            best_traj_acc = val_traj_acc
            best_d_acc = val_d_acc
            best_goal_d_acc = val_goal_d_acc
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('optim', copy.deepcopy(optimizer.state_dict()))])
            count_patience = 0
        else:
            count_patience += 1

    model.load_state_dict(state['model'])
    test_acc, test_traj_acc, _, _, test_mrr = eval_one_epoch(test_loaders, args, model, type='topk', epoch=99999)
    print(f'Final test result: [{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}], '
          f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}], '
          f'm: [{test_mrr[0]:.3f}, {test_mrr[1]:.3f}, {test_mrr[2]:.3f}, {test_mrr[3]:.3f}, {test_mrr[4]:.3f}]')

    # fo = open("./log/0315/large_batch_margin.txt", "a")
    # fo.write(f'Margin: {args.margin}\n'
    #          f'Final test result: [{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}], '
    #          f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}]\n\n')
    # fo.close()

    if args.save:
        if not os.path.exists(os.path.join(args.base_dir, f'checkpoints', f'{args.dataset}')):
            os.mkdir(os.path.join(args.base_dir, f'checkpoints', f'{args.dataset}'))
        torch.save(state, os.path.join(args.base_dir, f'checkpoints', f'{args.dataset}',
                                       f'{args.days}_{args.multi}_{args.seq_len}_{args.pre_len}_{args.batch_size}.pkl'))