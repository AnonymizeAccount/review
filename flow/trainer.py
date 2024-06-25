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
from .test import train_one_epoch, eval_one_epoch


# def train(loader, args, MLP, opt_mlp, model, opt_model):
def train(loader, args, model, opt_model):
    train_loaders, val_loaders, test_loaders = loader
    best_d_acc, best_goal_d_acc, count_patience, state, dur = -1, 0, 0, None, []
    loss, train_acc, train_traj_acc, train_d_acc, val_acc, val_traj_acc, val_d_acc, test_acc, test_traj_acc, test_d_acc, inf_time_list = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, []
    best_acc_1, best_traj_acc_1, best_acc_2, best_traj_acc_2, best_acc_3, best_traj_acc_3, best_acc_4, best_traj_acc_4, best_acc_5, best_traj_acc_5 = \
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

    # distill_criterion = torch.nn.MSELoss()

    for ep in range(args.epoch):
        if count_patience > args.patience:
            break
        start_time = time.time()

        # loss, train_acc, train_traj_acc, train_d_acc, train_goal_d_acc = \
        #     train_one_epoch(train_loaders, args, MLP, opt_mlp, model, opt_model, distill_criterion=distill_criterion, epoch=ep)
        # val_acc, val_traj_acc, val_d_acc, val_goal_d_acc, val_mrr = \
        #     eval_one_epoch(val_loaders, args, MLP, opt_mlp, model, opt_model, epoch=ep, type='topk')
        # test_acc, test_traj_acc, _, test_goal_d_acc, test_mrr = \
        #     eval_one_epoch(test_loaders, args, MLP, opt_mlp, model, opt_model, type='topk', epoch=ep, inf_time_list=inf_time_list)

        loss, train_acc, train_traj_acc, train_d_acc, train_goal_d_acc = \
            train_one_epoch(train_loaders, args, model, opt_model)
        val_acc, val_traj_acc, val_d_acc, val_goal_d_acc, val_mrr, _ = \
            eval_one_epoch(val_loaders, args, model, opt_model, epoch=ep, type='topk')
        test_acc, test_traj_acc, _, test_goal_d_acc, test_mrr, _ = \
            eval_one_epoch(test_loaders, args, model, opt_model, type='topk', epoch=ep, inf_time_list=inf_time_list)

        dur.append(time.time() - start_time)

        # print(f'Ep {ep + 1} | p:{count_patience} | Loss:{loss:.2f} | T: {train_acc:.3f}, {train_traj_acc:.3f} | '
        #       f'V: {val_acc[0]:.3f}, {val_traj_acc[0]:.3f}, m: {val_mrr[0]:.3f} | '
        #       f'T: [{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}] '
        #       f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}], '
        #       f'm: [{test_mrr[0]:.3f}, {test_mrr[1]:.3f}, {test_mrr[2]:.3f}, {test_mrr[3]:.3f}, {test_mrr[4]:.3f}] | '
        #       f'Time: {dur[-1]:.2f}')

        print(f'Ep {ep + 1} | p:{count_patience} | Loss:{loss:.2f} | T: {train_acc:.3f}, {train_traj_acc:.3f} | '
              f'V: [{val_acc[0]:.3f}, {val_acc[1]:.3f}, {val_acc[2]:.3f}, {val_acc[3]:.3f}, {val_acc[4]:.3f}] '
              f'[{val_traj_acc[0]:.3f}, {val_traj_acc[1]:.3f}, {val_traj_acc[2]:.3f}, {val_traj_acc[3]:.3f}, {val_traj_acc[4]:.3f}] | '
              f'T: [{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}] '
              f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}], '
              f'm: [{test_mrr[0]:.3f}, {test_mrr[1]:.3f}, {test_mrr[2]:.3f}, {test_mrr[3]:.3f}, {test_mrr[4]:.3f}] | '
              f'Time: {dur[-1]:.2f}')

        better = np.sum(np.array([val_traj_acc[0] > best_traj_acc_1, val_acc[0] > best_acc_1,
                                  val_traj_acc[1] > best_traj_acc_2, val_acc[1] > best_acc_2,
                                  val_traj_acc[2] > best_traj_acc_3, val_acc[2] > best_acc_3,
                                  val_traj_acc[3] > best_traj_acc_4, val_acc[3] > best_acc_4,
                                  val_traj_acc[4] > best_traj_acc_5, val_acc[4] > best_acc_5]) * 1)

        if better >= 2:
            if val_acc[0] > best_acc_1:
                best_acc_1 = val_acc[0]
            if val_traj_acc[0] > best_traj_acc_1:
                best_traj_acc_1 = val_traj_acc[0]
            if val_acc[1] > best_acc_2:
                best_acc_2 = val_acc[1]
            if val_traj_acc[1] > best_traj_acc_2:
                best_traj_acc_2 = val_traj_acc[1]
            if val_acc[2] > best_acc_3:
                best_acc_3 = val_acc[2]
            if val_traj_acc[2] > best_traj_acc_3:
                best_traj_acc_3 = val_traj_acc[2]
            if val_acc[3] > best_acc_4:
                best_acc_4 = val_acc[3]
            if val_traj_acc[3] > best_traj_acc_4:
                best_traj_acc_4 = val_traj_acc[3]
            if val_acc[4] > best_acc_5:
                best_acc_5 = val_acc[4]
            if val_traj_acc[4] > best_traj_acc_5:
                best_traj_acc_5 = val_traj_acc[4]


            # state = dict([('model', copy.deepcopy(model.state_dict())),
            #               ('mlp', copy.deepcopy(MLP.state_dict())),
            #               ('opt_mlp', copy.deepcopy(opt_mlp.state_dict())),
            #               ('opt_model', copy.deepcopy(opt_model.state_dict()))])
            state = dict([('model', copy.deepcopy(model.state_dict())),
                          ('opt_model', copy.deepcopy(opt_model.state_dict()))])
            count_patience = 0
        else:
            count_patience += 1

    model.load_state_dict(state['model'])
    args.state = state
    # MLP.load_state_dict(state['mlp'])
    # test_acc, test_traj_acc, _, _, test_mrr = eval_one_epoch(test_loaders, args, MLP, opt_mlp, model, opt_model, type='topk', epoch=99999)
    test_acc, test_traj_acc, _, _, test_mrr, _ = eval_one_epoch(test_loaders, args, model, opt_model, type='topk', epoch=99999)
    # print(f'Final test result: [{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}], '
    #       f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}], '
    #       f'm: [{test_mrr[0]:.3f}, {test_mrr[1]:.3f}, {test_mrr[2]:.3f}, {test_mrr[3]:.3f}, {test_mrr[4]:.3f}]')

    print(f'a: {args.a} | b: {args.b} | c: {args.c} | d: {args.d} | e: {args.e} | f: {args.f} | g: {args.g} | '
          f'[{test_acc[0]:.3f}, {test_acc[1]:.3f}, {test_acc[2]:.3f}, {test_acc[3]:.3f}, {test_acc[4]:.3f}], '
          f'[{test_traj_acc[0]:.3f}, {test_traj_acc[1]:.3f}, {test_traj_acc[2]:.3f}, {test_traj_acc[3]:.3f}, {test_traj_acc[4]:.3f}], '
          f'm: [{test_mrr[0]:.3f}, {test_mrr[1]:.3f}, {test_mrr[2]:.3f}, {test_mrr[3]:.3f}, {test_mrr[4]:.3f}]')