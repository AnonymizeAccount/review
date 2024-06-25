import os
import copy
import time
import torch
import numpy as np
from .funcs import new_compute_acc, compute_acc


def eval_one_epoch(loaders, args, model, type=None, epoch=None):
    total, right, traj_total, d_total, d_right, traj_right, goal_d_total, goal_d_correct = 0, 0, 0, 0, 1, 0, 0, 0
    if type == 'topk':
        right, traj_right = [0 for _ in range(5)], [0 for _ in range(5)]
    model.eval()
    with torch.no_grad():
        for loader in loaders:
            for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
                x = torch.LongTensor(x).to(args.device)
                y = torch.LongTensor(y).to(args.device)
                # if epoch < 100:
                goal = y[:, -1]
                direction_x = torch.LongTensor(direction_x).to(args.device)
                direction_y = torch.LongTensor(direction_y).to(args.device)

                if args.model in ['rnn', 'gru', 'lstm', ]:
                    pred, pred_d, _, direction_correct = model(x, direction_x, length, goal)
                    batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, preds_topk = \
                        new_compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                        model.padding_loglikelihood, model.batch_long, obs=x[:, -1] - 1, type=type)
                else:
                    model.normalizeEmbedding()
                    pred, pred_d, _, direction_correct = model(x, direction_x, length, goal)
                    batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, _, preds_topk = \
                        compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                    model.padding_loglikelihood, model.batch_long, obs=x[:, -1] - 1, type=type, model=model)

                total += batch_total
                traj_total += batch_traj_total
                goal_d_total += args.batch_size
                goal_d_correct += direction_correct.item()

                if type != 'topk':
                    right += batch_right
                    traj_right += batch_traj_right
                else:
                    for k in range(len(batch_right)):
                        right[k] += batch_right[k]
                        traj_right[k] += batch_traj_right[k]

    if type == 'topk':
        right = np.array(right)
        traj_right = np.array(traj_right)

    return right / total, traj_right / traj_total, d_right / d_total, goal_d_correct / goal_d_total


def train_one_epoch(loaders, args, model, optimizer, epoch=None):
    loss_all, total, right, traj_total, d_total, d_right, traj_right, goal_d_total, goal_d_correct = [], 0, 0, 0, 0, 1, 0, 0, 0

    model.train()
    for loader in loaders:
        for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
            x = torch.LongTensor(x).to(args.device)
            y = torch.LongTensor(y).to(args.device)
            # if epoch < 100:
            goal = y[:, -1]
            optimizer.zero_grad()

            direction_x = torch.LongTensor(direction_x).to(args.device)
            direction_y = torch.LongTensor(direction_y).to(args.device)

            if args.model in ['rnn', 'gru', 'lstm', ]:
                pred, pred_d, loss_kg, direction_correct = model(x, direction_x, length, goal, type='train', epoch=epoch, y=y)
                loss = model.compute_loss(pred, y, pred_d, direction_y) + loss_kg
                batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, preds_topk = \
                    new_compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                    model.padding_loglikelihood, model.batch_long, obs=x[:, -1]-1)
            else:
                model.normalizeEmbedding()
                pred, pred_d, loss_kg, direction_correct = model(x, direction_x, length, goal, type='train', epoch=epoch, y=y)
                loss = model.compute_loss(pred, y, pred_d, direction_y) + loss_kg
                batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, loss_ranking, preds_topk = \
                    compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                model.padding_loglikelihood, model.batch_long, obs=x[:, -1] - 1, model=model)
                loss += loss_ranking

            # # batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, preds_topk = \
            # #     new_compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
            # #                     model.padding_loglikelihood, model.batch_long, obs=x[:, -1]-1)
            #
            # batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, loss_ranking, preds_topk = \
            #     compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
            #                 model.padding_loglikelihood, model.batch_long, obs=x[:, -1]-1, model=model)
            # loss += loss_ranking

            if not args.rand:
                loss.backward()
                optimizer.step()
                loss_all.append(loss.item())
            total += batch_total
            right += batch_right
            traj_total += batch_traj_total
            traj_right += batch_traj_right
            goal_d_total += args.batch_size
            goal_d_correct += direction_correct.item()

    return np.mean(loss_all), right / total, traj_right / traj_total, d_right / d_total, goal_d_correct / goal_d_total