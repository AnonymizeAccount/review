import os
import copy
import time
import torch
import numpy as np
from .funcs import new_compute_acc, compute_acc, compute_recall_mrr


def eval_one_epoch(loaders, args, model, type=None, epoch=None, inf_time_list=None):
    total, right, traj_total, d_total, d_right, traj_right, goal_d_total, goal_d_correct, mrr = 0, 0, 0, 0, 1, 0, 0, 0, 0
    if type == 'topk':
        right, traj_right, mrr = [0 for _ in range(5)], [0 for _ in range(5)], [0 for _ in range(5)]
        inference_time, num_requests, model_time, nor_time, multi_time = 0, 0, 0, 0, 0

    model.eval()
    with torch.no_grad():
        for loader in loaders:
            for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
                if type == 'topk' and args.time:
                    num_requests += x.shape[0]
                x = torch.LongTensor(x).to(args.device)
                y = torch.LongTensor(y).to(args.device)
                # if epoch < 100:
                goal = y[:, -1]
                direction_x = torch.LongTensor(direction_x).to(args.device)
                direction_y = torch.LongTensor(direction_y).to(args.device)

                if type == 'topk' and args.time:
                    inference_start = time.time()

                if args.model in ['rnn', 'gru', 'lstm', 'nettraj', 'mlp']:
                    if type == 'topk' and args.time:
                    #     inference_batch_start = time.time()
                        model_time_start = time.time()
                    pred, pred_d, _, direction_correct = model(x, direction_x, length, goal)

                    if type == 'topk' and args.time:
                        model_time += time.time()-model_time_start
                    # if type == 'topk':
                    #     inference_time += time.time() - inference_batch_start
                    batch_mrr, batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, preds_topk, end_time, b_multi_time = \
                        new_compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                        model.padding_loglikelihood, model.batch_long, obs=x[:, -1] - 1, type=type)
                else:
                    if type == 'topk' and args.time:
                        nor_time_start = time.time()
                    model.normalizeEmbedding()
                    if type == 'topk' and args.time:
                        nor_time += time.time() - nor_time_start

                    if type == 'topk' and args.time:
                        model_time_start = time.time()
                    pred, pred_d, _, direction_correct, _ = model(x, direction_x, length, goal)
                    if type == 'topk' and args.time:
                        model_time += time.time()-model_time_start
                    # if type == 'topk':
                    #     inference_time += time.time() - inference_batch_start
                    batch_mrr, batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, _, preds_topk, end_time, rank_time, b_multi_time = \
                        compute_recall_mrr(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                           model.padding_loglikelihood, model.batch_long, obs=x[:, -1] - 1, type=type, model=model)
                if type == 'topk' and args.time:
                    inference_time += end_time - inference_start
                    multi_time += b_multi_time

                total += batch_total
                traj_total += batch_traj_total
                goal_d_total += args.batch_size
                goal_d_correct += direction_correct.item()

                if type != 'topk':
                    right += batch_right
                    traj_right += batch_traj_right
                    mrr += batch_mrr
                else:
                    for k in range(len(batch_right)):
                        right[k] += batch_right[k]
                        traj_right[k] += batch_traj_right[k]
                        mrr[k] += batch_mrr[k]
    if type == 'topk':
        right = np.array(right)
        traj_right = np.array(traj_right)
        mrr = np.array(mrr)

    if type == 'topk' and args.time:
        # inference_time = time.time() - inference_start
        # inference_time = end_time - inference_start
        time_per_1krequests = inference_time / num_requests * 1e4 * 1e3
        model_time = model_time / num_requests * 1e4 * 1e3
        multi_time = multi_time / num_requests * 1e7
        if args.model in ['rnn', 'gru', 'lstm', 'nettraj', 'mlp']:
            print(f'num_requests: {num_requests} | multi_time: {multi_time:.3f} | model_time: {model_time:.3f}ms | time: {time_per_1krequests:.3f}ms')
        else:
            rank_time = rank_time / num_requests * 1e4 * 1e3
            nor_time = nor_time / num_requests * 1e7
            print(f'num_requests: {num_requests} | multi_time: {multi_time:.3f} | nor_time: {nor_time:.3f}ms | model_time: {model_time:.3f}ms | rank_time: {rank_time:.3f}ms | time: {time_per_1krequests:.3f}ms')

        inf_time_list.append(time_per_1krequests)
        if epoch >= 20:
            print(f'time: mean: {np.mean(inf_time_list)} | std: {np.std(inf_time_list)}')
            assert 1==2

    return right / total, traj_right / traj_total, d_right / d_total, goal_d_correct / goal_d_total, mrr / goal_d_total


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

            if args.model in ['rnn', 'gru', 'lstm', 'nettraj', 'mlp']:
                pred, pred_d, loss_kg, direction_correct = model(x, direction_x, length, goal, type='train', epoch=epoch, y=y)
                loss = model.compute_loss(pred, y, pred_d, direction_y) + loss_kg
                batch_mrr, batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, preds_topk, _, _ = \
                    new_compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                    model.padding_loglikelihood, model.batch_long, obs=x[:, -1]-1)
            else:
                model.normalizeEmbedding()
                pred, pred_d, loss_kg, direction_correct, mlp_out = model(x, direction_x, length, goal, type='train', epoch=epoch, y=y)
                loss = model.compute_loss(pred, y, pred_d, direction_y) + loss_kg + model.compute_loss(mlp_out, y, pred_d, direction_y)
                batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, loss_ranking, preds_topk = \
                    compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
                                model.padding_loglikelihood, model.batch_long, obs=x[:, -1] - 1, model=model)
                loss += loss_ranking

            # batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, preds_topk = \
            #     new_compute_acc(pred, pred_d, y, direction_y, args, model.graph_edges, model.node_adj_edges, model.ids, model.offset,
            #                     model.padding_loglikelihood, model.batch_long, obs=x[:, -1]-1)
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