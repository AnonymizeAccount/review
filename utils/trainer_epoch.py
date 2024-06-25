import os
import copy
import time
import torch
import numpy as np


def eval_one_epoch(loaders, args, model, type=None, epoch=None):
    total, right, traj_total, d_total, d_right, traj_right = 0, 0, 0, 0, 1, 0
    model.eval()
    if type == 'topk':
        num_requests = 0
        inference_start = time.time()
    with torch.no_grad():
        for loader in loaders:
            for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
                if type == 'topk':
                    num_requests += x.shape[0]
                x = torch.LongTensor(x).to(args.device)
                y = torch.LongTensor(y).to(args.device)

                if args.model in ['nettraj']:
                    direction_x = torch.LongTensor(direction_x).to(args.device)
                    direction_y = torch.LongTensor(direction_y).to(args.device)
                    pred = model(x, direction_x, length)
                    batch_total, batch_right, batch_traj_total, batch_traj_right, _ = model.new_compute_acc(pred, y, direction_y)
                elif args.model in ['rnn', 'gru', 'lstm',] or args.wo_d is False:
                    if args.wo_d is False:
                        direction_x = torch.LongTensor(direction_x).to(args.device)
                        direction_y = torch.LongTensor(direction_y).to(args.device)
                        pred, pred_d = model(x, direction_x, length)
                        batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, _, _ = \
                            model.new_compute_acc(pred, pred_d, y, direction_y, obs=x[:, -1]-1, type=type, epoch=epoch)
                    else:
                        pred = model(x, length)
                        batch_total, batch_right, batch_traj_total, batch_traj_right, _ = model.new_compute_acc(pred, y)
                else:
                    raise ValueError

                total += batch_total
                right += batch_right
                traj_total += batch_traj_total
                traj_right += batch_traj_right

    if type == 'topk':
        inference_time = time.time() - inference_start
        time_per_1krequests = inference_time / num_requests * 1e4
        print(f'num_requests: {num_requests} | time: {time_per_1krequests:.3f}s')
        assert 1==2

    return right / total, traj_right / traj_total, d_right / d_total, 0


def train_one_epoch(loaders, args, model, optimizer, epoch=None):
    loss_all, total, right, traj_total, d_total, d_right, traj_right = [], 0, 0, 0, 0, 1, 0

    model.train()
    for loader in loaders:
        for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
            x = torch.LongTensor(x).to(args.device)
            y = torch.LongTensor(y).to(args.device)
            optimizer.zero_grad()
            # batch_total, batch_right, batch_traj_total, batch_traj_right, _ = model.compute_acc(pred, y)
            # print(f'x: {x.shape}, y: {y.shape}')
            if args.model in ['nettraj']:
                direction_x = torch.LongTensor(direction_x).to(args.device)
                direction_y = torch.LongTensor(direction_y).to(args.device)
                pred = model(x, direction_x, length)
                batch_total, batch_right, batch_traj_total, batch_traj_right, _ = model.new_compute_acc(pred, y, direction_y)
                loss = model.compute_loss(pred, direction_y)
            elif args.model in ['rnn', 'gru', 'lstm',] or args.wo_d is False:
                if args.wo_d is False:
                    direction_x = torch.LongTensor(direction_x).to(args.device)
                    direction_y = torch.LongTensor(direction_y).to(args.device)
                    pred, pred_d = model(x, direction_x, length)
                    batch_total, batch_right, batch_traj_total, batch_traj_right, d_total, d_right, _, _ = \
                        model.new_compute_acc(pred, pred_d, y, direction_y, obs=x[:, -1]-1)
                    loss = model.compute_loss(pred, y, pred_d, direction_y)
                else:
                    pred = model(x, length)
                    batch_total, batch_right, batch_traj_total, batch_traj_right, _ = model.new_compute_acc(pred, y)
                    loss = model.compute_loss(pred, y)
            else:
                raise ValueError

            loss.backward()
            optimizer.step()
            loss_all.append(loss.item())
            total += batch_total
            right += batch_right
            traj_total += batch_traj_total
            traj_right += batch_traj_right

    return np.mean(loss_all), right / total, traj_right / traj_total, d_right / d_total, 0

