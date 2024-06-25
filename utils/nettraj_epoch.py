import os
import copy
import time
import torch
import numpy as np


def train(loader, args, model, optimizer):
    train_loaders, val_loaders, test_loaders = loader

    for ep in range(args.epoch):
        train_path, train_traj, train_path_total, train_traj_total = [], [], 0, 0
        val_path, val_traj, val_path_total, val_traj_total = [], [], 0, 0
        model.train()
        for loader in train_loaders:
            for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
                x = torch.LongTensor(x).to(args.device)
                y = torch.LongTensor(y).to(args.device)
                direction_x = torch.LongTensor(direction_x).to(args.device)
                direction_y = torch.LongTensor(direction_y).to(args.device)
                pred = model(x, direction_x, length, y=y)
                optimizer.zero_grad()
                loss = model.compute_loss(pred, direction_y, x[:, -1], y)
                loss.backward()
                optimizer.step()

                best_traj_right, best_path_right = model.compute_metrics(pred, y, direction_y, x[:, -1])
                train_traj.append(best_traj_right)
                train_path.append(best_path_right)
                train_traj_total += args.batch_size
                train_path_total += args.batch_size * args.pre_len

        model.eval()
        with torch.no_grad():
            for loader in test_loaders:
                for i, (x, y, direction_x, direction_y, length) in enumerate(loader.get_iterator()):
                    x = torch.LongTensor(x).to(args.device)
                    y = torch.LongTensor(y).to(args.device)
                    direction_x = torch.LongTensor(direction_x).to(args.device)
                    direction_y = torch.LongTensor(direction_y).to(args.device)
                    pred = model(x, direction_x, length, y=y)
                    best_traj_right, best_path_right = model.compute_metrics(pred, y, direction_y, x[:, -1])

                    val_traj.append(best_traj_right)
                    val_path.append(best_path_right)
                    val_traj_total += args.batch_size
                    val_path_total += args.batch_size * args.pre_len

        print(f'Epoch: {ep+1} | Train: [{np.sum(train_path) / train_path_total:.3f}, {np.sum(train_traj) / train_traj_total:.3f}] | '
              f'Val: [{np.sum(val_path) / val_path_total:.3f}, {np.sum(val_traj) / val_traj_total:.3f}]')
