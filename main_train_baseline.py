import argparse
import random
import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn

import baselines.MAE.models_mae
from tqdm import tqdm

from baselines.MAE import models_vit, models_mae
from data.datasets import DownStreamDataset
from utils.io_utils import load_access_street_view, get_images, load_task_data, calc


class Linear(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.project = nn.Linear(embed_dim, 2)

    def forward(self, image_latent):
        temp = torch.max(image_latent, 1)[0]
        logits = self.project(temp)
        return logits.squeeze(1)


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = models_vit.__dict__['vit_large_patch16'](
        num_classes=1,
        drop_path_rate=0.1,
        global_pool=True
    )
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    model = model.cuda()

    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False

    return model


def train(model, criterion, optimizer, loader, args, epoch):
    device = torch.device(args.gpu)
    model.train()
    all_predictions = []
    all_truths = []
    total_loss = 0.0
    for images, y in loader:
        images = images.to(device=device)
        y = y.to(device=device).float()

        optimizer.zero_grad()

        predicts = model(images)

        # print(predicts.shape, y.shape)
        loss = criterion(predicts, y)
        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        all_predictions.extend(predicts.cpu().detach().numpy())
        all_truths.extend(y.cpu().detach().numpy())

    return calc("Train", epoch, all_predictions, all_truths, total_loss / len(loader))


def evaluate(model, loader, args, epoch):
    device = torch.device(args.gpu)
    model.eval()

    all_y, all_predicts = [], []
    with torch.no_grad():
        for images, y in loader:
            images = images.to(device=device)

            y = y.to(device=device).float()

            predicts = model(images)

            all_y.append(y.cpu().numpy())
            all_predicts.append(predicts.cpu().numpy())
    all_y = np.concatenate(all_y)
    all_predicts = np.concatenate(all_predicts)
    return calc("Eval", epoch, all_predicts, all_y, None)


def main(args):
    # pip install timm==0.3.2
    # todo: change timm to 1.0.7
    os.makedirs(f"./log/{args.model}", exist_ok=True)
    checkpoints_dir = f"./baselines/{args.model}/checkpoints/{args.save_name}.pt"
    os.makedirs(f"./baselines/{args.model}/checkpoints/", exist_ok=True)

    ava_indexs = load_access_street_view(args.city)

    task_data = load_task_data(args.city)

    image_dataset = []
    for index in tqdm(ava_indexs):
        street_views, images = get_images(index, args.city)
        image_dataset.append([images, [task_data[0][int(index)][-1], task_data[1][int(index)][-1]]])

    # load model
    chkpt_dir = 'baselines/MAE/mae_pretrain_vit_large.pth'
    model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')

    # split the dataset into train and test
    train_size = int(0.8 * len(image_dataset))
    test_size = len(image_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])
    train_dataset = DownStreamDataset(train_dataset, model)
    val_dataset = DownStreamDataset(val_dataset, model, train_dataset.mean, train_dataset.std)

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = Linear(model.embed_dim).to(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = 123456789

    for epoch in range(args.epoch):
        train(model, criterion, optimizer, train_loader, args, epoch)
        cur_metrics = evaluate(model, train_loader, args, epoch)
        # evaluate(model, test_loader, args, "test")
        checkpoint_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if cur_metrics['mse'] < best_val:
            torch.save(
                checkpoint_dict,
                checkpoints_dir,
            )
            best_val = cur_metrics['mse']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save_name",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="MAE",
        help="model name",
    )

    parser.add_argument(
        "--city",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],  # ["New York City", "San Francisco", "Washington", "Chicago"]
        help=""
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch_size",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100000,
        help="num epochs",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda:0",
        help="device",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default="1e-3",
        help="lr",
    )

    args = parser.parse_args()

    main(args)

'''


# load model
# run on an image

'''
