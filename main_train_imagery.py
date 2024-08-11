# encoding: utf-8
from utils.io_utils import load_access_street_view, get_images, load_task_data, calc, init_seed, get_imagery, \
    init_logging

import argparse
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from baselines.MAE import models_vit
from data.datasets import DownStreamDataset, ImageryDataset
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import ViTImageProcessor, ViTModel

from simclr import SimCLR
from simclr.modules import get_resnet
import open_clip

embed_dims = {
    'MAE': 1024,
    "ResNet": 2048,
    'SimCLR': 2048,
    'CLIP': 768,
    "ViT": 768
}


class Linear(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.project = nn.Linear(embed_dim, 1)
        self.act = nn.GELU()

    def forward(self, image_latent):
        logits = self.project(image_latent)
        return logits.squeeze(1)


def prepare_model(args):
    # build model
    model = None
    processor = None

    if args.model.startswith('MAE'):
        chkpt_dir = 'baselines/MAE/mae_pretrain_vit_large.pth'
        model = models_vit.__dict__['vit_large_patch16'](
            num_classes=1,
            drop_path_rate=0.1,
            global_pool=True
        )
        # load model
        checkpoint = torch.load(chkpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
        model = model.to(args.gpu)
        processor = None
    elif args.model.startswith('ResNet'):
        chkpt_dir = 'baselines/ResNet/'
        processor = AutoImageProcessor.from_pretrained(chkpt_dir)
        model = ResNetForImageClassification.from_pretrained(chkpt_dir)
        model.classifier = torch.nn.Sequential()  # to remove classifier
    elif args.model.startswith('SimCLR'):
        chkpt_dir = 'baselines/SimCLR/checkpoint_100.tar'
        encoder = get_resnet('resnet50', pretrained=False)  # don't load a pre-trained model from PyTorch repo
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        model = SimCLR(encoder=encoder, projection_dim=64, n_features=n_features)
        model.load_state_dict(torch.load(chkpt_dir, map_location=torch.device(args.gpu)))
        model = model.to(args.gpu)
        processor = None
    elif args.model.startswith('CLIP'):
        chkpt_dir = 'baselines/CLIP/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/open_clip_pytorch_model.bin'
        model, _, transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained=chkpt_dir
        )
        model.to(args.gpu)
        processor = transform
    elif args.model.startswith('ViT'):
        chkpt_dir = 'baselines/ViT'
        processor = ViTImageProcessor.from_pretrained(chkpt_dir)
        model = ViTModel.from_pretrained(chkpt_dir)

    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False

    return model, processor


def train(model, criterion, optimizer, loader, args, epoch, city_size):
    device = torch.device(args.gpu)
    model.train()
    all_predictions = []
    all_truths = []
    all_city = []
    total_loss = 0.0
    for images, y, c in loader:
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
        all_city.extend(c.cpu().detach().numpy())

    return calc("Train", epoch, all_predictions, all_truths, all_city, total_loss / len(loader), city_size, args.target)


def evaluate(model, loader, args, epoch, city_size):
    device = torch.device(args.gpu)
    model.eval()

    all_predicts = []
    all_y = []
    all_city = []
    with torch.no_grad():
        for images, y, c in loader:
            images = images.to(device=device)

            y = y.to(device=device).float()

            predicts = model(images)

            all_y.extend(y.cpu().numpy())
            all_predicts.extend(predicts.cpu().numpy())

            # print(y.shape, predicts.shape, images.shape)

            all_city.extend(c.cpu().numpy())

    return calc("Eval", epoch, all_predicts, all_y, all_city, None, city_size, args.target)


city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


def main(args):
    # pip install timm==0.3.2
    # todo: change timm to 1.0.7

    init_seed(args.seed)

    type = "im"

    init_logging(args, type)

    checkpoints_dir = f"/home/wangb/VLMProfiling/baselines/{args.model}/checkpoints/single-{type}-{args.city_size}-{args.target}.pt"
    os.makedirs(f"/home/wangb/VLMProfiling/baselines/{args.model}/checkpoints/", exist_ok=True)

    image_dataset = []

    city = args.city_size
    # todo: for test, can repeat the following code with

    ava_indexs = load_access_street_view(city)

    model, preprocessor = None, None

    task_data = load_task_data(city, args.target)
    for index in tqdm(ava_indexs):
        sucess_path = f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/success"
        if not os.path.exists(sucess_path):
            continue
        _, images = get_imagery(index, city, args.model, model, preprocessor)
        image_dataset.append([images, task_data[int(index)][-1], city])

    # split the dataset into train and test
    train_size = int(0.7 * len(image_dataset))
    val_size = int(0.8 * len(image_dataset)) - train_size
    test_size = len(image_dataset) - train_size - val_size
    print(train_size, val_size, test_size)

    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(image_dataset, [train_size, val_size, test_size])

    if len(test_dataset) > 100:
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [100, len(test_dataset) - 100])

    train_dataset = ImageryDataset(train_dataset, args.target, model_name=args.model, preprocessor=preprocessor)
    val_dataset = ImageryDataset(val_dataset, args.target, train_dataset.mean, train_dataset.std, model_name=args.model,
                                 preprocessor=preprocessor)
    test_dataset = ImageryDataset(test_dataset, args.target, train_dataset.mean, train_dataset.std,
                                  model_name=args.model,
                                  preprocessor=preprocessor)

    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Linear(embed_dims[args.model]).to(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = -123456789
    best_turn = 0

    for epoch in range(args.epoch):
        train(model, criterion, optimizer, train_loader, args, epoch, args.city_size)
        cur_metrics = evaluate(model, val_loader, args, epoch, args.city_size)
        evaluate(model, test_loader, args, epoch, args.city_size)
        # evaluate(model, test_loader, args, "test")

        if cur_metrics['r2'] > best_val:

            checkpoint_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(
                checkpoint_dict,
                checkpoints_dir,
            )
            best_val = cur_metrics['r2']
            print("best:", best_val)
            best_turn = 0
        else:
            best_turn += 1
            if best_turn > args.patience:
                break
    # load state dict
    checkpoint = torch.load(checkpoints_dir)
    model.load_state_dict(checkpoint['state_dict'])
    evaluate(model, test_loader, args, "test", args.city_size)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        default="CLIP",  # or ResNet
        choices=["MAE", "ResNet", "SimCLR", "CLIP", "ViT"],
        help="model name",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
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
        default=1e-3,
        help="lr",
    )

    # huggingface-cli download --resume-download google/vit-base-patch16-224-in21k --local-dir ./vit-base-patch16-224-in21k
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="patience",
    )

    parser.add_argument(
        "--city_size",
        type=int,
        default=3,
        help="number of cities",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=2,
        help="Carbon or Population or NightLight",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed",
    )

    args = parser.parse_args()

    main(args)

'''


# load model
# run on an image

'''
