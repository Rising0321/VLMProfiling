# encoding: utf-8
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.datasets import DownStreamDataset, WalkDataset
from utils.io_utils import load_access_street_view, load_task_data, calc, init_seed, init_logging, \
    get_graph_and_images_dual, shrink_graph

embed_dims = {
    'MAE': 1024,
    "ResNet": 2048,
    'SimCLR': 2048,
    'CLIP': 768,
    "ViT": 768
}


class Linear(nn.Module):
    def __init__(self, embed_dim, type):
        super().__init__()
        if type == 'sv+im':
            self.project = nn.Linear(embed_dim * 11, 1)
        elif type == 'sv':
            self.project = nn.Linear(embed_dim * 10, 1)

    def forward(self, image_latent):
        # temp = torch.max(image_latent, 0)[0]
        temp = image_latent.view(-1)
        # weight = 1 / 10
        # image_latent = image_latent * weight
        # temp = torch.sum(image_latent, dim=0)
        logits = self.project(temp)
        return logits


def random_walk(g, start_point, args, index):
    images = []
    for i in range(10):
        try:
            embedding_path = f"/home/wangb/OpenVIRL/data/{city_names[args.city_size]}/{index}/image_embeddings/{g.nodes[start_point]['image']}-{args.model}.npy"
            images.append(np.load(embedding_path))
            neighbors = list(g.neighbors(start_point))
            len_nei = len(neighbors)
            start_point = neighbors[random.randint(0, len_nei - 1)]
        except Exception as e:
            print(e)
    return torch.tensor(images).float()


def evaluate(model, loader, args):
    device = torch.device(args.gpu)
    model.eval()

    all_y = []
    all_predicts = []
    all_city = []

    for index, y, s in tqdm(loader):
        index = int(index[0])

        # if index == 983 or index == 979:
        #     continue
        sub_g, street_views, images = get_graph_and_images_dual(index, args.city_size, args.target)

        new_g, start_point = shrink_graph(sub_g)

        images = random_walk(new_g, start_point, args, index)
        images = torch.cat((images, s), dim=0)
        images = images.to(device)
        image_latent = model(images)

        all_predicts.append(float(image_latent.cpu().detach()))
        all_y.append(float(y.cpu().detach()))
        all_city.append(args.city_size)

    return calc("Eval", 233, all_predicts, all_y, all_city, None, args.city_size, args.target)


city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


def main(args):
    # pip install timm==0.3.2
    # todo: change timm to 1.0.7

    init_seed(args.seed)

    type = "sv+im"

    init_logging(args, "random")

    checkpoints_dir = f"./baselines/{args.model}/checkpoints/cat-{type}-{args.city_size}-{args.target}.pt"
    # os.makedirs(checkpoints_dir.replace(f"{args.save_name}.pt", ""), exist_ok=True)

    idx_dataset = []

    city = args.city_size

    # todo: for test, can repeat the following code with
    ava_indexs = load_access_street_view(city)

    task_data = load_task_data(city, args.target)

    for index in tqdm(ava_indexs):
        sucess_path = f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/success"
        if not os.path.exists(sucess_path):
            continue
        satellite = np.load(
            f'/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/satellite_embedding_{args.model}.npy')
        idx_dataset.append([index, task_data[int(index)][-1], satellite])

    # split the dataset into train and test
    train_size = int(0.7 * len(idx_dataset))
    val_size = int(0.8 * len(idx_dataset)) - train_size
    test_size = len(idx_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(idx_dataset, [train_size, val_size, test_size])

    train_dataset = WalkDataset(train_dataset, args.target)
    val_dataset = WalkDataset(val_dataset, args.target, train_dataset.mean, train_dataset.std)
    test_dataset = WalkDataset(test_dataset, args.target, train_dataset.mean, train_dataset.std)

    # data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = Linear(embed_dims[args.model], type).to(args.gpu)
    checkpoint = torch.load(checkpoints_dir)
    model.load_state_dict(checkpoint['state_dict'])

    evaluate(model, test_loader, args)


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
        default="ViT",
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
        default=1,
        help="Carbon or Population",
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
