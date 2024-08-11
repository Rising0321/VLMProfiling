import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# encoding: utf-8
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from data.datasets import DownStreamDataset, WalkDataset
from utils.io_utils import load_access_street_view, load_task_data, calc, init_seed, init_logging, \
    get_graph_and_images_dual, shrink_graph
from DualVLMWalk import ask_start_image, ask_middle_image, get_model, ask_summary_image
import concurrent.futures

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


def output_words(city, index, text, model_name="minicpm"):
    file_name = f"./log/supervised/{city}/{index}.md"
    os.makedirs(f"./log/supervised/{city}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(text)
        file.write("\n")


def output_images(city, index, image, model_name="minicpm"):
    file_name = f"./log/supervised/{city}/{index}_image.md"
    os.makedirs(f"./log/supervised/{city}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(str(image))
        file.write("\n")


def random_walk(g, start_point, args, index, llm, tokenizer, images, city):
    output_words(city, index, "# Experimental Result")
    output_words(city, index, "##  Start Edge Caption")

    id = int(g.nodes[start_point]["image"])
    image = images[id]
    output_words(city, index, f"![no_name](../../data/{city}/{index}/squeeze_images/{id}.jpg)")
    output_images(city, index, id)

    images_emb = []
    summary = ask_start_image(llm, tokenizer, image, "")
    output_words(city, index, summary)

    for i in trange(9):
        try:
            neighbors = list(g.neighbors(start_point))
            best_neighbor = -1
            best_answer = -1
            best_image = -1
            for neighbor in neighbors:
                id = int(g.nodes[neighbor]["image"])
                now_image = images[id]
                answer, reason = ask_middle_image(llm, tokenizer, now_image, summary)

                output_words(city, index,
                             f"![no_name](../../data/{city}/{index}/squeeze_images/{id}.jpg)")
                output_words(city, index, reason)

                if answer > best_answer:
                    best_answer = answer
                    best_neighbor = neighbor
                    best_image = now_image
                    output_words(city, index, "best answer updated!!!!!!!!!!!!!!!!")

            start_point = best_neighbor

            output_words(city, index, f"###  update summary")
            summary = ask_summary_image(llm, tokenizer, best_image, summary)
            output_words(city, index, summary)

            output_images(city, index, int(g.nodes[best_neighbor]["image"]))
        except Exception as e:
            print(e)
            continue
    # exit(0)
    return torch.tensor(images_emb).float()


def evaluate(model, loader, args, llm, tokenizer, city):
    device = torch.device(args.gpu)
    # model.eval()

    all_y = []
    all_predicts = []
    all_city = []

    for index, y, s in tqdm(loader):
        index = int(index[0])

        sub_g, street_views, images = get_graph_and_images_dual(index, args.city_size, args.target)

        new_g, start_point = shrink_graph(sub_g)

        random_walk(new_g, start_point, args, index, llm, tokenizer, images, city)
        continue

    return
    #     image_emb = random_walk(new_g, start_point, args, index, llm, tokenizer, images)
    #     image_emb = torch.cat((image_emb, s), dim=0)
    #     image_emb = image_emb.to(device)
    #     image_latent = model(image_emb)
    #
    #     all_predicts.append(float(image_latent.cpu().detach()))
    #     all_y.append(float(y.cpu().detach()))
    #     all_city.append(args.city_size)
    #
    # return calc("Eval", 233, all_predicts, all_y, all_city, None, args.city_size, args.target)


city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


# run_logs = "./run_logs/city/"
# zero_logs = "./zero_logs/city/target/"

def main(args):
    # pip install timm==0.3.2
    # todo: change timm to 1.0.7

    init_seed(args.seed)

    idx_dataset = []

    city = args.city_size

    # todo: for test, can repeat the following code with
    ava_indexs = load_access_street_view(city)

    task_data = load_task_data(city, args.target)

    llm, tokenizer = get_model(args)

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

    if len(test_dataset) > 100:
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [100, len(test_dataset) - 100])

    train_dataset = WalkDataset(train_dataset, args.target)
    val_dataset = WalkDataset(val_dataset, args.target, train_dataset.mean, train_dataset.std)
    test_dataset = WalkDataset(test_dataset, args.target, train_dataset.mean, train_dataset.std)

    # data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model = Linear(embed_dims[args.model], type).to(args.gpu)
    # checkpoint = torch.load(checkpoints_dir)
    # model.load_state_dict(checkpoint['state_dict'])
    model = None

    evaluate(model, test_loader, args, llm, tokenizer, city_names[city])


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
