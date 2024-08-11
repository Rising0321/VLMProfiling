# encoding: utf-8

import os

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

from data.datasets import DownStreamDataset, WalkDataset
from utils.io_utils import load_access_street_view, load_task_data, calc, init_seed, init_logging, \
    get_graph_and_images_dual, shrink_graph, parse_json, get_model
from DualVLMWalk import ask_start_image, ask_middle_image, ask_summary_image
import concurrent.futures


def output_words(city, index, text, model_dir, model_name="minicpm"):
    file_name = f"{model_dir}/supervised/{city}/{index}.md"
    os.makedirs(f"{model_dir}/supervised/{city}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(text)
        file.write("\n")


def output_images(city, index, image, model_dir, model_name="minicpm"):
    file_name = f"{model_dir}/supervised/{city}/{index}_image.md"
    os.makedirs(f"{model_dir}/supervised/{city}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(str(image))
        file.write("\n")


def get_image_cnt(city, index, path, model_dir):
    path = f"{model_dir}/supervised/{city}/{index}_image.md"

    if not os.path.exists(path):
        return False

    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    images = [i for i in list(content.split('\n')) if i]

    return len(images) > 10


def random_walk(g, start_point, args, index, llm, tokenizer, images, city, model_dir):
    output_words(city, index, "# Experimental Result", model_dir)
    output_words(city, index, "##  Start Edge Caption", model_dir)

    id = int(g.nodes[start_point]["image"])
    image = images[id]
    output_words(city, index, f"![no_name](../../data/{city}/{index}/squeeze_images/{id}.jpg)", model_dir)
    output_images(city, index, id, model_dir)

    images_emb = []
    summary = ask_start_image(llm, tokenizer, image, "")
    output_words(city, index, summary, model_dir)

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
                             f"![no_name](../../data/{city}/{index}/squeeze_images/{id}.jpg)", model_dir)
                output_words(city, index, reason, model_dir)

                if answer > best_answer:
                    best_answer = answer
                    best_neighbor = neighbor
                    best_image = now_image
                    output_words(city, index, "best answer updated!!!!!!!!!!!!!!!!", model_dir)

            start_point = best_neighbor

            output_words(city, index, f"###  update summary", model_dir)
            summary = ask_summary_image(llm, tokenizer, best_image, summary)
            output_words(city, index, summary, model_dir)

            output_images(city, index, int(g.nodes[best_neighbor]["image"]), model_dir)
        except Exception as e:
            print(e)
            continue
    # exit(0)
    return torch.tensor(images_emb).float()


def evaluate(loader, args, llm, tokenizer, city, model_dir, image_dir):
    for index, y, s in tqdm(loader):
        index = int(index)

        if get_image_cnt(city, index, "", model_dir):
            print(f"skip {index}")
            continue
        else:
            if os.path.exists(f"{model_dir}/supervised/{city}/{index}_image.md"):
                os.remove(f"{model_dir}/supervised/{city}/{index}_image.md")
                os.remove(f"{model_dir}/supervised/{city}/{index}.md")

        sub_g, street_views, images = get_graph_and_images_dual(index, args.city_size, image_dir)

        new_g, start_point = shrink_graph(sub_g)

        random_walk(new_g, start_point, args, index, llm, tokenizer, images, city, model_dir)


city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


def main(args):
    # pip install timm==0.3.2
    # todo: change timm to 1.0.7

    image_dir, model_dir, save_dir, log_dir, task_dir = parse_json(args)

    init_seed(args.seed)

    idx_dataset = []

    city = args.city_size

    ava_indexs = load_access_street_view(image_dir, city)

    task_data = load_task_data(city, args.target, task_dir)

    for index in tqdm(ava_indexs):
        sucess_path = f"{image_dir}/{city_names[city]}/{index}/success"
        if not os.path.exists(sucess_path):
            continue
        satellite = np.load(
            f'{image_dir}/{city_names[city]}/{index}/satellite_embedding_{args.model}.npy')
        idx_dataset.append([index, task_data[int(index)][-1], satellite])

    # split the dataset into train and test
    train_size = int(0.7 * len(idx_dataset))
    val_size = int(0.8 * len(idx_dataset)) - train_size
    test_size = len(idx_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(idx_dataset, [train_size, val_size, test_size])

    if len(test_dataset) > 100:
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [100, len(test_dataset) - 100])

    for index, y, s in test_dataset:
        index = int(index)

    llm, tokenizer = get_model(args)

    evaluate(test_dataset, args, llm, tokenizer, city_names[city], model_dir, image_dir)


'''
    CUDA_VISIBLE_DEVICES=2 python vlm_walk.py --city_size 0  --location zrx
    CUDA_VISIBLE_DEVICES=3 python vlm_walk.py --city_size 1  --location zrx
    CUDA_VISIBLE_DEVICES=0 python vlm_walk.py --city_size 2  --location wb
    CUDA_VISIBLE_DEVICES=1 python vlm_walk.py --city_size 3  --location wb
    
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--location",
        type=str,
        default="wb",
        help="wb or zrx",
    )

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
        default=0,
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

    parser.add_argument(
        "--llm",
        type=str,
        default="minicpm",
        help="llm",
    )

    args = parser.parse_args()

    main(args)
