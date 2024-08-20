# encoding: utf-8

import os

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from data.datasets import DownStreamDataset, WalkDataset
from utils.io_utils import load_access_street_view, load_task_data, init_seed, \
    get_graph_and_images_dual, shrink_graph, parse_json
from DualVLMWalk_gpt import ask_start_image, ask_middle_image, ask_summary_image


def output_words(city, index, text, model_dir, model_name):
    file_name = f"{model_dir}/supervised/{city}/{model_name}/{index}.md"
    os.makedirs(f"{model_dir}/supervised/{city}/{model_name}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(text)
        file.write("\n")


def output_images(city, index, image, model_dir, model_name):
    file_name = f"{model_dir}/supervised/{city}/{model_name}/{index}_image.md"
    os.makedirs(f"{model_dir}/supervised/{city}/{model_name}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(str(image))
        file.write("\n")


def get_image_cnt(city, index, path, model_dir, model_name):
    path = f"{model_dir}/supervised/{city}/{model_name}/{index}_image.md"

    if not os.path.exists(path):
        return False

    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    images = [i for i in list(content.split('\n')) if i][-9:]

    return images


def random_walk(g, start_point, args, index, llm, tokenizer, images, city, model_dir):
    id = int(g.nodes[start_point]["image"])
    image = images[id]
    images = get_image_cnt(city, index, llm, model_dir, llm)
    plt.plot(start_point[0], start_point[1], 'x', color='red', markersize=20)
    for i in range(9):
        try:
            neighbors = list(g.neighbors(start_point))
            tmp = -1
            for neighbor in neighbors:
                id = int(g.nodes[neighbor]["image"])
                # print(neighbor, id)
                if id == int(images[i]):
                    tmp = neighbor
                    # print("heree")
                    break
            # print(start_point, tmp)
            if tmp == -1:
                raise ValueError("tmp can not be -1")
            plt.quiver(start_point[0], start_point[1], tmp[0] - start_point[0], tmp[1] - start_point[1]
                       , angles='xy', scale_units='xy', scale=1, color="r")
            start_point = tmp
        except Exception as e:
            print(index, e)
            continue

    # print(index)
    os.makedirs(f'./case/{index}/', exist_ok=True)
    plt.savefig(f'./case/{index}/{llm}.png')
    plt.close('all')
    return 0


def evaluate(loader, args, llm, tokenizer, city, model_dir, image_dir):
    for index in tqdm(loader):
        index = int(index)

        # if get_image_cnt(city, index, "", model_dir, llm):
        #     print(f"skip {index}")
        #     continue
        # else:
        #     if os.path.exists(f"{model_dir}/supervised/{city}/{llm}/{index}_image.md"):
        #         os.remove(f"{model_dir}/supervised/{city}/{llm}/{index}_image.md")
        #         os.remove(f"{model_dir}/supervised/{city}/{llm}/{index}.md")

        sub_g, street_views, images = get_graph_and_images_dual(index, args.city_size, image_dir)

        new_g, start_point = shrink_graph(sub_g)

        random_walk(new_g, start_point, args, index, llm, tokenizer, images, city, model_dir)


city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


def main(args):
    # pip install timm==0.3.2
    # todo: change timm to 1.0.7
    print(args.llm)

    image_dir, model_dir, save_dir, log_dir, task_dir = parse_json(args)

    init_seed(args.seed)

    idx_dataset = []

    city = args.city_size

    ava_indexs = load_access_street_view(image_dir, city)

    task_data = load_task_data(city, args.target, task_dir)

    for index in tqdm(ava_indexs):
        # sucess_path = f"{image_dir}/{city_names[city]}/{index}/success"
        # if not os.path.exists(sucess_path):
        #     continue
        # satellite = np.load(
        #     f'{image_dir}/{city_names[city]}/{index}/satellite_embedding_{args.model}.npy')
        idx_dataset.append([index, task_data[int(index)][-1]])

    # split the dataset into train and test
    train_size = int(0.7 * len(idx_dataset))
    val_size = int(0.8 * len(idx_dataset)) - train_size
    test_size = len(idx_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(idx_dataset, [train_size, val_size, test_size])

    if len(test_dataset) > 100:
        test_dataset, _ = torch.utils.data.random_split(test_dataset, [100, len(test_dataset) - 100])

    for index, y in test_dataset:
        index = int(index)

    if args.llm == "FireLLaVA" or args.llm == "Gemini" or args.llm == "Claude" or args.llm == 'minicpm':
        llm, tokenizer = args.llm, 0

    test_dataset = ['2141', '901', '2717', '1307', '2569', '1629', '1603', '1655', '1323', '844', '1696', '1341',
                    '1990', '1449', '1465', '1571', '654', '978', '1371', '2046', '1344', '2045', '853', '2042', '2226',
                    '1110', '2701', '1867', '2357', '1564', '1850', '2119', '1591', '1378', '1701', '2060', '1081',
                    '387', '546', '1866', '192', '1644', '2367', '1175', '1786', '2492', '376', '1434', '450', '2637',
                    '641', '2343', '854', '1446', '1585', '1589', '716', '1572', '1613', '1459', '1472', '1306', '2074',
                    '1843', '1590', '259', '721', '1467', '576', '2269', '912', '1601', '1982', '516', '1809', '1731',
                    '1728', '3088', '1402', '1708', '1789', '1499', '1387', '1258', '2720', '2475', '1976', '2242',
                    '2634', '2562', '447', '1498', '2964', '785', '1587', '841', '1810', '1663', '1181', '1631']

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
        default="local",
        help="wb or zrx or local",
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
        default=0,
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
        default="Claude",
        choices=["minicpm", "FireLLaVA", "Gemini", "Claude"],
        help="llm",
    )

    args = parser.parse_args()

    main(args)
