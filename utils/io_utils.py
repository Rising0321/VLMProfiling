# encoding: utf-8
import logging
import re
import torch

from ast import literal_eval

from transformers import AutoModel, AutoTokenizer

from utils.math_utils import get_dis
import os
import osmnx as ox
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np

import networkx as nx
import pyproj
import mercantile
import pandas as pd
from torchvision import transforms

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error)

from geopy.distance import geodesic, distance
from great_circle_calculator.great_circle_calculator import intermediate_point

city_names = ["New York City", "San Francisco", "Washington", "Chicago"]
value_paths = ['Carbon', 'Population', 'NightLight']


def get_tile(lon, lat, crs):
    src_crs = pyproj.CRS(crs)
    dst_crs = pyproj.CRS('EPSG:4326')
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    lon, lat = transformer.transform(lon, lat)
    tile = mercantile.tile(lon, lat, 14)
    return tile, lon, lat


def parse_edgelist_line(line):
    match = re.match(r"\(([^)]+)\) \(([^)]+)\) (.+)", line)
    if match:
        node1_str, node2_str, attr_str = match.groups()
        node1 = literal_eval(f"({node1_str})")
        node2 = literal_eval(f"({node2_str})")
        attr = ""
        return node1, node2, attr
    else:
        raise ValueError("Line format is incorrect")


def parse_json(args):
    # load data/config.json
    import json
    with open(f"./data/config.json", 'r') as f:
        data = json.load(f)

    data = data[args.location]
    image_dir = data["image_dir"]
    model_dir = data["model_dir"]
    save_dir = data["save_dir"]
    log_dir = data["log_dir"]
    task_dir = data["task_dir"]

    return image_dir, model_dir, save_dir, log_dir, task_dir


def load_access_street_view(image_dir, city):
    temp = os.listdir(f"{image_dir}/{city_names[city]}")

    return temp


def read_images(street_views, city, index, image_dir):
    images = {}
    for idx, item in street_views.iterrows():
        path = os.path.join(image_dir, f'{city_names[city]}', f'{index}', 'squeeze_images', f"{item['id_0']}.jpg")
        # path = f"{image_dir}/{city_names[city]}/{index}/squeeze_images/{item['id_0']}.jpg"
        if not os.path.exists(path):
            real_path = path.replace('squeeze_images', 'images')
            success_dir = os.path.join(image_dir, f'{city_names[city]}', f'{index}', 'squeeze_images')
            os.makedirs(success_dir, exist_ok=True)
            image = Image.open(real_path).convert('RGB')
            image = image.resize((224, 224))
            # save image to real_root_path
            image.save(path)
            images[int(item['id_0'])] = image
        else:
            try:
                image = Image.open(path).convert('RGB')
                images[int(item['id_0'])] = image  # TODO=
            except Exception as e:
                print(e)
                print(path)
                continue
    return images


def get_strat_point(sub_g):
    node_list = []
    x_min, x_max, y_min, y_max = 123456789, -123456789, 123456789, -123456789
    for node in sub_g.nodes:
        node_list.append(node)
        x_min = min(x_min, node[0])
        x_max = max(x_max, node[0])
        y_min = min(y_min, node[1])
        y_max = max(y_max, node[1])

    real_mid = [[(x_min + x_max) / 2, (y_min + y_max) / 2]]

    ans = -1
    minn_dis = 123456789
    for node in node_list:
        if get_dis(node, real_mid[0]) < minn_dis:
            minn_dis = get_dis(node, real_mid[0])
            ans = node

    return ans


def print_bottom(sub_g, street_views, colors_edge, colors):
    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, edge in enumerate(sub_g.edges):
        node1, node2 = edge
        x1, y1 = node1
        x2, y2 = node2
        plt.plot([x1, x2], [y1, y2], color=colors_edge[idx])

    for idx, item in street_views.iterrows():
        lon, lat = parse_coord(item['target'])
        plt.plot(lon, lat, "o", color=colors[idx])


def assign_color(street_views):
    # random assign a color to each street view
    colors = []
    for item in street_views.iterrows():
        colors.append(np.random.rand(3, ))
    return colors


def parse_coord(target):
    lon, lat = target[1:-1].strip().split(",")
    return float(lon), float(lat)


def assign_edge_color(sub_g, street_views, color_node):
    colors = []
    for idx1, edge in enumerate(sub_g.edges(data=True)):
        pos = 0
        for idx2, item in street_views.iterrows():
            lon, lat = parse_coord(item['target'])
            t_lon, t_lat = edge[2]['target']
            dist = geodesic((lat, lon), (t_lat, t_lon)).m
            if dist < 1e-3:
                sub_g.edges[edge[0], edge[1]]["image"] = item['id_0']  # but this maybe not exists
                # print(sub_g.edges[edge[0], edge[1]]["image"])
                # TODO
                lon, lat = parse_coord(item['target'])
                sub_g.edges[edge[0], edge[1]]["coord"] = [lon, lat]
                pos = idx2
                break
        colors.append(color_node[pos])
    # for idx1, edge in enumerate(sub_g.edges(data=True)):
    #     print(edge[2]['image'])
    # exit(0)
    return colors


def my_read_edge_list(file_path):
    # G = nx.Graph()

    G = ox.load_graphml(file_path)
    crs = G.graph['crs']
    sub_g = nx.Graph()

    for u, v, data in G.edges(data=True):
        if 'lon' in G.nodes[u] and 'lat' in G.nodes[u]:
            point1 = (float(G.nodes[u]['lon']), float(G.nodes[u]['lat']))
        else:
            x, y = G.nodes[u]['x'], G.nodes[u]['y']
            _, lon, lat = get_tile(x, y, crs)
            point1 = (lon, lat)
        if 'lon' in G.nodes[v] and 'lat' in G.nodes[v]:
            point2 = (float(G.nodes[v]['lon']), float(G.nodes[v]['lat']))
        else:
            x, y = G.nodes[v]['x'], G.nodes[v]['y']
            _, lon, lat = get_tile(x, y, crs)
            point2 = (lon, lat)
        point1_swapped = (point1[1], point1[0])  # (latitude, longitude)
        point2_swapped = (point2[1], point2[0])  # (latitude, longitude)

        dist = distance(point1_swapped, point2_swapped).meters
        if dist < 5:
            midpoint = point1
        else:
            midpoint = intermediate_point(point1, point2, 0.5)
        sub_g.add_edge(point1, point2, target=midpoint)
    sub_g.graph['crs'] = crs
    # print('G edges:', len(G.edges))
    # print('sub_g edges: ', len(sub_g.edges))
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         node1, node2, attr = parse_edgelist_line(line)
    #         G.add_edge(node1, node2)
    return sub_g


def get_graph_and_images_dual(index, city, image_dir):
    # street_views = index.replace(".edgelist", ".npy")
    sub_g = my_read_edge_list(
        f"{image_dir}/{city_names[city]}/{index}/roads.graphml")
    street_views = pd.read_csv(
        f"{image_dir}/{city_names[city]}/{index}/matches.csv")
    colors = assign_color(street_views)
    colors_edge = assign_edge_color(sub_g, street_views, colors)
    images = read_images(street_views, city, index, image_dir)
    start_point = get_strat_point(sub_g)
    # print_bottom(sub_g, street_views, colors_edge, colors)
    return sub_g, street_views, images


def output_words(args, text):
    filename = f"./log/{args.model}/{args.save_name}.md"
    # 打开文件，追加内容，然后关闭文件
    with open(filename, 'a') as file:
        file.write("\n")
        file.write(text)
        file.write("\n")


def get_streetView_diversity_distanceBase(sub_g, node):
    R = 0.003  # 距离的半径
    diversity = 0
    for now_node in sub_g.nodes:
        if (node[0] - now_node[0]) ** 2 + (node[1] - now_node[1]) ** 2 <= R ** 2:
            diversity += (R ** 2 - ((node[0] - now_node[0]) ** 2 + (node[1] - now_node[1]) ** 2)) * 1000000  # 比例可调整
    return diversity


def get_start_point(g):
    ans = -1
    diversity_max = 0
    for node in g.nodes:
        # diversity = get_streetView_diversity_crossingBase(sub_g, node)
        diversity = get_streetView_diversity_distanceBase(g, node)
        if diversity > diversity_max:
            diversity_max = diversity
            ans = node
    # output_words(args, "### Start_point's diversity is " + str(diversity_max))
    return ans


def shrink_graph(g):
    new_g = nx.Graph()
    for edge in g.edges(data=True):
        node1, node2, _ = edge
        for node in [node1, node2]:
            color_now = g.edges[edge[0], edge[1]]["image"]
            coord_now = g.edges[edge[0], edge[1]]["coord"]
            for neighbor in g.neighbors(node):
                neigh_color = g.edges[(node, neighbor)]["image"]
                neigh_coord = g.edges[(node, neighbor)]["coord"]
                if neigh_color != color_now:
                    coord_now = tuple(coord_now)
                    neigh_coord = tuple(neigh_coord)
                    new_g.add_edge(coord_now, neigh_coord)
                    # print(coord_now, neigh_coord)
                    new_g.nodes[coord_now]["image"] = color_now
                    new_g.nodes[neigh_coord]["image"] = neigh_color
    # print(len(new_g.edges))
    return new_g, get_start_point(new_g)


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def norm_image(image):
    image = image.resize((224, 224))

    image = np.array(image) / 255.

    assert image.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    image = image - imagenet_mean
    image = image / imagenet_std
    return image


def transfer_image(model, model_name, image, preprocessor):
    model.eval()
    trans = transforms.ToTensor()
    with torch.no_grad():
        if model_name == "MAE":
            image = norm_image(image)
            images = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            images = torch.einsum('nhwc->nchw', images).float().cuda()
            image = model.get_embedding(images).cpu().numpy().squeeze(0)
            # print(image.shape)  # [1024, ]
        elif model_name == 'ResNet':
            image = trans(image)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).cuda()
            inputs = preprocessor(image, do_rescale=False)
            # print(image.shape)  # torch.Size([45, 2048, 1, 1])
            image = model(
                pixel_values=torch.from_numpy(np.stack(inputs['pixel_values']))).logits.squeeze().cpu().numpy()
            # print(image.shape)  # [2048, ]
        elif model_name == 'SimCLR':
            image = trans(image)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).cuda()
            image, _, _, _ = model(image, image)
            image = image.detach().cpu().numpy().squeeze(0)
            # print(image.shape)
        elif model_name == 'CLIP':
            image = preprocessor(image).unsqueeze(0).cuda()
            image = model.encode_image(image)
            image = image.detach().cpu().numpy().squeeze(0)
        elif model_name == "ViT":
            image = trans(image)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).cuda()
            # image = (image - image.min()) / (image.max() - image.min())
            image = preprocessor(images=image, return_tensors="pt", do_rescale=False)
            image = model(**image)
            image = image.last_hidden_state[:, 0, :].detach().cpu().numpy().squeeze(0)
            # print(image.shape) # [768, 0]
    return image


def get_images(index, city, model_name, model, preprocessor, image_dir):
    index = int(index)

    real_root_path = f"{image_dir}/{city_names[city]}/{index}/squeeze_images"

    root_path = f"{image_dir}/{city_names[city]}/{index}/images"

    embedding_dir = f"{image_dir}/{city_names[city]}/{index}/image_embeddings"

    os.makedirs(embedding_dir, exist_ok=True)

    temp = os.listdir(root_path)

    images = []

    for idx, item in enumerate(temp):
        try:
            real_path = f"{real_root_path}/{item}"
            if not os.path.exists(real_path):
                path = f"{root_path}/{item}"
                image = Image.open(path).convert('RGB')
                image = image.resize((224, 224))
                # save image to real_root_path
                image.save(real_path)
            else:
                image = Image.open(real_path).convert('RGB')
        except Exception as e:
            print(e)
            print(real_path, item)
            continue

        item_name = item.replace(".jpg", "")

        embed_path = f"{embedding_dir}/{item_name}-{model_name}.npy"
        if not os.path.exists(embed_path):
            embedding = transfer_image(model, model_name, image, preprocessor)
            np.save(embed_path, embedding)
        else:
            embedding = np.load(embed_path)
        images.append(embedding)
    return None, images


'''
# todo : zhushi this line
import shutil
if os.path.exists(real_root_path):
    shutil.rmtree(real_root_path)
os.makedirs(real_root_path, exist_ok=True)
'''


def get_imagery(index, city, model_name, model, preprocessor, image_dir):
    index = int(index)

    root_path = f"{image_dir}/{city_names[city]}/{index}/satellite.tif"

    embeding_path = f"{image_dir}/{city_names[city]}/{index}/satellite_embedding_{model_name}.npy"

    if not os.path.exists(embeding_path):
        image = Image.open(root_path).convert('RGB')
        image = image.resize((224, 224))
        embedding = transfer_image(model, model_name, image, preprocessor)
        np.save(embeding_path, embedding)
    else:
        embedding = np.load(embeding_path)

    return None, embedding


def load_task_data(city, target, task_dir):
    return np.load(f"{task_dir}/{city_names[city]}/{value_paths[target]}.npy")


from loguru import logger


def init_logging(args, prefix, log_dir):
    os.makedirs(f"{log_dir}/{args.model}", exist_ok=True)
    logger.remove(handler_id=None)  # remove default logger
    file_name = f"{prefix}-{args.model}-{args.city_size}-{args.target}-{args.seed}-{args.lr}.log"
    logger.add(os.path.join(f"{log_dir}", file_name), level="INFO")
    logger.info(args)


def log_result(str):
    logger.info(str)
    print(str)


def calc_one(phase, epoch, all_predicts, all_y, loss, name):
    metrics = {}
    if loss is not None:
        metrics["loss"] = loss
    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)
    metrics["pcc"] = np.corrcoef(all_y, all_predicts)[0, 1]

    if name != "Total":
        log_result(
            f"{name}: {phase} Epoch: {epoch} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )

    return metrics


def calc(phase, epoch, all_predicts, all_y, all_city, loss, city_size, target):
    all_predicts = all_predicts
    all_y = all_y
    i = city_size
    new_predicts = []
    new_y = []

    for j in range(len(all_city)):
        if all_city[j] == i:
            new_predicts.append(all_predicts[j])
            new_y.append(all_y[j])
    target_name = value_paths[target]
    calc_one(phase, epoch, new_predicts, new_y, loss, f'{city_names[i]}: {target_name}')

    return calc_one(phase, epoch, all_predicts, all_y, loss, 'Total')


def get_model(args):
    import json
    with open(f"./data/config.json", 'r') as f:
        data = json.load(f)

    data = data[args.location]
    model_path = data[args.llm]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = AutoModel.from_pretrained(model_path,
                                      torch_dtype=torch.float16,
                                      trust_remote_code=True)
    model = model.to(device=args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    model.eval()

    return model, tokenizer


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
