# encoding: utf-8

import re
import torch

from ast import literal_eval
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

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error)

from geopy.distance import geodesic, distance
from great_circle_calculator.great_circle_calculator import intermediate_point

city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


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


def load_access_street_view(city):
    temp = os.listdir(f"/home/wangb/OpenVIRL/data/{city_names[city]}")

    return temp


def read_images(street_views, city, index):
    images = {}
    for idx, item in street_views.iterrows():
        path = f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/squeeze_images/{item['id_0']}.jpg"
        image = Image.open(path).convert('RGB')
        images[int(item['id_0'])] = image  # TODO=
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
            dist = geodesic(parse_coord(item['target']), edge[2]['target']).m
            if dist < 1e-3:
                sub_g.edges[edge[0], edge[1]]["image"] = item['id_0']  # but this maybe not exists
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
    print('G edges:', len(G.edges))
    print('sub_g edges: ', len(sub_g.edges))
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         node1, node2, attr = parse_edgelist_line(line)
    #         G.add_edge(node1, node2)
    return sub_g


def get_graph_and_images(index, city, value_path):
    street_views = index.replace(".graphml", ".csv")
    sub_g = my_read_edge_list(
        f"/home/work/wangb/OpenVIRL/data/networks/{city_names[city]}/{value_path}/{index}")
    street_views = pd.read_csv(
        f"/home/work/wangb/OpenVIRL/data/streetviews/{city_names[city]}/{value_path}/{street_views}")
    colors = assign_color(street_views)
    colors_edge = assign_edge_color(sub_g, street_views, colors)
    images = read_images(street_views, city)
    start_point = get_strat_point(sub_g)
    print_bottom(sub_g, street_views, colors_edge, colors)
    return sub_g, street_views, images, start_point


def get_graph_and_images_dual(index, city, value_path):
    # street_views = index.replace(".edgelist", ".npy")
    sub_g = my_read_edge_list(
        f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/roads.graphml")
    street_views = pd.read_csv(
        f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/matches.csv")
    colors = assign_color(street_views)
    colors_edge = assign_edge_color(sub_g, street_views, colors)
    images = read_images(street_views, city, index)
    start_point = get_strat_point(sub_g)
    print_bottom(sub_g, street_views, colors_edge, colors)
    return sub_g, street_views, images


def get_images(index, city):
    index = int(index)
    real_root_path = f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/squeeze_images/"
    '''
    # todo : zhushi this line
    import shutil
    if os.path.exists(real_root_path):
        shutil.rmtree(real_root_path)
    os.makedirs(real_root_path, exist_ok=True)
    '''
    root_path = f"/home/wangb/OpenVIRL/data/{city_names[city]}/{index}/images"
    temp = os.listdir(root_path)

    images = []

    for idx, item in enumerate(temp):
        real_path = f"{real_root_path}/{item}"
        try:
            if not os.path.exists(real_path):
                path = f"{root_path}/{item}"
                image = Image.open(path).convert('RGB')
                image = image.resize((224, 224))
                # save image to real_root_path
                image.save(real_path)
            else:
                image = Image.open(real_path).convert('RGB')
            images.append(image)
        except Exception as e:
            print(e)
            print(real_path, item)
            continue

    return None, images


def load_task_data(city):
    temp1 = np.load(f"/home/wangb/zhangrx/LLMInvestrigator/data/TaskData/{city_names[city]}/Carbon.npy")
    temp2 = np.load(f"/home/wangb/zhangrx/LLMInvestrigator/data/TaskData/{city_names[city]}/Population.npy")
    return [temp1, temp2]


def calc_one(phase, epoch, all_predicts, all_y, loss, name):
    metrics = {}
    if loss is not None:
        metrics["loss"] = loss
    metrics["mse"] = mean_squared_error(all_y, all_predicts)
    metrics["r2"] = r2_score(all_y, all_predicts)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(all_y, all_predicts)
    metrics["mape"] = mean_absolute_percentage_error(all_y, all_predicts)

    if name != "Total":
        print(
            f"{name}: {phase} Epoch: {epoch} "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )
    return metrics


def calc(phase, epoch, all_predicts, all_y, all_city, loss, city_size):
    all_predicts = all_predicts
    all_y = all_y
    for i in range(city_size):
        new_predicts = []
        new_y = []
        for target in range(1):
            for j in range(len(all_city)):
                if all_city[j] == i:
                    new_predicts.append(all_predicts[j][target])
                    new_y.append(all_y[j][target])
            target_name = "Carbon" if target == 0 else "Population"
            calc_one(phase, epoch, new_predicts, new_y, loss, f'{city_names[i]}: {target_name}')

    return calc_one(phase, epoch, all_predicts, all_y, loss, 'Total')


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

