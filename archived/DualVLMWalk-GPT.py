# -*- coding: utf-8 -*-
import argparse
import os
import random

import matplotlib.pyplot as plt

import numpy as np

from ast import literal_eval
import re

import math

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer

import torch

from utils.io_utils import load_access_street_view, get_graph_and_images, get_graph_and_images_dual, output_words
from utils.math_utils import pnt2line, get_dis

import networkx as nx
import base64

import requests
import json
from io import BytesIO

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

city_names = ["New York City", "San Francisco", "Washington", "Chicago"]







def get_model(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = AutoModel.from_pretrained('/home/wangb/zhangrx/models/MiniCPM-Llama3-V-2_5',
                                      torch_dtype=torch.float16,
                                      trust_remote_code=True)
    model = model.to(device=args.gpu)

    tokenizer = AutoTokenizer.from_pretrained('/home/wangb/zhangrx/models/MiniCPM-Llama3-V-2_5',
                                              trust_remote_code=True)
    model.eval()

    return model, tokenizer


def ask_start_image(model, tokenizer, image, previous_summary):
    # 假设你是一个城市调查员。你在调查当地的人口密度与碳排放量。请你根据看到的图片记录一些有用的信息，这些信息会用于给其他人员进行分析。未来你还会看到一些信息的，只需要描述你看到的这张照片即可。
    question = "Assume you are a urban investigator. " \
               "Offer a comprehensive summary of human activity, urban infrastructure, and environments in the image"

    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.2,
        stream=True
    )

    generated_text = ""
    for new_text in res:
        generated_text += new_text

    print(generated_text)
    return generated_text


def get_score(generated_text):
    match = re.search(r".*\{\s*\"Difference_Score\": (.+)\s*}.*", generated_text)

    if match:
        score = match.groups()[0]
        print(score)
        return int(score)
    else:
        raise ValueError("Line format is incorrect")


def ask_summary_image(model, tokenizer, image, previous_summary):
    # 假设你是一个城市调查员。描述你看到的图片与后文输入的summary里在human activity, urban infrastructure, and environments的区别。
    # 最后请输出一个你估计的相似度分数，范围在0-100之间。形式为{Difference: aaa; Answer: bbb}。Summary是{previous_summary}
    question = "Assume you are a urban investigator. " \
               "You will input a summary of human activity, urban infrastructure, and environments in the following. " \
               "Please modify the summary through what you see in the image. Only output the modified version" \
               f"The summary is {previous_summary}"

    msgs = [{'role': 'user', 'content': question}]

    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.2,
        stream=True
    )

    generated_text = ""
    for new_text in res:
        generated_text += new_text

    print(generated_text)
    return generated_text


def ask_middle_image(model, tokenizer, image, previous_summary):
    # 假设你是一个城市调查员。描述你看到的图片与后文输入的summary里在human activity, urban infrastructure, and environments的区别。
    # 最后请输出一个你估计的相似度分数，范围在0-100之间。形式为{Difference: aaa; Answer: bbb}。Summary是{previous_summary}
    question = "Assume you are a urban investigator. " \
               "Describe the difference between the image you see and the summary following in terms of human activity, urban infrastructure, and environments. " \
               "Please output an estimated difference score ranging from 0 to 100 in a json format as {Difference_Score: x} in the end. " \
               f"The summary is {previous_summary}"

    msgs = [{'role': 'user', 'content': question}]

    while True:
        print("Enter while")
        try:
            generated_text = ""
            if args.model == 'MiniCPM':
                res = model.chat(
                    image=image,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.2,
                    stream=True
                )
                for new_text in res:
                    generated_text += new_text
            elif args.model == 'GPT':
                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_base64 = f"data:image/jpeg;base64,{encoded_image}"
                print("sent requests")
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer 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",
                    },
                    data=json.dumps({
                        "model": "openai/gpt-4o-mini-2024-07-18",  # Optional
                        "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": question
                                },
                                {
                                    "type": "image_url",
                                    "image_url": image_base64
                                }
                                ]
                            }
                        ]
                    })
                )
                print(response.json())
                res = response.json()
                generated_text = res['choices'][0]['message']['content']

            print(generated_text)
            score = get_score(generated_text)
            return score, generated_text

        except Exception as e:
            print(e)
            continue




def llm_walk(sub_g, start_point, model, tokenizer, images, args):
    output_words(args, "# Experimental Result")
    output_words(args, "##  Start Edge Caption")

    now_point = start_point

    id = int(sub_g.nodes[now_point]["image"])
    image = images[id]
    output_words(args, f"![no_name](../../data/{city_names[args.city]}/{index}/squeeze_images/{id}.jpg)")

    summary = ask_start_image(model, tokenizer, image, "")
    output_words(args, summary)

    for iter in range(5):

        output_words(args, f"##  Start Round {iter}")

        neighbors = list(sub_g.neighbors(now_point))

        best_neighbor = -1
        best_answer = -1
        best_image = -1
        for neighbor in neighbors:

            output_words(args, f"###  conside neighbor {neighbor}")

            id = int(sub_g.nodes[neighbor]["image"])
            now_image = images[id]
            # ask the answer by the summary
            answer, reason = ask_middle_image(model, tokenizer, now_image, summary)
            output_words(args, f"![no_name](../../data/{city_names[args.city]}/{index}/squeeze_images/{id}.jpg)")
            output_words(args, reason)
            if answer > best_answer:
                best_answer = answer
                best_neighbor = neighbor
                best_image = now_image
                output_words(args, "best answer updated!!!!!!!!!!!!!!!!")

        output_words(args, f"###  update summary")
        summary = ask_summary_image(model, tokenizer, best_image, summary)
        output_words(args, summary)

        plt.quiver(now_point[0], now_point[1], best_neighbor[0] - now_point[0], best_neighbor[1] - now_point[1]
                   , angles='xy', scale_units='xy', scale=1, color="r")

        now_point = best_neighbor

    plt.show()
    # plt.savefig(f"./log/{args.model}/ans.jpg")


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# CUDA_VISIBLE_DEVICES=1 python DualVLMWalk.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_name",
        type=str,
        default="GPT_test",
        help="save_name",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="GPT",
        choices=["MiniCPM", "GPT"],
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
        "--value_path",
        type=str,
        default="Carbon",
        help="type name",
    )

    parser.add_argument(
        '--gpu',
        type=str,
        default='cuda:0'
    )

    # print(torch.cuda.is_available())
    # exit(0)
    args = parser.parse_args()

    os.makedirs(f"./log/{args.model}", exist_ok=True)

    ava_indexs = load_access_street_view(args.city)
    model, tokenizer = get_model(args)
    for index in tqdm(ava_indexs):
        if int(index) == 14:
            sub_g, street_views, images = get_graph_and_images_dual(index, args.city, args.value_path)
            new_g, start_point = shrink_graph(sub_g)
            llm_walk(new_g, start_point, model, tokenizer, images, args)
            break