import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

from utils.io_utils import norm_image


class DownStreamDataset(Dataset):
    def __init__(self, dataset, model, mean=None, std=None):
        super().__init__()
        self.imgs = []
        self.labels = []
        self.citys = []
        for images, y, c in tqdm(dataset):
            y_1, y_2 = y

            if y_1 < 0 or y_1 > 10000 or y_2 < 0 or y_2 > 10000:
                continue

            self.imgs.append(images)
            self.labels.append(y)
            self.citys.append(c)

        if mean is None:
            self.mean = mean = np.mean(self.labels, axis=0)
            self.std = std = np.std(self.labels, axis=0)
            print(mean, std)

        self.labels = (self.labels - mean) / std
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.citys = torch.tensor(self.citys, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgs = self.imgs[index]
        random.shuffle(imgs)
        imgs = imgs[:10]
        labels = self.labels[index]
        return torch.tensor(imgs), labels, self.citys[index]
