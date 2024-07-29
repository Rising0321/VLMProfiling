import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from tqdm import tqdm

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


class DownStreamDataset(Dataset):
    def __init__(self, dataset, model, mean=None, std=None):
        super().__init__()
        self.imgs = []
        self.labels = []
        for images, y in tqdm(dataset):
            y_1, y_2 = y

            if y_1 < 0 or y_1 > 10000 or y_2 < 0 or y_2 > 10000:
                continue

            new_list = []
            for image in images:
                new_list.append(norm_image(image))

            model.eval()
            with torch.no_grad():
                images = torch.tensor(new_list, dtype=torch.float32)
                images = torch.einsum('nhwc->nchw', images).float().cuda()
                image = model.get_embedding(images).cpu().numpy()

            self.imgs.append(image)
            self.labels.append(y)
        if mean is None:
            self.mean = mean = np.mean(self.labels, axis=0)
            self.std = std = np.std(self.labels, axis=0)
            print(mean, std)
        self.labels = (self.labels - mean) / std

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgs = self.imgs[index]
        random.shuffle(imgs)
        imgs = imgs[:10]
        labels = self.labels[index]
        return torch.tensor(imgs), labels
