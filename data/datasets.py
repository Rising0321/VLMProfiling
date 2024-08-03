import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from torchvision import transforms

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


class DownStreamDataset(Dataset):
    def __init__(self, dataset, type, mean=None, std=None, model_name="MAE", preprocessor=None):
        super().__init__()
        self.imgs = []
        self.labels = []
        self.citys = []
        for images, y, c in tqdm(dataset):

            if type == 0:
                if y < 0 or y > 500:
                    continue
            if type == 1:
                if y < 0 or y > 10000:
                    continue

            if len(images) < 10:
                continue

            new_list = torch.tensor(np.stack(images))

            self.imgs.append(new_list)
            self.labels.append(y)
            # print(y)
            self.citys.append(c)

        if mean is None:
            self.mean = mean = np.mean(self.labels, axis=0)
            self.std = std = np.std(self.labels, axis=0)
            # print(mean, std)

        self.labels = (self.labels - mean) / std
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.citys = torch.tensor(self.citys, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # random select 10 images from self.img[index]
        imgs = self.imgs[index]
        indexs = random.sample(range(len(imgs)), 10)
        imgs = imgs[indexs]

        return imgs, self.labels[index], self.citys[index]


class ImageryDataset(Dataset):
    def __init__(self, dataset, type, mean=None, std=None, model_name="MAE", preprocessor=None):
        super().__init__()
        self.imgs = []
        self.labels = []
        self.citys = []
        for image, y, c in tqdm(dataset):
            if type == 0:
                if y < 0 or y > 500:
                    continue
            if type == 1:
                if y < 0 or y > 10000:
                    continue

            image = torch.tensor(image)

            self.imgs.append(image)
            self.labels.append(y)

            # print(y)
            self.citys.append(c)

        if mean is None:
            self.mean = mean = np.mean(self.labels, axis=0)
            self.std = std = np.std(self.labels, axis=0)
            # print(mean, std)

        self.labels = (self.labels - mean) / std
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.citys = torch.tensor(self.citys, dtype=torch.long)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index], self.labels[index], self.citys[index]
