import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor



resnet = models.resnet18(weights='IMAGENET1K_V1').children()
print(list(resnet))
modules = list(resnet)[:-1]
print(modules)
# modules_2 = list(resnet)[]
model = nn.Sequential(*modules)

classifier = nn.Sequential(*list(resnet)[-1])


image = Image.open('./img.png')

processor = AutoImageProcessor.from_pretrained("../ResNet")

inputs = processor(image, return_tensors="pt")

emb = model(inputs['pixel_values'])

print(model)

print(emb)

print(emb.shape)

emb = emb.view(emb.size(0), -1)

# 通过分类器进行预测
with torch.no_grad():
    logits = classifier(emb)

# 打印分类结果
predicted_label = logits.argmax(-1).item()
print(predicted_label)



