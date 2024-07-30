from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from PIL import Image

image = Image.open('./img.png')

processor = AutoImageProcessor.from_pretrained("../ResNet")
model = ResNetForImageClassification.from_pretrained("../ResNet")

inputs = processor(image, return_tensors="pt")

print(inputs['pixel_values'].shape)
# print(processor)

# print(list(model.children())[:-1])

# model = torch.nn.Sequential(*list(model.children())[:-1])
print(model.classifier)

tmp = model.classifier
model.classifier = torch.nn.Sequential()

emb = model(**inputs)

print(emb)

print(emb.logits.shape)

# outputs = model(**inputs, output_hidden_states=True)

# model predicts one of the 1000 ImageNet classes
# print(outputs)

logits = tmp(emb.logits)
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
