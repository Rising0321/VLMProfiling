import os

models = ["ResNet", "SimCLR", "CLIP", "ViT", "MAE"]
cities = [0, 1, 2, 3]
targets = [0, 1]
lrs = [1e-3, 1e-4, 1e-5, 5e-4, 5e-5]
# excute python main_train_imagery.py --model {model} --city_size {city} --target {target}

for model in models:
    for city in cities:
        for target in targets:
            for lr in lrs:
                os.system(
                    f"/home/wangb/.conda/envs/llm/bin/python main_train_imagery.py --model {model} --city_size {city} --target {target} --lr {lr}")