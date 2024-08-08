import os

models = ["ResNet", "SimCLR", "CLIP", "ViT", "MAE"]
cities = [3]
targets = [0, 1, 2]
lrs = [1e-3, 1e-4, 1e-5, 5e-4, 5e-5]
# excute python main_train_imagery.py --model {model} --city_size {city} --target {target}

for model in models:
    for city in cities:
        for target in targets:
            # os.system(
            #     f"/home/wangb/.conda/envs/llm/bin/python main_train_imagery.py --model {model} --city_size {city} --target {target}")

            # os.system(
            #     f"/home/wangb/.conda/envs/llm/bin/python main_train_baseline.py --model {model} --city_size {city} --target {target}")

            # os.system(
            #     f"/home/wangb/.conda/envs/llm/bin/python main_train_baseline_imagery.py --model {model} --city_size {city} --target {target}")

            os.system(
                f"/home/wangb/.conda/envs/llm/bin/python main_train_VLM.py --model {model} --city_size {city} --target {target}")
