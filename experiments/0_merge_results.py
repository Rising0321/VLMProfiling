import os
import re

models = ["MAE", "CLIP", "ViT", "ResNet", "SimCLR"]
cities = [0, 1, 2, 3]
targets = [0, 1]
seed = 42
results = []

pattern = r"mse: (-?[\d\.]+)\s+r2: (-?[\d\.]+)\s+rmse: (-?[\d\.]+)\s+mae: (-?[\d\.]+)\s+mape: (-?[\d\.]+)\s+pcc: (-?[\d\.]+)"

for city in cities:
    for target in targets:
        for model in models:
            result_path = f"../logs/{model}-{city}-{target}-{seed}.log"

            # open the file and read the last line
            with open(result_path, 'r') as file:
                lines = file.readlines()
                last_line = lines[-1]
                # 2024-07-31 23:50:38.268 | INFO     | utils.io_utils:log_result:275 - Washington: Carbon: Eval Epoch: test mse: 0.4704	r2: 0.5637	rmse: 0.6858	mae: 0.4912	mape: 2.4505	pcc: 0.7709
                # use the last line to get the results
                print(last_line)
                matches = re.search(pattern, last_line)
                mse = matches.group(1)
                r2 = matches.group(2)
                rmse = matches.group(3)
                mae = matches.group(4)
                mape = matches.group(5)
                pcc = matches.group(6)
                results.append([model, city, target, mse, r2, rmse, mae, mape, pcc])
import csv

results_path = "results.csv"
with open(results_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["model", "city", "target", "mse", "r2", "rmse", "mae", "mape", "pcc"])
    writer.writerows(results)
