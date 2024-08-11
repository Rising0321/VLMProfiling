import os
import re
import csv

results_path = "sum_res.csv"
models = ["ResNet", "SimCLR", "CLIP", "ViT", "MAE"]

cities = [0, 1, 2, 3]
targets = [0, 1, 2]
seed = 42
results = []
best_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

lr = 1e-3

name1 = "im"
name2 = "sv+im"

pattern = r"mse: (-?[\d\.]+)\s+r2: (-?[\d\.]+)\s+rmse: (-?[\d\.]+)\s+mae: (-?[\d\.]+)\s+mape: (-?[\d\.]+)\s+pcc: (-?[\d\.]+)"


def read_res(result_path):
    with open(result_path, 'r') as file:
        lines = file.readlines()
        last_line = lines[-1]
        # 2024-07-31 23:50:38.268 | INFO     | utils.io_utils:log_result:275 - Washington: Carbon: Eval Epoch: test mse: 0.4704	r2: 0.5637	rmse: 0.6858	mae: 0.4912	mape: 2.4505	pcc: 0.7709
        # use the last line to get the results
        print(result_path, last_line)
        matches = re.search(pattern, last_line)
        mse = matches.group(1)
        r2 = matches.group(2)
        rmse = matches.group(3)
        mae = matches.group(4)
        mape = matches.group(5)
        pcc = matches.group(6)
    return mse, float(r2), rmse, mae, mape, pcc


for city in cities:
    for target in targets:
        best_r2 = -10000
        ress = []
        for model in models:
            result_path = f"../logs/{name1}-{model}-{city}-{target}-{seed}-{lr}.log"

            mse1, r2_1, rmse1, mae1, mape1, pcc1 = read_res(result_path)
            if best_r2 < r2_1:
                best_r2 = r2_1

            result_path = f"../logs/{name2}-{model}-{city}-{target}-{seed}-{lr}.log"

            mse2, r2_2, rmse2, mae2, mape2, pcc2 = read_res(result_path)
            if best_r2 < r2_2:
                best_r2 = r2_2

            ress.append(r2_1)
            ress.append(r2_2)
        cnt = 0
        for res in ress:
            if res == best_r2:
                best_cnt[cnt] += 1
                break
            cnt += 1

print(best_cnt)
