import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("csv_1", type=str)
parser.add_argument("csv_2", type=str)

args = parser.parse_args()

pred_data = pd.read_csv(args.csv_1)
val_data = pd.read_csv(args.csv_2)

# compare label and return accuracy
correct = 0
for i in range(len(pred_data)):
    if pred_data["label"][i] == val_data["label"][i]:
        correct += 1

accuracy = correct / len(pred_data)

print(accuracy)
