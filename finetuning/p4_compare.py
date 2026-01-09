import json
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

pwd = Path(__file__).parent

IN_PATH = pwd / "outputs" / "eval_logs"
OUT_PATH = pwd / "outputs" / "plots"

file_paths = [
    IN_PATH / "eval_outputs_base_x8.jsonl",
    IN_PATH / "eval_outputs_finetuned_3ga4dhm9_x8.jsonl",
    # IN_PATH / "eval_outputs_finetuned_ef002ugp_x8.jsonl",
]

files = {
    "base": file_paths[0],
    "FT": file_paths[1],
}

os.makedirs(OUT_PATH, exist_ok=True)
OUT_FILE = OUT_PATH / "comparison_3ga4dhm9.png"

print(files)
file_correct = {}

for file_name, file_path in files.items():
    file_data = []
    with open(file_path, "r") as f:
        for line in f:
            line_data = json.loads(line)
            file_data.append(line_data)

    file_correct[file_name] = []
    for sample in file_data:
        file_correct[file_name].append(float(np.mean(sample["correct"])))
    # print(file_correct[file_name])

noise_x = np.random.normal(0, 0.02, len(file_correct["base"]))
noise_y = np.random.normal(0, 0.02, len(file_correct["FT"]))
xs = file_correct["base"] + noise_x
ys = file_correct["FT"] + noise_y


plt.scatter(xs, ys, s=10)
plt.xlabel("Base Model Accuracy")
plt.ylabel("Finetuned Model Accuracy")
plt.title("Comparison of Base and Finetuned Model Accuracy")
plt.savefig(OUT_FILE)

for file_name, correct_list in file_correct.items():
    print(file_name, f"Mean: {np.mean(correct_list)}, Std: {np.std(correct_list)}")