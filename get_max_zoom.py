import json
import numpy as np

f = open("dataset/dataset.json", "r")
dataset = json.load(f)

temp = []

for key in dataset:
    temp.append(dataset[key]["zoom_factor"])

temp = np.array(temp)

print(max(temp))
