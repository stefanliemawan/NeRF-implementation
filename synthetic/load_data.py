import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import imageio
from PIL import Image
import glob
import os
import tensorflow as tf
import json

# Load the data, drums
folder_path = "../datasets/nerf_synthetic/drums"

data = {
    "train": {},
    "test": {},
    "val": {},
}

for key in data.keys():
    with open(f"{folder_path}/transforms_{key}.json", "r") as f:
        filedata = json.load(f)
        frames = filedata["frames"]
        data[key]["focal"] = frames[0]["rotation"]

        images = []
        poses = []
        for frame in frames:
            file_path = frame["file_path"][2:]
            with open(f"{folder_path}/{file_path}.png", "rb") as f:
                im = Image.open(f)
                images.append(np.asarray(im))

            poses.append(frame["transform_matrix"])

    data[key]["images"] = np.array(images)
    data[key]["poses"] = np.array(poses)


for key in data.keys():
    print(key, "focal", data[key]["focal"])
    print(key ,"images shape", data[key]["images"].shape)
    print(key ,"poses shape", data[key]["poses"].shape)

np.save("data.npy", data)
print("\ndata saved to data.npy")