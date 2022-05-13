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

# Load the data, drums
file_path = "./datasets/NeRF_Data/nerf_synthetic/drums/train"
images = []
rgbs = []
for filename in glob.glob(f"{file_path}/*.png"):
    im = Image.open(filename)
    images.append(np.asarray(im))
    print(np.asarray(im))
    print(im.getpixel((400,400)))
    break

images = np.array(images)
(num_images, H, W, _) = images.shape
print("Shape: ", images.shape)

# Plot a random image from the dataset for visualization.
# plt.imshow(images[np.random.randint(low=0, high=num_images)])
# plt.show()
