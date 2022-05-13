from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np

from load_prepare_data import get_rgb
import variables as var

def load_data():
    # Download the data if it does not already exist.
    file_name = "tiny_nerf_data.npz"
    url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
    if not os.path.exists(file_name):
        data = keras.utils.get_file(fname=file_name, origin=url)

    data = np.load(data)

    return data 

    # Plot a random image from the dataset for visualization.
    # plt.imshow(images[np.random.randint(low=0, high=num_images)])
    # plt.show()

def main():
    data = load_data()
    images = data["images"]

    print("Images shape:", images.shape)

    print(images[0])

    (num_images, H, W, _) = images.shape
    (poses, focal) = (data["poses"], data["focal"])

    


main()