import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

# Initialize global variables.
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 1
NUM_SAMPLES = 4
POS_ENCODE_DIMS = 16
EPOCHS = 20
RANDOM_SEED = 42


data = np.load("data.npy", allow_pickle=True)[()]

focal = data["train"]["focal"]

train_images = data["train"]["images"]
train_poses = data["train"]["poses"]

val_images = data["train"]["images"]
val_poses = data["train"]["poses"]

(num_images, H, W, _) = train_images.shape



# Download the data if it does not already exist.
# file_name = "tiny_nerf_data.npz"
# url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
# if not os.path.exists(file_name):
#     data = keras.utils.get_file(fname=file_name, origin=url)

# data = np.load(data)
# images = data["images"]
# im_shape = images.shape
# (num_images, H, W, _) = images.shape
# (poses, focal) = (data["poses"], data["focal"])

# Plot a random image from the dataset for visualization.
# plt.imshow(images[np.random.randint(low=0, high=num_images)])
# plt.show()