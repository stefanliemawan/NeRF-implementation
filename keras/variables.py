import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

# Initialize global variables.
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 1
NUM_SAMPLES = 10
POS_ENCODE_DIMS = 16
EPOCHS = 50
RANDOM_SEED = 42

print("BATCH_SIZE:", BATCH_SIZE)
print("NUM_SAMPLES:", NUM_SAMPLES)
print("POS_ENCODE_DIMS:", POS_ENCODE_DIMS)
print("EPOCHS:", EPOCHS)

# Download the data if it does not already exist.
file_name = "tiny_nerf_data.npz"
url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
if not os.path.exists(file_name):
    data = keras.utils.get_file(fname=file_name, origin=url)

data = np.load(data)
images = data["images"]
im_shape = images.shape
(num_images, H, W, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])


# Create the training split.
split_index = int(num_images * 0.8)

# Plot a random image from the dataset for visualization.
# plt.imshow(images[np.random.randint(low=0, high=num_images)])
# plt.show()