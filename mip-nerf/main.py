import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import imageio
from tqdm import tqdm
import glob
from contextlib import redirect_stdout

from variables import *
from prepare_input import cast_cones, get_rays
from model import NeRF, get_nerf_model, get_train_monitor

(ray_origins, ray_directions) = get_rays(poses[0])
print(ray_origins)
print(ray_origins.shape)
print(ray_directions.shape)

(t_val, (mean, cov)) = cast_cones(poses[0])

print(t_val.shape)
print(mean.shape)
print(cov.shape)

# def create_gif(path_to_images, name_gif):
#     filenames = glob.glob(path_to_images)
#     filenames = sorted(filenames)
#     images = []
#     for filename in tqdm(filenames):
#         images.append(imageio.imread(filename))
#     kargs = {"duration": 0.25}
#     imageio.mimsave(name_gif, images, "GIF", **kargs)

# # Split the images into training and validation.
# train_images = images[:split_index]
# val_images = images[split_index:]

# # Split the poses into training and validation.
# train_poses = poses[:split_index]
# val_poses = poses[split_index:]

# print(train_images.shape)
# print(val_images.shape)

# print(train_poses.shape)
# print(val_poses.shape)


# # # Make the training pipeline.
# train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
# train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
# train_ray_ds = train_pose_ds.map(cast_cones, num_parallel_calls=AUTO)
# training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
# train_ds = (
#     training_ds.shuffle(BATCH_SIZE)
#     .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
#     .prefetch(AUTO)
# )

# # Make the validation pipeline.
# val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
# val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
# val_ray_ds = val_pose_ds.map(cast_cones, num_parallel_calls=AUTO)
# validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
# val_ds = (
#     validation_ds.shuffle(BATCH_SIZE)
#     .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
#     .prefetch(AUTO)
# )

# print(train_ds.element_spec)
# print(val_ds.element_spec)

# Dimension not equal

# nerf_model = get_nerf_model()
# nerf_model.build((BATCH_SIZE, 100, 100, 10, 96))
# nerf_model.summary()


# if not os.path.exists("./result"):
#     os.makedirs("./result")

# with open("./result/model_summary.txt", "w") as f:
#     with redirect_stdout(f):
#         nerf_model.summary()

# model = NeRF(nerf_model)
# model.compile(
#     optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
# )

# # Create a directory to save the images during training.
# if not os.path.exists("./result/images"):
#     os.makedirs("./result/images")


# train_monitor = get_train_monitor(train_ds)

# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     callbacks=[train_monitor],
#     steps_per_epoch=split_index//BATCH_SIZE,
# )

# create_gif("./result/images/*.png", "./result/training.gif")