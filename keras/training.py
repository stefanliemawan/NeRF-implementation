from tensorflow import keras
import tensorflow as tf
import os
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import glob
from contextlib import redirect_stdout

import variables as var
from model import get_nerf_model, render_rgb_depth
from render3d import render

class NeRF(keras.Model):
    def __init__(self, nerf_model):
        super().__init__()
        self.nerf_model = nerf_model

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")

    def train_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        with tf.GradientTape() as tape:
            # Get the predictions from the model.
            rgb, _ = render_rgb_depth(
                model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
            )
            loss = self.loss_fn(images, rgb)

        # Get the trainable variables.
        trainable_variables = self.nerf_model.trainable_variables

        # Get the gradeints of the trainiable variables with respect to the loss.
        gradients = tape.gradient(loss, trainable_variables)

        # Apply the grads and optimize the model.
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    def test_step(self, inputs):
        # Get the images and the rays.
        (images, rays) = inputs
        (rays_flat, t_vals) = rays

        # Get the predictions from the model.
        rgb, _ = render_rgb_depth(
            model=self.nerf_model, rays_flat=rays_flat, t_vals=t_vals, rand=True
        )
        loss = self.loss_fn(images, rgb)

        # Get the PSNR of the reconstructed images and the source images.
        psnr = tf.image.psnr(images, rgb, max_val=1.0)

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]

def create_gif(path_to_images, name_gif):
    filenames = glob.glob(path_to_images)
    filenames = sorted(filenames)
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)


def train(train_ds, val_ds):
    test_imgs, test_rays = next(iter(train_ds))
    test_rays_flat, test_t_vals = test_rays

    loss_list = []

    class TrainMonitor(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = logs["loss"]
            loss_list.append(loss)
            test_recons_images, depth_maps = render_rgb_depth(
                model=self.model.nerf_model,
                rays_flat=test_rays_flat,
                t_vals=test_t_vals,
                rand=True,
                train=False,
            )

            # Plot the rgb, depth and the loss plot.
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
            ax[0].imshow(keras.preprocessing.image.array_to_img(test_recons_images[0]))
            ax[0].set_title(f"Predicted Image: {epoch:03d}")

            ax[1].imshow(keras.preprocessing.image.array_to_img(depth_maps[0, ..., None]))
            ax[1].set_title(f"Depth Map: {epoch:03d}")

            ax[2].plot(loss_list)
            ax[2].set_xticks(np.arange(0, var.EPOCHS + 1, 5.0))
            ax[2].set_title(f"Loss Plot: {epoch:03d}")

            fig.savefig(f"./result/images/{epoch:03d}.png")
            # plt.show()
            # plt.close()


    num_pos = var.H * var.W * var.NUM_SAMPLES
    nerf_model = get_nerf_model(num_layers=8, num_pos=num_pos)
    nerf_model.build((num_pos, 2 * 3 * var.POS_ENCODE_DIMS + 3))
    nerf_model.summary()

    with open("./result/model_summary.txt", "w") as f:
        with redirect_stdout(f):
            nerf_model.summary()

    model = NeRF(nerf_model)
    model.compile(
        optimizer=keras.optimizers.Adam(), loss_fn=keras.losses.MeanSquaredError()
    )

    # Create a directory to save the images during training.
    if not os.path.exists("./result/images"):
        os.makedirs("./result/images")


    model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=var.BATCH_SIZE,
        epochs=var.EPOCHS,
        callbacks=[TrainMonitor()],
        steps_per_epoch=var.split_index//var.BATCH_SIZE,
    )

    create_gif("./result/images/*.png", "./result/training.gif")

    # Inference

    nerf_model = model.nerf_model

    test_recons_images, depth_maps = render_rgb_depth(
        model=nerf_model,
        rays_flat=test_rays_flat,
        t_vals=test_t_vals,
        rand=True,
        train=False,
    )

    # Create subplots.
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 20))

    for ax, ori_img, recons_img, depth_map in zip(
        axes, test_imgs, test_recons_images, depth_maps
    ):
        ax[0].imshow(keras.preprocessing.image.array_to_img(ori_img))
        ax[0].set_title("Original")

        ax[1].imshow(keras.preprocessing.image.array_to_img(recons_img))
        ax[1].set_title("Reconstructed")

        ax[2].imshow(
            keras.preprocessing.image.array_to_img(depth_map[..., None]), cmap="inferno"
        )
        ax[2].set_title("Depth Map")

    render(nerf_model)