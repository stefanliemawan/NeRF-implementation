import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from variables import *


def get_nerf_model(num_layers=8, dense_units=64):
    sample_input = layers.Input(shape=(100, 100, 12, 88), batch_size=BATCH_SIZE)

    x = sample_input
    for i in range(num_layers):
        x = layers.Dense(units=dense_units, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection.
            x = layers.concatenate([x, sample_input], axis=-1)

    rgb = layers.Dense(units=3, activation="sigmoid")(x)
    sigma = layers.Dense(units=1, activation="relu")(x)

    # outputs = layers.Dense(units=4)(x)

    model = keras.Model(inputs=sample_input, outputs=[rgb, sigma])

    return model

def render_image_depth(rgb, sigma, t_vals):
	sigma = sigma[..., 0]
	delta = t_vals[..., 1:] - t_vals[..., :-1]
	delta_shape = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1]
	delta = tf.concat(
		[delta, tf.broadcast_to([1e10], shape=delta_shape)], axis=-1)

	# calculate alpha from sigma and delta values
	alpha = 1.0 - tf.exp(-sigma * delta)

	# calculate the exponential term for easier calculations
	exp_term = 1.0 - alpha
	epsilon = 1e-10

	# calculate the transmittance and weights of the ray points
	transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1,
		exclusive=True)
	weights = alpha * transmittance
	
	# build the image and depth map from the points of the rays
	image = tf.reduce_sum(weights[..., None] * rgb, axis=-2)
	depth = tf.reduce_sum(weights * t_vals, axis=-1)
	
	# return rgb, depth map and weights
	return (image, depth, weights)


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
        self.ssim_metric = keras.metrics.Mean(name="ssim")

    def train_step(self, inputs):
        (images, (t_vals, samples)) = inputs

        with tf.GradientTape() as tape:
            (rgb, sigma) = self.nerf_model(samples)
            (predicted_images, _, _) = render_image_depth(rgb, sigma, t_vals)
            # modify the render_image_depth to use samples
            loss = self.loss_fn(images, predicted_images)

        # hierarchical sampling?

        trainable_variables = self.nerf_model.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        psnr = tf.image.psnr(images, rgb, max_val=1.0)
        ssim = tf.image.ssim(images, rgb, max_val=1.0)

        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)
        
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result(), "ssim": self.ssim_metric.result()}

    def test_step(self, inputs):
        (images, (t_vals, samples)) = inputs

        (rgb, sigma) = self.model(samples)
        (predicted_images, _, _) = render_image_depth(rgb, sigma, t_vals)
        
        loss = self.loss_fn(images, predicted_images)

        psnr = tf.image.psnr(images, rgb, max_val=1.0)
        ssim = tf.image.ssim(images, rgb, max_val=1.0)

        self.loss_tracker.update_state(loss)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)
        
        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result(), "ssim": self.ssim_metric.result()}

def get_train_monitor(train_ds):
    (test_images, (t_vals, samples)) = next(iter(train_ds))

    class TrainMonitor(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            (test_predicted_images, test_sigma) = self.model.predict(samples)

            (test_predicted_images, test_depths, test_weights) = self.model.render_image_depth(test_predicted_images, test_sigma, t_vals)
            
            (_, ax) = plt.subplots(nrows=1, ncols=4, figsize=(10, 10))
            ax[0].imshow(keras.preprocessing.image.array_to_img(test_predicted_images[0]))
            ax[0].set_title(f"Predicted Image")
            ax[2].imshow(keras.preprocessing.image.array_to_img(test_depths[0, ..., None]), 
				cmap="inferno")
            ax[2].set_title(f"Depth Image")
            ax[3].imshow(keras.preprocessing.image.array_to_img(test_images[0]))
            ax[3].set_title(f"Real Image")

            plt.savefig(f"./result/images/{epoch:03d}.png")
            plt.close()

    train_monitor = TrainMonitor()
    
    return train_monitor