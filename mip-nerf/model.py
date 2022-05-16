import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from variables import *


def get_model(num_layers=8, dense_units=64):
    sample_input = layers.Input(shape=(BATCH_SIZE, 100, 100, 10, 96))

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


class NerfTrainer(keras.Model):
    def __init__(self, render_image_depth):
        super().__init__()
        self.render_image_depth = render_image_depth
    
    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer

        self.loss_fn = loss_fn
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")
        self.ssim_metric = keras.metrics.Mean(name="ssim")

    def train_step(self, inputs):
        (images, samples) = inputs

        with tf.GradientTape() as tape:
            (rgb, sigma) = self.model(samples)
            # modify the render_image_depth to use samples