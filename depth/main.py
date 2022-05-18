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
from PIL import Image
import io
import cv2

# Initialize global variables.
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 1
NUM_SAMPLES = 10
POS_ENCODE_DIMS_RAYS = 16
POS_ENCODE_DIMS_DIRS = 16
EPOCHS = 50
RANDOM_SEED = 42
NEAR = 2.0
FAR = 6.0
NUM_LAYERS = 8
DENSE_UNITS = 64
DEPTH_LOSS_BALANCER = 0.1

print("BATCH_SIZE:", BATCH_SIZE)
print("NUM_SAMPLES:", NUM_SAMPLES)
print("POS_ENCODE_DIMS_RAYS:", POS_ENCODE_DIMS_RAYS)
print("POS_ENCODE_DIMS_DIRS:", POS_ENCODE_DIMS_DIRS)
print("EPOCHS:", EPOCHS)

# Download the data if it does not already exist.
file_name = "tiny_nerf_data.npz"
url = "https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz"
if not os.path.exists(file_name):
    data = keras.utils.get_file(fname=file_name, origin=url)

data = np.load(data)
images = data["images"]
(num_images, IMAGE_HEIGHT, IMAGE_WIDTH, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

split_index = int(num_images * 0.9)

depths = images[:,:,:,0]
print(depths.shape)

# depths = []

# for depth_file in os.listdir("../datasets/tiny_nerf/depths"):
#     image = Image.open(f"../datasets/tiny_nerf/depths/{depth_file}")
#     depths.append(np.asarray(image))

# depths = np.asarray(depths)
# depths = depths[:, :, :, 0]

# not sure if this is correct for depths

def get_rays(pose):
    x, y = tf.meshgrid(
        tf.range(IMAGE_WIDTH, dtype=tf.float32),
        tf.range(IMAGE_HEIGHT, dtype=tf.float32),
        indexing="xy",
    )

    x_camera = (x - IMAGE_WIDTH * 0.5) / focal
    y_camera = (y - IMAGE_HEIGHT * 0.5) / focal

    xyz_camera = tf.stack([x_camera, -y_camera, -tf.ones_like(x)], axis=-1)
    
    rotation = pose[:3, :3]
    translation = pose[:3, -1]

    xyz_camera = xyz_camera[..., None, :]
    xyz_world = xyz_camera * rotation

    ray_directions = tf.reduce_sum(xyz_world, axis=-1)
    ray_directions = ray_directions / tf.norm(ray_directions, axis=-1, keepdims=True)
    ray_origins = tf.broadcast_to(translation, tf.shape(ray_directions))

    t_vals = tf.linspace(NEAR, FAR, NUM_SAMPLES)
    # add randomness
    shape = list(ray_origins.shape[:-1]) + [NUM_SAMPLES]
    noise = tf.random.uniform(shape=shape) * (FAR - NEAR) / NUM_SAMPLES
    t_vals = t_vals + noise

    return (ray_origins, ray_directions, t_vals)


def positional_encoding(x, encode_dims):
    positions = [x]
    for i in range(encode_dims):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0 ** i * x))
    return tf.concat(positions, axis=-1)

def hierarchical_sampling(t_vals_mid, weights, num_samples_fine):
	weights += 1e-5

	pdf = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
	cdf = tf.cumsum(pdf, axis=-1)
	cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], axis=-1)

	u_shape = [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, num_samples_fine]
	u = tf.random.uniform(shape=u_shape)

	indices = tf.searchsorted(cdf, u, side="right")

	below = tf.maximum(0, indices-1)
	above = tf.minimum(cdf.shape[-1]-1, indices)
	indices_g = tf.stack([below, above], axis=-1)
	
	cdf_g = tf.gather(cdf, indices_g, axis=-1,
		batch_dims=len(indices_g.shape)-2)
	
	t_vals_mid_g = tf.gather(t_vals_mid, indices_g, axis=-1,
		batch_dims=len(indices_g.shape)-2)

	denom = cdf_g[..., 1] - cdf_g[..., 0]
	denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)

	t = (u - cdf_g[..., 0]) / denom
	samples = (t_vals_mid_g[..., 0] + t * 
		(t_vals_mid_g[..., 1] - t_vals_mid_g[..., 0]))
	
	return samples

def get_model(num_layers=8, dense_units=64):
    ray_input = layers.Input(shape=(None, None, None, 2 * 3 * POS_ENCODE_DIMS_RAYS + 3), batch_size=BATCH_SIZE)
    dir_input = layers.Input(shape=(None, None, None, 2 * 3 * POS_ENCODE_DIMS_DIRS + 3), batch_size=BATCH_SIZE)

    x = ray_input
    for i in range(num_layers):
        x = layers.Dense(units=dense_units, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection.
            x = layers.concatenate([x, ray_input], axis=-1)

    sigma = layers.Dense(units=1, activation="relu")(x)

    feature = layers.Dense(units=dense_units)(x)

    feature = layers.concatenate([feature, dir_input], axis=-1)
    x = layers.Dense(units=dense_units//2, activation="relu")(feature)

    rgb = layers.Dense(units=3, activation="sigmoid")(x)

    model = keras.Model(inputs=[ray_input, dir_input], outputs=[rgb, sigma])

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
    def __init__(self, coarse_model, fine_model, num_samples_fine):
        super().__init__()
        self.coarse_model = coarse_model
        self.fine_model = fine_model
        self.num_samples_fine = num_samples_fine
    
    def compile(self, optimizer_coarse, optimizer_fine, loss_fn):
        super().compile()
        self.optimizer_coarse = optimizer_coarse
        self.optimizer_fine = optimizer_fine

        self.loss_fn = loss_fn
        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.psnr_metric = keras.metrics.Mean(name="psnr")
        self.ssim_metric = keras.metrics.Mean(name="ssim")

    def train_step(self, inputs):
        # simplify this?
        (images, depths, rays) = inputs
        (rays_origins_coarse, rays_directions_coarse, t_vals_coarse) = rays

        # Equation: r(t) = o + td -> Building the "r" here.
        rays_coarse = rays_origins_coarse[..., None, :] + (rays_directions_coarse[..., None,:] * t_vals_coarse[..., None])
        rays_coarse = positional_encoding(rays_coarse, POS_ENCODE_DIMS_RAYS)

        # same as flatten in keras?
        directions_coarse_shape = tf.shape(rays_coarse[..., :3])
        directions_coarse = tf.broadcast_to(rays_directions_coarse[..., None, :], shape=directions_coarse_shape)
        directions_coarse = positional_encoding(directions_coarse, POS_ENCODE_DIMS_DIRS)

        with tf.GradientTape() as coarse_tape:
            (rgb_coarse, sigma_coarse) = self.coarse_model([rays_coarse, directions_coarse])
            render_coarse = render_image_depth(rgb_coarse, sigma_coarse, t_vals_coarse)
            (images_coarse, depths_coarse, weights_coarse) = render_coarse
        
            image_loss_coarse = self.loss_fn(images, images_coarse)
            depth_loss_coarse = self.loss_fn(depths, depths_coarse)

            loss_coarse = image_loss_coarse + depth_loss_coarse
            # loss_coarse = image_loss_coarse + (DEPTH_LOSS_BALANCER * depth_loss_coarse)

        t_vals_coarse_mid = (0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1]))

        t_vals_fine = hierarchical_sampling(t_vals_coarse_mid, weights_coarse, self.num_samples_fine)
        t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine], axis=-1), axis=-1) # why concat and sort?

        rays_fine = (rays_origins_coarse[..., None, :] + (rays_directions_coarse[..., None, :] * t_vals_fine[..., None]))
        rays_fine = positional_encoding(rays_fine, POS_ENCODE_DIMS_RAYS)

        directions_fine_shape = tf.shape(rays_fine[..., :3])
        directions_fine = tf.broadcast_to(rays_directions_coarse[..., None, :], shape=directions_fine_shape)
        directions_fine = positional_encoding(directions_fine, POS_ENCODE_DIMS_DIRS)
        
        with tf.GradientTape() as fine_tape:
            (rgb_fine, sigma_fine) = self.fine_model([rays_fine, directions_fine])
            render_fine = render_image_depth(rgb_fine, sigma_fine, t_vals_fine)
            (images_fine, depths_fine, weights_fine) = render_fine

            image_loss_fine = self.loss_fn(images, images_fine)
            depth_loss_fine = self.loss_fn(depths, depths_fine)

            loss_fine = image_loss_fine + depth_loss_fine
            

        trainable_variables_coarse = self.coarse_model.trainable_variables
        gradients_coarse = coarse_tape.gradient(loss_coarse, trainable_variables_coarse)
        self.optimizer_coarse.apply_gradients(zip(gradients_coarse, trainable_variables_coarse))

        trainable_variables_fine = self.fine_model.trainable_variables
        gradients_fine = fine_tape.gradient(loss_fine, trainable_variables_fine)
        self.optimizer_fine.apply_gradients(zip(gradients_fine, trainable_variables_fine))

        psnr = tf.image.psnr(images, images_fine, max_val=1.0)
        ssim = tf.image.ssim(images, images_fine, max_val=1.0)
        
        self.loss_tracker.update_state(loss_fine)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)

        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result(), "ssim": self.ssim_metric.result()}

    def test_step(self, inputs):
        # simplify this?
        (images, depths, rays) = inputs
        (rays_origins_coarse, rays_directions_coarse, t_vals_coarse) = rays

        # Equation: r(t) = o + td -> Building the "r" here.
        rays_coarse = rays_origins_coarse[..., None, :] + (rays_directions_coarse[..., None,:] * t_vals_coarse[..., None])
        rays_coarse = positional_encoding(rays_coarse, POS_ENCODE_DIMS_RAYS)

        # same as flatten in keras?
        directions_coarse_shape = tf.shape(rays_coarse[..., :3])
        directions_coarse = tf.broadcast_to(rays_directions_coarse[..., None, :], shape=directions_coarse_shape)
        directions_coarse = positional_encoding(directions_coarse, POS_ENCODE_DIMS_DIRS)

        (rgb_coarse, sigma_coarse) = self.coarse_model([rays_coarse, directions_coarse])
        render_coarse = render_image_depth(rgb_coarse, sigma_coarse, t_vals_coarse)
        (_, _, weights_coarse) = render_coarse

        t_vals_coarse_mid = (0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1]))

        t_vals_fine = hierarchical_sampling(t_vals_coarse_mid, weights_coarse, self.num_samples_fine)
        t_vals_fine = tf.sort(tf.concat([t_vals_coarse, t_vals_fine], axis=-1), axis=-1)

        rays_fine = (rays_origins_coarse[..., None, :] + (rays_directions_coarse[..., None, :] * t_vals_fine[..., None]))
        rays_fine = positional_encoding(rays_fine, POS_ENCODE_DIMS_RAYS)

        directions_fine_shape = tf.shape(rays_fine[..., :3])
        directions_fine = tf.broadcast_to(rays_directions_coarse[..., None, :], shape=directions_fine_shape)
        directions_fine = positional_encoding(directions_fine, POS_ENCODE_DIMS_DIRS)
        
        (rgb_fine, sigma_fine) = self.fine_model([rays_fine, directions_fine])
        render_fine = render_image_depth(rgb_fine, sigma_fine, t_vals_fine)
        (images_fine, depths_fine , _) = render_fine

        image_loss_fine = self.loss_fn(images, images_fine)
        depth_loss_fine = self.loss_fn(depths, depths_fine)

        loss_fine = image_loss_fine +  depth_loss_fine

        psnr = tf.image.psnr(images, images_fine, max_val=1.0)
        ssim = tf.image.ssim(images, images_fine, max_val=1.0)
        
        self.loss_tracker.update_state(loss_fine)
        self.psnr_metric.update_state(psnr)
        self.ssim_metric.update_state(ssim)

        return {"loss": self.loss_tracker.result(), "psnr": self.psnr_metric.result(), "ssim": self.ssim_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.psnr_metric]

def get_train_monitor(train_ds):
    loss_list = []

    (test_images, test_depths, test_rays) = next(iter(train_ds))
    (test_rays_origins_coarse, test_rays_directions_coarse, test_t_vals_coarse) = test_rays

    test_rays_coarse = (test_rays_origins_coarse[..., None, :] + 
    	(test_rays_directions_coarse[..., None, :] * test_t_vals_coarse[..., None]))

    test_rays_coarse = positional_encoding(test_rays_coarse, POS_ENCODE_DIMS_RAYS)
    test_dirs_coarse_shape = tf.shape(test_rays_coarse[..., :3])
    test_dirs_coarse = tf.broadcast_to(test_rays_directions_coarse[..., None, :],
    	shape=test_dirs_coarse_shape)
    test_dirs_coarse = positional_encoding(test_dirs_coarse, POS_ENCODE_DIMS_DIRS)
        
    class TrainMonitor(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = logs["loss"]
            loss_list.append(loss)          
            (test_rgb_coarse, test_sigma_coarse) = self.model.coarse_model.predict(
            	[test_rays_coarse, test_dirs_coarse])

            test_render_coarse = render_image_depth(test_rgb_coarse, test_sigma_coarse, test_t_vals_coarse)
            (test_image_coarse, _, test_weights_coarse) = test_render_coarse

            test_t_vals_coarse_mid = (0.5 * 
            	(test_t_vals_coarse[..., 1:] + test_t_vals_coarse[..., :-1]))

            test_t_vals_fine = hierarchical_sampling(test_t_vals_coarse_mid, test_weights_coarse, self.model.num_samples_fine)
            test_t_vals_fine = tf.sort(
            	tf.concat([test_t_vals_coarse, test_t_vals_fine], axis=-1),
            	axis=-1)

            test_rays_fine = (test_rays_origins_coarse[..., None, :] + 
            	(test_rays_directions_coarse[..., None, :] * test_t_vals_fine[..., None])
            )
            test_rays_fine = positional_encoding(test_rays_fine, POS_ENCODE_DIMS_RAYS)

            test_dirs_fine_shape = tf.shape(test_rays_fine[..., :3])
            test_dirs_fine = tf.broadcast_to(test_rays_directions_coarse[..., None, :],
            	shape=test_dirs_fine_shape)
            test_dirs_fine = positional_encoding(test_dirs_fine, POS_ENCODE_DIMS_DIRS)

            test_rgb_fine, test_sigma_fine = self.model.fine_model.predict(
            	[test_rays_fine, test_dirs_fine])

            test_render_fine = render_image_depth(test_rgb_fine, test_sigma_fine, test_t_vals_fine)
            (test_image_fine, test_depth_fine, _) = test_render_fine

            (_, ax) = plt.subplots(nrows=1, ncols=6, figsize=(20, 5))

            ax[0].imshow(keras.preprocessing.image.array_to_img(test_image_coarse[0]))
            ax[0].set_title(f"Coarse Image")
            
            ax[1].imshow(keras.preprocessing.image.array_to_img(test_image_fine[0]))
            ax[1].set_title(f"Fine Image")
            
            ax[2].imshow(keras.preprocessing.image.array_to_img(test_images[0]))
            ax[2].set_title(f"Real Image")

            ax[3].imshow(keras.preprocessing.image.array_to_img(test_depth_fine[0, ..., None]), 
            	cmap="inferno")
            ax[3].set_title(f"Fine Depth Image")

            ax[4].imshow(keras.preprocessing.image.array_to_img(test_depths[0, ..., None]), 
            	cmap="inferno")
            ax[4].set_title(f"Real Depth Image")

            ax[5].plot(loss_list)
            ax[5].set_xticks(np.arange(0, EPOCHS + 1, 5.0))
            ax[5].set_title(f"Loss Plot: {epoch:03d}")

            plt.savefig(f"./result/images/{epoch:03d}.png")
            plt.close()
	
    
    trainMonitor = TrainMonitor()
	
    return trainMonitor

def create_gif(path_to_images, name_gif):
    depth_filenames = glob.glob(path_to_images)
    depth_filenames = sorted(depth_filenames)
    images = []
    for depth_filename in tqdm(depth_filenames):
        images.append(imageio.imread(depth_filename))
    kargs = {"duration": 0.25}
    imageio.mimsave(name_gif, images, "GIF", **kargs)

def train():
    # Split the images into training and validation.
    train_images = images[:split_index]
    val_images = images[split_index:]

    train_depths = depths[:split_index]
    val_depths = depths[split_index:]


    # Split the poses into training and validation.
    train_poses = poses[:split_index]
    val_poses = poses[split_index:]

    print(train_images.shape)
    print(val_images.shape)

    print(train_poses.shape)
    print(val_poses.shape)


    # Make the training pipeline.
    train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_depth_ds = tf.data.Dataset.from_tensor_slices(train_depths)
    train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
    train_ray_ds = train_pose_ds.map(get_rays, num_parallel_calls=AUTO)
    training_ds = tf.data.Dataset.zip((train_img_ds, train_depth_ds ,train_ray_ds))
    train_ds = (
        training_ds.shuffle(BATCH_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    # Make the validation pipeline.
    val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
    val_depth_ds = tf.data.Dataset.from_tensor_slices(val_depths)
    val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
    val_ray_ds = val_pose_ds.map(get_rays, num_parallel_calls=AUTO)
    validation_ds = tf.data.Dataset.zip((val_img_ds, val_depth_ds, val_ray_ds))
    val_ds = (
        validation_ds.shuffle(BATCH_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=AUTO)
        .prefetch(AUTO)
    )

    print(train_ds.element_spec)
    print(val_ds.element_spec)


    # (ray_origins, ray_directions, t_vals) = get_rays(poses[0])
    # (image, depth, weights) = render_image_depth(rgb, sigma, t_vals)

    # instantiate the coarse model
    coarse_model = get_model(
        num_layers=NUM_LAYERS,
        dense_units=DENSE_UNITS,
    )

    coarse_model.build((None, None, None, 2 * 3 * POS_ENCODE_DIMS_RAYS + 3, BATCH_SIZE))
    coarse_model.summary()

    # instantiate the fine model
    fine_model = get_model(
        num_layers=NUM_LAYERS,
        dense_units=DENSE_UNITS,
    )

    # instantiate the nerf trainer model
    nerf_trainer_model = NerfTrainer(
        coarse_model=coarse_model, 
        fine_model=fine_model,
        num_samples_fine=NUM_SAMPLES
    )

    # compile the nerf trainer model with Adam optimizer and MSE loss
    nerf_trainer_model.compile(
        optimizer_coarse=keras.optimizers.Adam(),
        optimizer_fine=keras.optimizers.Adam(),
        loss_fn=keras.losses.MeanSquaredLogarithmicError ()
    )

    if not os.path.exists("./result"):
        os.makedirs("./result")

    with open("./result/model_summary.txt", "w") as f:
        with redirect_stdout(f):
            coarse_model.summary()

    if not os.path.exists("./result/images"):
        os.makedirs("./result/images")

    train_monitor = get_train_monitor(train_ds)

    nerf_trainer_model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[train_monitor],
        steps_per_epoch=split_index//BATCH_SIZE,
    )

    create_gif("./result/images/*.png", "./result/training.gif")

train()