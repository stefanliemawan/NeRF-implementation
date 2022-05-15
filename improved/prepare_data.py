import tensorflow as tf

import variables as var


def encode_position(x):
    """Encodes the position into its corresponding Fourier feature.

    Args:
        x: The input coordinate.

    Returns:
        Fourier features tensors of the position.
    """
    positions = [x]
    for i in range(var.POS_ENCODE_DIMS):
        for fn in [tf.sin, tf.cos]:
            positions.append(fn(2.0 ** i * x))
    return tf.concat(positions, axis=-1)


def get_rays(height, width, focal, pose):
    """Computes origin point and direction vector of rays.

    Args:
        height: Height of the image.
        width: Width of the image.
        focal: The focal length between the images and the camera.
        pose: The pose matrix of the camera.

    Returns:
        Tuple of origin point and direction vector for rays.
    """
    # Build a meshgrid for the rays, for the image pixels
    i, j = tf.meshgrid(
        tf.range(width, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        indexing="xy",
    )

    # Normalize the x axis coordinates.
    transformed_i = (i - width * 0.5) / focal

    # Normalize the y axis coordinates.
    transformed_j = (j - height * 0.5) / focal

    # Create the direction unit vectors.
    directions = tf.stack([transformed_i, -transformed_j, -tf.ones_like(i)], axis=-1)
    
    # Get the camera matrix
    camera_matrix = pose[:3, :3]
    height_width_focal = pose[:3, -1]

    # Get origins and directions for the rays.
    transformed_dirs = directions[..., None, :]
    camera_dirs = transformed_dirs * camera_matrix
    ray_directions = tf.reduce_sum(camera_dirs, axis=-1)
    ray_origins = tf.broadcast_to(height_width_focal, tf.shape(ray_directions))

    # Return the origins and directions.
    return (ray_origins, ray_directions)


def render_flat_rays(ray_origins, ray_directions, near, far, num_samples, rand=False):
    """Renders the rays and flattens it.

    Args:
        ray_origins: The origin points for rays.
        ray_directions: The direction unit vectors for the rays.
        near: The near bound of the volumetric scene.
        far: The far bound of the volumetric scene.
        num_samples: Number of sample points in a ray.
        rand: Choice for randomising the sampling strategy.

    Returns:
       Tuple of flattened rays and sample points on each rays.
    """
    # Compute 3D query points.
    # Equation: r(t) = o+td -> Building the "t" here.
    t_vals = tf.linspace(near, far, num_samples)
    if rand:
        # Inject uniform noise into sample space to make the sampling
        # continuous.
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = tf.random.uniform(shape=shape) * (far - near) / num_samples
        t_vals = t_vals + noise

    # Equation: r(t) = o + td -> Building the "r" here.
    rays = ray_origins[..., None, :] + (
        ray_directions[..., None, :] * t_vals[..., None]
    )
    rays_flat = tf.reshape(rays, [-1, 3])
    rays_flat = encode_position(rays_flat)
    return (rays_flat, t_vals)


def map_fn(pose):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: The pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the
        camera pose.
    """
    (ray_origins, ray_directions) = get_rays(height=var.H, width=var.W, focal=var.focal, pose=pose)

    print(pose)
    print(ray_origins)
    print(ray_directions)
    
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=var.NUM_SAMPLES,
        rand=True,
    )

    print(rays_flat)
    print(t_vals)
    
    return (rays_flat, t_vals)


def train_val_split():

    # Split the images into training and validation.
    train_images = var.images[:var.split_index]
    val_images = var.images[var.split_index:]

    # Split the poses into training and validation.
    train_poses = var.poses[:var.split_index]
    val_poses = var.poses[var.split_index:]

    # Make the training pipeline.
    train_img_ds = tf.data.Dataset.from_tensor_slices(train_images)
    train_pose_ds = tf.data.Dataset.from_tensor_slices(train_poses)
    train_ray_ds = train_pose_ds.map(map_fn, num_parallel_calls=var.AUTO)
    training_ds = tf.data.Dataset.zip((train_img_ds, train_ray_ds))
    train_ds = (
        training_ds.shuffle(var.BATCH_SIZE)
        .batch(var.BATCH_SIZE, drop_remainder=True, num_parallel_calls=var.AUTO)
        .prefetch(var.AUTO)
    )

    # Make the validation pipeline.
    val_img_ds = tf.data.Dataset.from_tensor_slices(val_images)
    val_pose_ds = tf.data.Dataset.from_tensor_slices(val_poses)
    val_ray_ds = val_pose_ds.map(map_fn, num_parallel_calls=var.AUTO)
    validation_ds = tf.data.Dataset.zip((val_img_ds, val_ray_ds))
    val_ds = (
        validation_ds.shuffle(var.BATCH_SIZE)
        .batch(var.BATCH_SIZE, drop_remainder=True, num_parallel_calls=var.AUTO)
        .prefetch(var.AUTO)
    )

    # return ray_ds only?

    return train_ds, val_ds