import numpy as np
import tensorflow as tf
from tqdm import tqdm
import imageio

import variables as var
from model import render_rgb_depth
from prepare_data import get_rays, render_flat_rays

def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, tf.cos(phi), -tf.sin(phi), 0],
        [0, tf.sin(phi), tf.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [tf.cos(theta), 0, -tf.sin(theta), 0],
        [0, 1, 0, 0],
        [tf.sin(theta), 0, tf.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return tf.convert_to_tensor(matrix, dtype=tf.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

def render(nerf_model):
    rgb_frames = []
    batch_flat = []
    batch_t = []

    # Iterate over different theta value and generate scenes.
    for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, 120, endpoint=False))):
        # Get the camera to world matrix.
        c2w = pose_spherical(theta, -30.0, 4.0)

        #
        ray_oris, ray_dirs = get_rays(var.H, var.W, var.focal, c2w)
        rays_flat, t_vals = render_flat_rays(
            ray_oris, ray_dirs, near=2.0, far=6.0, num_samples=var.NUM_SAMPLES, rand=False
        )

        if index % var.BATCH_SIZE == 0 and index > 0:
            batched_flat = tf.stack(batch_flat, axis=0)
            batch_flat = [rays_flat]

            batched_t = tf.stack(batch_t, axis=0)
            batch_t = [t_vals]

            rgb, _ = render_rgb_depth(
                nerf_model, batched_flat, batched_t, rand=False, train=False
            )

            temp_rgb = [np.clip(255 * img, 0.0, 255.0).astype(np.uint8) for img in rgb]

            rgb_frames = rgb_frames + temp_rgb
        else:
            batch_flat.append(rays_flat)
            batch_t.append(t_vals)

    rgb_video = "./result/rgb_video.mp4"
    imageio.mimwrite(rgb_video, rgb_frames, fps=30, quality=7, macro_block_size=None)