import tensorflow as tf
import math

from variables import *

def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.


    y = tf.exp(-0.5 * x_var) * tf.sin(x)
    y_var = tf.maximum(
        tf.cast(0, tf.float32), 0.5 * (1 - tf.exp(-2 * x_var) * tf.cos(2 * x)) - y**2)
    
    return y, y_var

def integrated_pos_encoding(x_coord, min_deg=0, max_deg=POS_ENCODE_DIMS, diag=True):
    """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        x_coord: a tuple containing: x, tf.ndarray, variables to be encoded. Should
        be in [-pi, pi]. x_cov, tf.ndarray, covariance matrices for `x`.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diag: bool, if true, expects input covariances to be diagonal (full
        otherwise).
    Returns:
        encoded: tf.ndarray, encoded variables.
    """
    if diag:
        x, x_cov_diag = x_coord
        scales = np.array([2**i for i in range(min_deg, max_deg)])
        shape = list(x.shape[:-1]) + [-1]
        y = tf.reshape(x[..., None, :] * scales[:, None], shape)
        y_var = tf.reshape(x_cov_diag[..., None, :] * scales[:, None]**2, shape)
    else:
        x, x_cov = x_coord
        num_dims = x.shape[-1]
        basis = tf.concat(
            [2**i * tf.eye(num_dims) for i in range(min_deg, max_deg)], 1)
        y = tf.linalg.matmul(x, basis)
        y_var = tf.reduce_sum((tf.linalg.matmul(x_cov, basis)) * basis, -2)

    return expected_sin(
        tf.concat([y, y + 0.5 * math.pi], axis=-1),
        tf.concat([y_var] * 2, axis=-1))[0]

def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = tf.maximum(1e-10, tf.reduce_sum(d**2, axis=-1, keepdims=True))

    if diag:
        d_outer_diag = d**2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = tf.eye(d.shape[-1])
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov

def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.
    Args:
        d: tf.float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                        (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.
    Args:
        d: tf.float32 3-vector, the axis of the cylinder
        t0: float, the starting distance of the cylinder.
        t1: float, the ending distance of the cylinder.
        radius: float, the radius of the cylinder
        diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    Returns:
        a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius**2 / 4
    t_var = (t1 - t0)**2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)

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

    return (ray_origins, ray_directions)

def cast_cones(pose, lindisp=True, randomized=True, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
    Args:
        t_vals: float array, the "fencepost" distances along the ray.
        ray_origins: float array, the ray origin coordinates.
        ray_directions: float array, the ray direction vectors.
        RADIUS: float array, the RADIUS (base RADIUS for cones) of the rays.
        RAY_SHAPE: string, the shape of the ray, must be 'cone' or 'cylinder'.
        lindisp: bool, sampling linearly in disparity rather than depth.
        diag: boolean, whether or not the covariance matrices should be diagonal.
    Returns:
        a tuple of arrays of means and covariances.
    """

    (ray_origins, ray_directions) = get_rays(pose)
    
    t_vals = tf.linspace(0., 1., NUM_SAMPLES + 1)
    if lindisp:
        t_vals = 1. / (1. / NEAR * (1. - t_vals) + 1. / FAR * t_vals)
    else:
        t_vals = NEAR * (1. - t_vals) + FAR * t_vals

    # if randomized:
    #     mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
    #     upper = tf.concat([mids, t_vals[..., -1:]], -1)
    #     lower = tf.concat([t_vals[..., :1], mids], -1)
    #     t_rand = tf.random.uniform([2,], [BATCH_SIZE, NUM_SAMPLES + 1])
    #     t_vals = lower + (upper - lower) * t_rand
    # else:
    # Broadcast t_vals to make the returned shape consistent.
    t_vals = tf.broadcast_to(t_vals, [BATCH_SIZE, NUM_SAMPLES + 1])

    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]

    if RAY_SHAPE == "CONE":
        gaussian_fn = conical_frustum_to_gaussian
    elif RAY_SHAPE == "CYLINDER":
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    
    means, covs = gaussian_fn(ray_directions, t0, t1, RADIUS, diag)
    means = means + ray_origins[..., None, :]

    samples = integrated_pos_encoding((means, covs), 0, POS_ENCODE_DIMS)

    return (t_vals, samples)
