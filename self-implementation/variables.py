import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

# Initialize global variables.
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 5
NUM_SAMPLES = 32
POS_ENCODE_DIMS = 16
EPOCHS = 20
RANDOM_SEED = 42
