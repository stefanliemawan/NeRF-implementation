import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf

from load_prepare_data import train_val_split



def main():
    train_ds, val_ds = train_val_split()

    print(train_ds.element_spec)
    print(val_ds.element_spec)


main()