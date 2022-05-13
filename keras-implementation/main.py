from tensorflow import keras
import matplotlib.pyplot as plt

from load_prepare_data import train_val_split
from model import create_model
import variables as var


def main():
    train_ds, val_ds = train_val_split()

    # print(train_ds.element_spec)
    # print(val_ds.element_spec)

    num_pos = var.H * var.W * var.NUM_SAMPLES

    # model = create_model(num_pos)
    
    # model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    # model.summary()


    
    # model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     batch_size=var.BATCH_SIZE,
    #     epochs=var.EPOCHS,
    #     steps_per_epoch=var.BATCH_SIZE,
    # )


main()