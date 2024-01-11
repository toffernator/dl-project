from pathlib import Path
from preprocessing.minMaxScaling import MinMaxScalerBatched
from downsampling import Downsampler

import utils as utils
import dataset_loader as dsloader
from tcn import TCN, tcn_full_summary
from tensorflow import keras
import tensorflow as tf
import numpy  as np
import h5py


def main():
    train, test = dsloader.get_intra_dataset_files()

    # first train data in Intra: "dataset/Intra/train/rest_105923_1.h5"
    matrix = train[0].load()

    scaler = MinMaxScalerBatched()
    scaler.applyMinMaxScaling(matrix)
    print(matrix)
    # reduced = Downsampler().chunk_and_average_columns(matrix, 2)    # Chunk size = 2 
    # print(reduced)
    # print(reduced.shape)

    matrix = Downsampler().decimate(matrix, 5)
    testM = Downsampler().decimate(testM, 5)

    n_features = matrix.shape[0]

    print(matrix.shape)

    # train_data = matrix.reshape(-1, 80, n_features)
    # test_data  = testM.reshape(-1, 80, n_features) # TODO

    X_train = matrix.T
    y_train = np.array([ np.array([1, 0, 0, 0]) for _ in X_train])

    n_epochs = 50
    n_splits =  5

    # scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 200*((len(testM)*0.8)/1024), 1e-5)

    model = keras.models.Sequential([
        TCN(input_shape=(7125, n_features), nb_filters=256, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32]),
        keras.layers.Dense(4)
    ])

    model.compile(optimizer="adam", loss="mae", metrics=keras.metrics.MeanAbsoluteError())

    history = model.fit(X_train, y_train, 
                        # validation_data=(X_valid, y_valid), 
                        epochs=n_epochs, 
                        batch_size=1024 
                        # callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)]
                        )


if __name__ == "__main__":
    main()
