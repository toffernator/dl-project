from pathlib import Path
from enum import Enum
from src.preprocessing.minMaxScaling import MinMaxScalerBatched
from src.downsampling import Downsampler

import src.utils as utils
from src.dataset_loader import (
    get_intra_dataset_files,
    get_cross_dataset_files,
    DatasetFile,
)
import h5py

import numpy as np

# import matplotlib.pyplot as plt
# from sklearn.model_selection import KFold
from tensorflow import keras
import tensorflow as tf

# from tcn import TCN, tcn_full_summary
# from src.model.lstm import lstm_model
from src.model.cnn import cnn_model
from src.model.cnn_attention import cnn_model_attention
# from src.model.rnn import model_RNN
# from src.model.tcn import tcn_model
# from src.model.EEGNet import eeg_model

from src.trainer import train_eval

class NN(Enum): 
    CNN = 1 
    LSTM = 2
    EEGNet= 3
# DOWNSAMPLE_FACTOR = 5
# INPUT_SHAPE = (248, 7125)

DOWNSAMPLE_FACTOR = 10
INPUT_SHAPE = (248, 3563)

TRAIN_EPOCHS = 30
BATCH_SIZE = 8
NETWORK = NN.CNN 

def run_preprocess(train, test):
    scaler = MinMaxScalerBatched()
    sampler = Downsampler()

    def preprocess(file):
        matrix = file.load()
        scaler.applyMinMaxScaling(matrix)
        matrix = sampler.decimate(matrix, DOWNSAMPLE_FACTOR)
        print(matrix.shape)
        file.save_preprocessed(matrix)

    for file in train:
        preprocess(file)

    for file in test:
        preprocess(file)


def main():
    # train, test = get_intra_dataset_files()
    train, test = get_cross_dataset_files()

    if not train[0].preprocessed:
        print("run preprocessing...")
        run_preprocess(train, test)

    # if(NETWORK == NN.CNN):
    #     model = cnn_model(INPUT_SHAPE, "intra")
    
    # elif(NETWORK == NN.LSTM):
    #     model = lstm_model(INPUT_SHAPE, "intra")
    
    # elif(NETWORK == NN.EEGNet):
    #     model = eeg_model(INPUT_SHAPE)
    
    # model = model_RNN(INPUT_SHAPE)
    # model = eeg_model(INPUT_SHAPE)
    # model = tcn_model(INPUT_SHAPE, BATCH_SIZE, "intra")
    model = cnn_model_attention(INPUT_SHAPE, "intra")
        
    train_eval(model, TRAIN_EPOCHS, BATCH_SIZE, train, test)

    # # first train data in Intra: "dataset/Intra/train/rest_105923_1.h5"
    # matrix = train[0].load()
    # testM = test[0].load()

    # scaler = MinMaxScalerBatched()
    # scaler.applyMinMaxScaling(matrix)
    # scaler.applyMinMaxScaling(testM)
    # print(matrix)
    # # reduced = Downsampler().chunk_and_average_columns(matrix, 2)    # Chunk size = 2
    # # print(reduced)
    # # print(reduced.shape)

    # matrix = Downsampler().decimate(matrix, 5)
    # testM = Downsampler().decimate(testM, 5)

    # n_features = matrix.shape[-1]

    # # train_data = matrix.reshape(-1, 80, n_features)
    # # test_data  = testM.reshape(-1, 80, n_features) # TODO

    # X_train = matrix
    # y_train = np.array([ np.array([1, 0, 0, 0]) for _ in matrix])

    # n_epochs = 50
    # n_splits =  5

    # scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 200*((len(testM)*0.8)/1024), 1e-5)

    # model = keras.models.Sequential([
    #     TCN(input_shape=(80, n_features), nb_filters=256, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32]),
    #     keras.layers.Dense(1)
    # ])

    # model.compile(optimizer="adam", loss="mae", metrics=keras.metrics.MeanAbsoluteError())

    # history = model.fit(X_train, y_train,
    #                     # validation_data=(X_valid, y_valid),
    #                     epochs=n_epochs,
    #                     batch_size=1024,
    #                     callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])


if __name__ == "__main__":
    main()
