import tensorflow as tf
keras = tf.keras
from keras import models, layers, losses
from keras.layers import BatchNormalization, MultiHeadAttention
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import (AveragePooling2D,Convolution1D,Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout)

def cnn_model(input_shape, name_suffix=None, dropoutRate=0.2):
    tensor_shape = (input_shape[0], input_shape[1], 1)
    name = "cnn_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential([

        Input(shape=(input_shape[0],input_shape[1],1)) ,
        Convolution2D(4, (4, 4), strides=(2, 1), padding="same"), 
        BatchNormalization(),
        Activation("relu"),
        AveragePooling2D((1, 4)),
        Dropout(dropoutRate),

        # layers.Conv2D(4, (4, 4), activation="relu", input_shape=tensor_shape, strides=(2, 1)),
        # layers.AveragePooling2D((1, 4)),
        # layers.Dropout(dropoutRate),

        Convolution1D(4, 4, strides=1),
        BatchNormalization(),
        Activation("relu"),
        AveragePooling2D((1, 4)),
        Dropout(dropoutRate),

        Convolution2D(4, (4, 4), strides=(2, 1), padding="same"), 
        BatchNormalization(),
        Activation("relu"),
        AveragePooling2D((1, 4)),
        Dropout(dropoutRate),

        # layers.Reshape((-1, 4)),

        # layers.LSTM(4),

        layers.Flatten(),
        layers.Dense(4, activation="softmax")
    ], name=name)

    model.compile(optimizer="adam",loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

    return model