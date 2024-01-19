import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses
from keras.layers import BatchNormalization, MultiHeadAttention
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import (
    AveragePooling2D,
    Convolution1D,
    Convolution2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    Flatten,
    Dropout,
)


def cnn_1d_model(input_shape, name_suffix=None, dropoutRate=0.2):
    tensor_shape = (input_shape[0], input_shape[1], 1)
    name = "cnn_1d_model" if not name_suffix else f"cnn_1d_model{name_suffix}"

    model = models.Sequential(
        [
            layers.Conv1D(
                4,
                32,
                input_shape=tensor_shape,
                strides=1,
                kernel_initializer="he_normal",
                activation="relu"
            ),
            layers.MaxPool2D((1, 4)),
            layers.Dropout(dropoutRate, noise_shape=None),


            layers.Conv1D(4, 4, strides=1, activation="relu"),
            layers.MaxPool2D((1, 4)),
            layers.Dropout(dropoutRate, noise_shape=None),


            layers.Conv1D(
                4, 4, strides=1, activation="relu"
            ),
            layers.MaxPool2D((1, 4)),
            layers.Dropout(dropoutRate, noise_shape=None),


            layers.Flatten(),
            layers.Dense(4, activation="softmax"),
        ],
        name=name,
    )

    model.compile(
        optimizer="adam", loss=losses.CategoricalCrossentropy(), metrics=["accuracy"]
    )

    return model
