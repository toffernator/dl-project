import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses, optimizers, initializers
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


def cnn_model(input_shape, name_suffix=None, dropoutRate=0.5):
    tensor_shape = (input_shape[0], input_shape[1], 1)
    name = "cnn_model" if not name_suffix else f"cnn_model_{name_suffix}"

    # model = models.Sequential(
    #     [
    #         Input(shape=(input_shape[0], input_shape[1], 1)),
    #         Convolution2D(4, (4, 4), strides=(2, 1), padding="same"),
    #         BatchNormalization(),
    #         Activation("relu"),
    #         AveragePooling2D((1, 4)),
    #         Dropout(dropoutRate),
    #         # layers.Conv2D(4, (4, 4), activation="relu", input_shape=tensor_shape, strides=(2, 1)),
    #         # layers.AveragePooling2D((1, 4)),
    #         # layers.Dropout(dropoutRate),
    #         Convolution1D(4, 4, strides=1),
    #         BatchNormalization(),
    #         Activation("relu"),
    #         AveragePooling2D((1, 4)),
    #         Dropout(dropoutRate),
    #         Convolution2D(4, (4, 4), strides=(2, 1), padding="same"),
    #         BatchNormalization(),
    #         Activation("relu"),
    #         AveragePooling2D((1, 4)),
    #         Dropout(dropoutRate),
    #         # layers.Reshape((-1, 4)),
    #         # layers.LSTM(4),
    #         layers.Flatten(),
    #         layers.Dense(4, activation="softmax"),
    #     ],
    #     name=name,
    # )

    # model = models.Sequential(
    #     [
    #         layers.Conv2D(
    #             4,
    #             (4, 4),
    #             activation="relu",
    #             input_shape=tensor_shape,
    #             strides=(2, 1),
    #             kernel_initializer="he_normal",
    #         ),
    #         layers.MaxPool2D((1, 4)),
    #         layers.Dropout(dropoutRate, noise_shape=None, seed=None),
    #         # BatchNormalization(),
    #         layers.Conv1D(4, 4, activation="relu", strides=1),
    #         layers.MaxPool2D((1, 4)),
    #         layers.Dropout(dropoutRate, noise_shape=None, seed=None),
    #         # BatchNormalization(),
    #         layers.Conv2D(
    #             4, (4, 4), activation="relu", input_shape=tensor_shape, strides=(2, 1)
    #         ),
    #         layers.MaxPool2D((1, 4)),
    #         layers.Dropout(dropoutRate, noise_shape=None, seed=None),
    #         layers.Flatten(),
    #         # BatchNormalization(),
    #         layers.Dense(4, activation="softmax"),
    #     ],
    #     name=name,
    # )

    init = initializers.he_uniform()

    model = models.Sequential(
        [
            layers.Input(shape=tensor_shape),
            layers.Conv2D(
                4,
                (5, 5),
                strides=(1, 1),
                kernel_initializer=init,
                padding="same",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D((1, 4)),
            # layers.Dropout(dropoutRate, noise_shape=None),
            layers.Conv1D(
                4,
                5,
                strides=1,
                kernel_initializer=init,
                padding="same",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D((1, 4)),
            # layers.Dropout(dropoutRate, noise_shape=None),
            layers.Conv2D(
                4,
                (5, 5),
                input_shape=tensor_shape,
                strides=(1, 1),
                kernel_initializer=init,
                padding="same",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(dropoutRate, noise_shape=None),
            layers.Dense(
                4,
                activation="softmax",
                # kernel_regularizer="l2",
                kernel_initializer=init,
            ),
        ],
        name=name,
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model
