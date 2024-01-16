import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses


def cnn_model(input_shape, name_suffix=None):
    tensor_shape = (input_shape[0], input_shape[1], 1)

    name = "cnn_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential(
        [
            layers.Conv2D(4, (5, 5), activation="relu", input_shape=tensor_shape),
            layers.MaxPool2D((1, 5)),
            layers.Conv2D(4, (5, 5), activation="relu"),
            layers.MaxPool2D((1, 5)),
            layers.Conv2D(4, (5, 5), activation="relu"),
            layers.MaxPool2D((1, 5)),
            layers.Flatten(),
            layers.Dense(4, activation="softmax"),
        ],
        name=name,
    )

    model.compile(
        optimizer="adam",
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model
