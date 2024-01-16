import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses, regularizers


def cnn_model(input_shape, name_suffix=None, dropout_rate=0.4, output_drouput=0.2):
    tensor_shape = (input_shape[0], input_shape[1], 1)

    name = "cnn_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential(
        [
            layers.Conv2D(
                5,
                (5, 5),
                activation="relu",
                input_shape=tensor_shape,
                kernel_initializer="he_normal",
            ),
            layers.MaxPool2D((2, 2)),
            layers.Dropout(rate=dropout_rate),
            # layers.Conv1D,
            layers.Conv2D(
                5,
                (5, 5),
                activation="relu",
                kernel_regularizer=regularizers.l2(l=0.01),
            ),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(rate=output_drouput),
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
