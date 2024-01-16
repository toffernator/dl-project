import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses

def lstm_model(input_shape, name_suffix=None):
    tensor_shape = (input_shape[0], input_shape[1])

    name = "lstm_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential(
        [
            layers.LSTM(32, input_shape=tensor_shape),
            layers.Dense(4),
        ],
        name=name,
    )

    model.compile(
        optimizer="adam",
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model
