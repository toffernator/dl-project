import tensorflow as tf
keras = tf.keras
from keras import models, layers, losses
from keras.layers import BatchNormalization, MultiHeadAttention

def cnn_model(input_shape, name_suffix=None, dropoutRate=0.5):
    tensor_shape = (input_shape[0], input_shape[1], 1)
    name = "cnn_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential([
        layers.Conv2D(4, (4, 4), activation="relu", input_shape=tensor_shape, strides=(2, 1)),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(dropoutRate),

        # layers.Reshape((-1, 4)),  # Reshaping for MultiHeadAttention
        # MultiHeadAttention(num_heads=2, key_dim=2),  # Self-attention layer
        # layers.Reshape((input_shape[0] // 2, input_shape[1] // 4, 4)),  # Reshaping back

        layers.Conv1D(4, 4, activation="relu", strides=1),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(dropoutRate),

        layers.Conv2D(4, (4, 4), activation="relu", strides=(2, 1)),
        layers.AveragePooling2D((1, 4)),
        layers.Dropout(dropoutRate),

        layers.Flatten(),
        layers.Dense(4, activation="softmax")
    ], name=name)

    model.compile(optimizer="adam",loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

    return model