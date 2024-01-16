import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses

def lstm_model(input_shape, name_suffix=None):
    tensor_shape = (input_shape[0], input_shape[1])

    name = "lstm_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential()

    model.add( layers.Embedding(
            input_dim=tensor_shape,
            # output_dim=embedding_size,
            # input_length=max_length,
            trainable=False,
            mask_zero=True,
            # weights=[embeddings]
        ))

    model.add(layers.LSTM(200, return_sequences=False))
    model.add(layers.Activation('softmax')) #this guy here
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(3, activation='softmax', activity_regularizer=activity_l2(0.0001)))

    model.compile(
        optimizer="adam",
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model
