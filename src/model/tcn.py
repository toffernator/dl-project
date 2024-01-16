import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses, metrics
from tcn import TCN

def tcn_model(input_shape, batch_size, name_suffix=None):
    tensor_shape = (input_shape[0], input_shape[1], 1)

    name = "tcn_model" if not name_suffix else f"tcn_model_{name_suffix}"

    model = keras.models.Sequential([
        TCN(input_shape=(batch_size, input_shape[1], input_shape[0]), nb_filters=32, return_sequences=True, dilations=[1, 2, 4, 8, 16, 32]),
        keras.layers.Dense(1)
    ], name = name
    )

    model.compile(
        optimizer="adam",
        loss="mae",
        metrics=keras.metrics.MeanAbsoluteError(),
    )

    return model

#def tcn_model(input_shape, name_suffix=None):
    #tensor_shape = (input_shape[0], input_shape[1], 1)

    #name = "tcn_model" if not name_suffix else f"tcn_model_{name_suffix}"


    #model = tf.keras.Sequential()
    #model.add(tf.keras.layers.InputLayer(input_shape=[None, 5])) #this has to change i guess.
    #for rate in (1, 2, 4, 8):
        #model.add(tf.keras.layers.Conv1D(
            #filters=32, kernel_size=2, padding="causal", activation="relu",
            #dilation_rate=rate))
    #model.add(tf.keras.layers.Dense(4, activation="softmax"))

    
    #model.compile(
        #optimizer="adam",
        #loss="mae",
        #metrics=keras.metrics.MeanAbsoluteError(),
    #)

    #return model
