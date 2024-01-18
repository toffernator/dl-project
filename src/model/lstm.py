import tensorflow as tf

keras = tf.keras
from keras import models, layers, losses
from keras.layers import ( LSTM, Dense)

def lstm_model(input_shape, name_suffix=None):
    tensor_shape = (input_shape[0], input_shape[1])

    name = "lstm_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape = input_shape))
    # model.add(SeqSelfAttention(attention_activation ='tanh')),
    model.add(LSTM(units=64, return_sequences=True))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dense(32))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

    return model
