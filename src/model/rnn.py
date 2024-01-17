
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Sequential
from keras import models, layers, losses
from keras.layers import Embedding, SimpleRNN

hidden_units = 150
dense_units = 4

def model_RNN( input_shape ):

    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation="tanh"))
    model.add(Dense(units=dense_units, activation='softmax'))
    model.compile(optimizer="adam",loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])
    return model
 