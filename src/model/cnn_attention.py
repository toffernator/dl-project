import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Embedding, LSTM, Reshape, Bidirectional
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Conv2D, MaxPooling2D
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense, TimeDistributed)
import tensorflow as tf


import tensorflow as tf
keras = tf.keras
from keras import models, layers, losses
from keras.layers import BatchNormalization, MultiHeadAttention
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import (AveragePooling2D,Convolution1D,Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout)
from keras_self_attention import SeqSelfAttention

def cnn_model_attention(input_shape, name_suffix=None, dropoutRate=0.2):
    tensor_shape = (input_shape[0], input_shape[1], 1)
    name = "cnn_model" if not name_suffix else f"cnn_model_{name_suffix}"

    model = models.Sequential([

        Input(shape=(input_shape[0],input_shape[1],1)) ,
        Convolution2D(4, (4, 4), strides=(2, 1), padding="same"), 
        # BatchNormalization(),
        Activation("relu"),
        AveragePooling2D((1, 4)),
        Dropout(dropoutRate),

        # layers.Conv2D(4, (4, 4), activation="relu", input_shape=tensor_shape, strides=(2, 1)),
        # layers.AveragePooling2D((1, 4)),
        # layers.Dropout(dropoutRate),

        Convolution1D(4, 4, strides=1),
        # BatchNormalization(),
        Activation("relu"),
        AveragePooling2D((1, 4)),
        Dropout(dropoutRate),

        Convolution2D(4, (4, 4), strides=(2, 1), padding="same"), 
        # BatchNormalization(),
        Activation("relu"),
        AveragePooling2D((1, 4)),
        Dropout(dropoutRate),

        layers.Reshape((-1, 4)),

        layers.LSTM(2, return_sequences=True),
        SeqSelfAttention(attention_activation ='tanh'),
        layers.LSTM(2, return_sequences=False),

        # layers.Flatten(),
        layers.Dense(4, activation="softmax")
    ], name=name)

    model.compile(optimizer="adam",loss=losses.CategoricalCrossentropy(), metrics=["accuracy"])

    return model

# def cnn_model_attention(input_shape, name):
#     n = input_shape[0]
#     ''' Create a standard deep 2D convolutional neural network'''
#     nclass = 4
#     inp = Input(shape=(n,input_shape[1],1))  
#     x = Convolution2D(4, (4,4), strides=(2, 1), padding="same")(inp)   
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     x = MaxPool2D()(x)
#     x = Dropout(rate=0.2)(x)
    
#     x = Convolution1D(4, 4, strides=1, padding="same")(inp)   
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     x = MaxPool2D()(x)
#     x = Dropout(rate=0.2)(x)
    
#     x = Convolution2D(4, (4,4), strides=(2, 1), padding="same")(inp)  
#     x = BatchNormalization()(x)
#     x = Activation("relu")(x)
#     x = MaxPool2D()(x)
#     x = Dropout(rate=0.2)(x)
    
#     # x = Convolution2D(64, (3,3), strides=(1, 1), padding="same")(x)
#     # x = BatchNormalization()(x)
#     # x = Activation("relu")(x)
#     # x = MaxPool2D()(x)
#     # x = Dropout(rate=0.2)(x)
    
#     x = Reshape((-1, 4))(x)
    
#     #LSTM
#     x = LSTM(2, return_sequences=True)(x)
#     x = BatchNormalization()(x)
#     x = SeqSelfAttention(attention_activation ='tanh')(x)
#     x = LSTM(2, return_sequences=False)(x)
    
#     out = Dense(nclass, activation=softmax)(x)
#     model = models.Model(inputs=inp, outputs=out)
    
#     opt = tf.keras.optimizers.Adam(learning_rate = 0.0001, weight_decay=1e-6)
#     model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
#     return model