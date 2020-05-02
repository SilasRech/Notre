import collections
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import optimizers, utils
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LSTM, ConvLSTM2D, Reshape
from tensorflow.keras import Model, Input
#from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import random
import math
from tensorflow.keras import layers

import global_parameters as gp


def neural_network(training, testing, train_label, test_label):
    # Configuring Inputs
    quant_test_label = utils.to_categorical(calculate_labels(120, test_label), num_classes=120)
    quant_train_labels = utils.to_categorical(calculate_labels(120, train_label), num_classes=120)

    average_melody = []

    # Clear Model
    tf.keras.backend.clear_session()

    shape_input_train = np.shape(training)[1]
    # Basic Neural Network Settings
    input_shape = (84, 10, 1)
    learning_rate = 0.0002
    #nadam = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

    # Building the architecture of the network
    model = create_network(input_shape)
    model.summary()
    # Add an Embedding Layer expecting the set input size

    # Add LSTM-Network
    #model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu', strides=(1, 1), padding='same',input_shape=input_shape))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    #model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same'))
    #model.add(layers.Conv2D(128, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same'))
    #model.add(BatchNormalization())
    #model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
   # model.add(layers.Conv2D(256, kernel_size=(2, 2), activation='relu', strides=(1, 1), padding='same'))
    #model.add(layers.Flatten())
    #model.summary()

    # Compiling and Building of the Model
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(training, quant_train_labels,
              batch_size=150,
              epochs=25,
              validation_data=(testing, quant_test_label), shuffle=True)


def calculate_labels(number_classes, raw_labels):

    starting_note = 73.42
    notes = []
    for i in range(number_classes):
        frequency = starting_note * math.pow(2, i/24)
        notes.append(frequency)

    labels = np.digitize(raw_labels, notes)

    return labels

def create_network(input_shape):

    input = Input(shape=input_shape)
    filters = 64
    kernel_size = (1, 1)
    units = 64

    x = BatchNormalization()(input)
    x = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(128, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.4)(x)
    x = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)


    #x = Reshape((42*5, 256))(x)

    #x = tf.keras.layers.LSTM(
    #units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
    #kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    #bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
    #recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    #dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False,
    #return_state=False, go_backwards=False, stateful=False, time_major=False,
    #unroll=False)(x)

    #x = tf.keras.layers.LSTM(
    #    units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
    #    kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    #    bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
    #    recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    #    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    #    dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False,
    #    return_state=False, go_backwards=False, stateful=False, time_major=False,
    #    unroll=False)(x)

    #x = tf.keras.layers.LSTM(
     #   units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
      #  kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
       # bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
       # recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
       # kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
       # dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False,
       # return_state=False, go_backwards=False, stateful=False, time_major=False,
       # unroll=False)(x)

    x = Flatten()(x)
    x = Dense(gp.classes_to_detect, activation='softmax')(x)

    #model = tf.keras.Sequential()
    #model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=(513, 8, 1)))
    #model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    #model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    #model.add(Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same'))

    #model.add(Reshape((513, 8*256)))

    #
    #model.add(Dense(gp.classes_to_detect, activation='softmax'))

    return Model(input, x)
