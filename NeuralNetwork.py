import collections
import matplotlib.pyplot as plt
import numpy as np
from keras import utils
from keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LSTM, Reshape
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
    #quant_test_real_labels =  utils.to_categorical(calculate_labels(120, test_label_real), num_classes=120)

    average_melody = []

    # Clear Model
    tf.keras.backend.clear_session()

    shape_input_train = np.shape(training)[1]
    # Basic Neural Network Settings
    input_shape = gp.input_shape
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
    history = model.fit(training, quant_train_labels,
              batch_size=150,
              epochs=15,
              validation_data=(testing, quant_test_label), shuffle=True)

    #acc = model.evaluate(x=test_real, y=quant_test_real_labels)
    #print("Testing Accuracy {}".format(acc))
"""
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
"""

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
    filters = gp.number_units_LSTM
    units = 120

    #First Block
    x = BatchNormalization()(input)
    #
    x = Conv2D(int(gp.last_filter_size / 8), kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(gp.last_filter_size / 8), kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    # Group 1

    x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(x)
    aux_output1 = Conv2D(int(gp.last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(gp.last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(gp.last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output1])
    x = Dropout(0.1)(x)

    # Group 2
    x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(x)

    aux_output2 = Conv2D(int(gp.last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu',
                         padding='same')(x)
    x = Conv2D(int(gp.last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(gp.last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output2])
    x = Dropout(0.1)(x)

    # Group 3
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(x)

    aux_output3 = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu',
                         padding='same')(x)
    x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output3])
    x = Dropout(0.1)(x)

    x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(x)
    #x = Dropout(0.3)(x)
    #x = Conv2D(int(gp.last_filter_size/2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    #x = BatchNormalization()(x)
    #x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(256, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Dropout(0.3)(x)

    #x = Reshape((513, 256))(x)

    #x = tf.keras.layers.Bidirectional(LSTM(256))(x)

    """
    x = Reshape((units, 1))(x)

    x = tf.keras.layers.LSTM(
        units, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None,
        recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
        dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False,
        return_state=False, go_backwards=False, stateful=False, time_major=False,
        unroll=False)(x)
    """
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
