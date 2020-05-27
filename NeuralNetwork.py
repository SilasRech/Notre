import numpy as np
import tensorflow as tf
from keras import utils
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LSTM, Reshape
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
import random
import math
import matplotlib.pyplot as plt
from tensorflow.keras import layers

import global_parameters as gp


def neural_network(train, eval, test, train_label, eval_label, test_label):
    # Configuring Inputs
    if gp.loaded_database == 'Benjamin':
        quant_test_label = utils.to_categorical(test_label[:, 1], num_classes=120)
        quant_eval_label = utils.to_categorical(eval_label[:, 1], num_classes=120)
        quant_train_label = utils.to_categorical(train_label[:, 1], num_classes=120)

    else:
        quant_test_label = utils.to_categorical(test_label, num_classes=120)
        quant_eval_label = utils.to_categorical(eval_label, num_classes=120)
        quant_train_label = utils.to_categorical(train_label, num_classes=120)

    average_melody = []

    # Clear Model
    tf.keras.backend.clear_session()

    # Basic Neural Network Settings
    input_shape = gp.input_shape
    learning_rate = 0.001
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

    # Building the architecture of the network
    model = create_network(input_shape)
    model.summary()

    # Compiling and Building of the Model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(train, quant_train_label,
              batch_size=100,
              epochs=30,
              validation_data=(eval, quant_eval_label), shuffle=True)

    acc = model.evaluate(x=test, y=quant_test_label)
    print("Testing Accuracy {}".format(acc))

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()

def create_network(input_shape):

    input = Input(shape=input_shape)
    filters = gp.number_units_LSTM
    units = 120

    #First Block
    x = BatchNormalization()(input)
    #
    x = Conv2D(int(gp.last_filter_size / 8), kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    x = Conv2D(int(gp.last_filter_size / 8), kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Dropout(0.1)(x)


    # Group 1
    #x = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), padding='same')(x)
    aux_output1 = Conv2D(int(gp.last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(gp.last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    x = Conv2D(int(gp.last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output1])
    #x = Dropout(0.1)(x)

    # Group 2
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)
    aux_output2 = Conv2D(int(gp.last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu',
                         padding='same')(x)
    x = Conv2D(int(gp.last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = BatchNormalization()(x)
    x = Conv2D(int(gp.last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output2])
    #x = Dropout(0.1)(x)

    # Group 3
    #x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)

    aux_output3 = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu',
                         padding='same')(x)
    x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output3])
    #x = Dropout(0.1)(x)

    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)
    #x = Dropout(0.1)(x)
    x = Conv2D(int(gp.last_filter_size/2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='same')(x)

    #x = BatchNormalization()(x)
    x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Conv2D(int(gp.last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Dropout(0.1)(x)

    #x = Reshape((120, 256*16))(x)

    #x = tf.keras.layers.Bidirectional(LSTM(200, activation='sigmoid'))(x)
    """
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
    x = Flatten()(x)
    x = Dense(gp.classes_to_detect, activation='softmax')(x)

    return Model(input, x)

def calculate_labels(number_classes, raw_labels):

    starting_note = 73.42
    notes = []
    for i in range(number_classes):
        frequency = starting_note * math.pow(2, i/24)
        notes.append(frequency)

    labels = np.digitize(raw_labels, notes)

    return labels