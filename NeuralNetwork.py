import glob
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LSTM, Reshape
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import layers
from tensorflow.keras import models
import feature_extraction as fe
from itertools import chain
from tqdm import tqdm
import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import random
from sklearn.preprocessing import normalize


def train_model(model_dir, basic_path, parameter):

    # Clear Model
    tf.keras.backend.clear_session()

    print('--' * 40)
    print('training data does not exists, starting to extract features')
    x_dirs = glob.glob(basic_path + "/*.wav")
    x_dirs = sorted(x_dirs)
    y_dirs = glob.glob(basic_path + "/*.xls")
    y_dirs = sorted(y_dirs)

    # Take 70% of all available data as training and 10% for evaluation
    end_training_data = int(round(0.7 * len(x_dirs)))
    end_eval_data = int(round(0.78 * len(x_dirs)))

    num_packages = 10

    if os.path.exists(os.path.join(basic_path, 'data/feats_eval.npy')):

        feats_eval = np.load(os.path.join(basic_path, 'data/feats_eval.npy'))
        target_eval = np.load(os.path.join(basic_path, 'data/target_eval.npy'))
    else:

        # save eval data once as training needs to be packaged
        feats_eval, target_eval = generator(x_dirs[end_training_data:end_eval_data], y_dirs[end_training_data:end_eval_data], parameter)

        if not os.path.exists(os.path.join(basic_path, 'data')):
            os.makedirs(os.path.join(basic_path, 'data'))

        np.save(os.path.join(basic_path, 'data/feats_eval.npy'), feats_eval)
        np.save(os.path.join(basic_path, 'data/target_eval.npy'), target_eval)

    # Building the architecture of the network
    model = create_network(parameter)

    # Callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs\multiclassDetection')

    x_dirs_train = x_dirs[:int(round(0.7 * len(x_dirs)))]
    y_dirs_train = y_dirs[:int(round(0.7 * len(x_dirs)))]

    # Take 70% of all available data as training and 10% for evaluation
    end_training_data = int(round(0.7 * len(x_dirs_train)))
    step_size = int(end_training_data / num_packages)

    num_packages = np.arange(0, end_training_data, step_size).astype(int)

    for j in range(1):

        shuffled_list = list(zip(x_dirs_train, y_dirs_train))
        random.shuffle(shuffled_list)
        x_dirs, y_dirs = zip(*shuffled_list)
        for i in range(len(num_packages)-1):
            print('Epoch:' + ' ' + str(j+1) + '\t' + 'Batch:' + ' ' + str(i+1))

            feats_train, target_train = generator(x_dirs[num_packages[i]:num_packages[i+1]], y_dirs[num_packages[i]:num_packages[i+1]], parameter)

            np.save(os.path.join(basic_path, 'data/feats_train{}.npy'.format(i)), feats_train)
            np.save(os.path.join(basic_path, 'data/target_train{}.npy'.format(i)), target_train)

            history = model.fit(feats_train, target_train,
                                batch_size=150,
                                epochs=1,
                                callbacks=[tensorboard],
                                validation_data=(feats_eval, target_eval),
                                shuffle=True)

            model.save(model_dir)

    return history


def testing_network(model_dir, basic_path, parameter):

    if os.path.exists(os.path.join(basic_path, 'data/feats_test.npy')):
        print('--' * 40)
        print('loading data')
        print('--' * 40)
        feats_test = np.load(os.path.join(basic_path, 'data/feats_test.npy'))
        target_test = np.load(os.path.join(basic_path, 'data/target_test.npy'))
        print('--' * 40)
        print('finished loading')
        print('--' * 40)

    else:

        x_dirs = glob.glob(basic_path + "/*.wav")
        x_dirs = sorted(x_dirs)
        y_dirs = glob.glob(basic_path + "/*.xls")
        y_dirs = sorted(y_dirs)

        # Take the last 20% of all available data as testing data
        start_test_data = int(round(0.97 * len(x_dirs)))

        feats_test, target_test = generator(x_dirs[start_test_data:], y_dirs[start_test_data:], parameter)

        if not os.path.exists(os.path.join(basic_path, '/data')):
            os.makedirs(os.path.join(basic_path, '/data'))

        np.save(os.path.join(basic_path, 'data/feats_test.npy'), feats_test)
        np.save(os.path.join(basic_path, 'data/target_test.npy'), target_test)

        if os.path.exists(os.path.join(basic_path, 'data/target_test.npy')):
            print('--' * 40)
            print('saving successful')
            print('--' * 40)

    model = models.load_model(model_dir)

    number_features = parameter['left_context'] + parameter['right_context'] + 1

    print('--' * 40)
    print('starting testing...')
    print('--' * 40)

    posteriors = model.predict(x=feats_test, verbose=1)

    #accuracy = calculate_accuracy(posteriors, target_test, a=0.1)
    accuracy = 0

    return accuracy


def create_network(parameter):

    # Input Shape
    number_features = parameter['left_context'] + parameter['right_context'] + 1
    input_shape = (parameter['num_bins'], number_features, 1)

    # Optimizer Settings
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

    last_filter_size = parameter['last_filter']
    input = Input(shape=input_shape)

    #First Block
    x = BatchNormalization()(input)
    ##
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    #x = Dropout(0.5)(x)

    # Group 1
    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same')(x)
    aux_output1 = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output1])
    #x = Dropout(0.5)(x)

    # Group 2
    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same')(x)
    aux_output2 = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output2])
    #x = Dropout(0.5)(x)

    # Group 3
    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same')(x)

    aux_output3 = Conv2D(int(last_filter_size), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.concatenate([x, aux_output3])
    #x = Dropout(0.1)(x)

    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size), kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='same', name='LastConv')(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    model.summary()
    output_shape = model.get_layer('LastConv').output_shape

    input2 = Input(shape=(output_shape[1], output_shape[2], output_shape[3]))

    x = Reshape((output_shape[1] * output_shape[2], output_shape[3]))(input2)

    x = tf.keras.layers.Bidirectional(LSTM(256, activation='sigmoid'))(x)

    x = Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = Dense(parameter['classes'], activation='sigmoid')(x)

    model_2 = Model(input2, x)

    # concatenate both parts
    conv_part = model(input)
    lstm_part = model_2(conv_part)
    model_all = tf.keras.Model(input, lstm_part)
    model_all.summary()

    # Compiling and Building of the Model
    model_all.compile(loss=weighted_binary_loss, optimizer=optimizer, metrics=['accuracy'])
    model_all.summary()

    return model_all


def wav_to_posterior(model, audio_file, parameters):

    feats = fe.extract_features(audio_file, parameters)

    # predict posteriors with the trained model
    posteriors = model.predict(feats)

    return posteriors


def generator(x_dirs, y_dirs, parameters):
    feats_list = []
    target_list = []

    length_feats = 0
    number_context = parameters['left_context'] + parameters['right_context'] + 1

    for i in tqdm(range(len(x_dirs))):
        # compute features
        feats = fe.extract_features(x_dirs[i], parameters)

        # get label
        #target = fe.create_multi_labels(y_dirs[i], parameters['classes'])
        target = fe.create_labels(y_dirs[i], parameters['classes'])

        minimal_length = np.min([feats.shape[0], target.shape[0]])

        # bring to the same length
        feats = feats[:minimal_length, :, :, :]
        target = target[:minimal_length, :]

        # append to list with features and targets
        length_feats += len(feats)
        feats_list.append(feats)
        target_list.append(target)

    # convert list to numpy array
    feats_list = list(chain.from_iterable(feats_list))
    feats_list_new = np.reshape(np.array(feats_list), newshape=(length_feats, parameters['num_bins'], number_context, 1))

    target_list = list(chain.from_iterable(target_list))
    target_list_new = np.reshape(np.array(target_list), newshape=(length_feats, parameters['classes']))

    return feats_list_new, target_list_new


def calculate_accuracy(posteriors, y_dirs, a):

    # round values to one or zero with threshold a
    posteriors_cleaned = smooth_classes(posteriors, a)
    max_classes = np.max(posteriors, axis=1)
    detected_classes = np.sum(posteriors_cleaned, axis=1)
    total = 0
    accuracy = 0

    # check if indices are the same
    for j in range(len(y_dirs)):
        if y_dirs[j] in posteriors_cleaned:
            accuracy += 1
            total += 1
        else:
            total += 1

    return accuracy/total


def smooth_classes(posteriors, a):

    posteriors_normalized = normalize(posteriors, axis=1)
    posteriors_cleaned = np.where(posteriors_normalized > a, 1, 0)

    return posteriors_cleaned


def weighted_binary_loss(y_true, y_pred):

    epsilon = 1e-7
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    y_true = tf.dtypes.cast(y_true, tf.float32)
    alpha = float(0.5)
    first = (1-alpha) * y_true * math_ops.log(y_pred + epsilon)
    second = alpha * (1 - y_true) * math_ops.log(1 - y_pred + epsilon)

    return -(first + second)

