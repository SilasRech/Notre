import glob
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, LSTM, Reshape
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from keras.layers import Conv2D, DepthwiseConv2D, Add
from tensorflow.keras import layers
from tensorflow.keras import models
import feature_extraction as fe
from itertools import chain
from tqdm import tqdm
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from sklearn.preprocessing import normalize


def optimizer_settings():
    # Optimizer Settings
    learning_rate = 0.001
    optimizer = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    initializer = tf.keras.initializers.GlorotNormal()

    return learning_rate, optimizer, initializer


def train_model(model_dir, basic_path, parameter):

    # Clear Model
    tf.keras.backend.clear_session()

    print('--' * 40)
    print('starting to extract features')
    x_dirs = glob.glob(basic_path + "/*.wav")
    x_dirs = sorted(x_dirs)
    y_dirs = glob.glob(basic_path + "/*.xls")
    y_dirs = sorted(y_dirs)

    # Take 70% of all available data as training and 10% for evaluation
    end_training_data = int(round(0.7 * len(x_dirs)))
    end_eval_data = int(round(0.74 * len(x_dirs)))

    num_packages = 10


    # save eval data once as training needs to be packaged
    feats_eval, target_eval = generator(x_dirs[end_training_data:end_eval_data], y_dirs[end_training_data:end_eval_data], parameter)

    # Building the architecture of the network
    model = create_network(parameter)

    x_dirs_train = x_dirs[:int(round(0.7 * len(x_dirs)))]
    y_dirs_train = y_dirs[:int(round(0.7 * len(x_dirs)))]

    # Take 70% of all available data as training and 10% for evaluation
    end_training_data = int(round(0.7 * len(x_dirs_train)))
    step_size = int(end_training_data / num_packages)

    num_packages = np.arange(0, end_training_data, step_size).astype(int)

    for j in range(3):
        #shuffled_list = list(zip(x_dirs_train, y_dirs_train))
        #random.shuffle(shuffled_list)
        #x_dirs, y_dirs = zip(*shuffled_list)

        for i in range(len(num_packages)-1):
            print('Epoch:' + ' ' + str(j+1) + '\t' + 'Batch:' + ' ' + str(i+1))

            feats_train, target_train = generator(x_dirs[num_packages[i]:num_packages[i+1]], y_dirs[num_packages[i]:num_packages[i+1]], parameter)

            #np.save(os.path.join(basic_path, 'data/feats_train_{}.npy'.format(i+1)), feats_train)
            #np.save(os.path.join(basic_path, 'data/target_train_{}.npy'.format(i+1)), target_train)

            #feats_train = np.load(os.path.join(basic_path, 'data/feats_train_{}.npy'.format(i+1)))
            #target_train = np.load(os.path.join(basic_path, 'data/target_train_{}.npy'.format(i+1)))

            history = model.fit(feats_train, target_train,
                                batch_size=150,
                                epochs=1,
                                validation_data=(feats_eval, target_eval),
                                shuffle=True)

            model.save(model_dir)

    return history


def testing_network(model_dir, basic_path, parameter):

    x_dirs = glob.glob(basic_path + "/*.wav")
    x_dirs = sorted(x_dirs)
    y_dirs = glob.glob(basic_path + "/*.xls")
    y_dirs = sorted(y_dirs)

    # Take the last 20% of all available data as testing data
    start_test_data = int(round(0.80 * len(x_dirs)))

    feats_test, target_test = generator(x_dirs[start_test_data:], y_dirs[start_test_data:], parameter)

    if not os.path.exists(os.path.join(basic_path, '/data')):
        os.makedirs(os.path.join(basic_path, '/data'))

    np.save(os.path.join(basic_path, 'data/feats_test.npy'), feats_test)
    np.save(os.path.join(basic_path, 'data/target_test.npy'), target_test)

    if os.path.exists(os.path.join(basic_path, 'data/target_test.npy')):
        print('--' * 40)
        print('saving successful')
        print('--' * 40)

    learning_rate, optimizer, initializer = optimizer_settings()

    relu6 = tf.nn.relu6

    model = models.load_model(model_dir, compile=False, custom_objects={'relu6': relu6})
    model.compile(loss=weighted_binary_loss, optimizer=optimizer, metrics=['accuracy'])

    print('--' * 40)
    print('starting testing...')
    print('--' * 40)

    posteriors = model.predict(x=feats_test, verbose=1)

    smoothed_posteriors = smooth_classes(posteriors, parameter['threshold'])
    accuracy = calculate_accuracy(smoothed_posteriors, target_test)

    accuracy_biased = []
    accuracy_utterance = []

    for i in range(149):
        # Select data
        new_posteriors = posteriors[i*1000:(i+1)*1000, :]
        new_target_test = target_test[i*1000:(i+1)*1000, :]

        new_smoothed_posteriors = smooth_classes(new_posteriors, parameter['threshold'])
        accuracy_biased.append(calculate_biased_accuracy(new_smoothed_posteriors, new_target_test))

        utterance_posterior = to_utterance(new_posteriors, parameter['threshold'], parameter['num_frames_utterance'])
        accuracy_utterance.append(calculate_accuracy(utterance_posterior, new_target_test))

    np.savetxt("accuracy_utterance.csv", accuracy_utterance, delimiter=",")
    np.savetxt("accuracy_biased.csv", accuracy_biased, delimiter=",")

    print('--' * 40)
    print("Total Standard Deviation Total: {} %".format(np.std(accuracy_utterance)))
    print('--' * 40)
    print("Total Standard Deviation Biased: {} %".format(np.std(accuracy_biased)))

    accuracy_biased_np = np.asarray(accuracy_biased)
    accuracy_biased = np.sum(accuracy_biased_np) / accuracy_biased_np.shape[0]

    accuracy_utterance_np = np.asarray(accuracy_utterance)
    accuracy_utterance = np.sum(accuracy_utterance_np) / accuracy_utterance_np.shape[0]

    print('--' * 40)
    print("Total Testing Accuracy: {} %".format(accuracy))
    print('--' * 40)
    print("Biased Testing Accuracy: {} % of played notes were detected correctly".format(accuracy_biased))
    print('--' * 40)
    print("Utterance Testing Accuracy: {} % of played notes were detected correctly".format(accuracy_utterance))
    print('--' * 40)

    return accuracy, accuracy_biased, accuracy_utterance


def wav_to_posterior(model, audio_file, parameters):

    # extract features
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
        target = fe.create_labels(y_dirs[i])

        # bring to the same length
        minimal_length = np.min([feats.shape[0], target.shape[0]])
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


def calculate_accuracy(posteriors, y_dirs):

    # Bring both matrices to the same length
    if len(posteriors) < len(y_dirs):
        y_dirs = y_dirs[:len(posteriors), :]
    else:
        posteriors = posteriors[:len(y_dirs), :]

    # check if indices are the same
    similarity = (posteriors.flatten() == y_dirs.flatten()) * 1

    return 100 * np.sum(similarity) / len(similarity)


def calculate_biased_accuracy(posteriors, y_dirs):

    # Bring both matrices to the same length
    if len(posteriors) < len(y_dirs):
        y_dirs = y_dirs[:len(posteriors), :]
    else:
        posteriors = posteriors[:len(y_dirs), :]

    similarity = np.multiply(posteriors, y_dirs).flatten()

    return 100 * np.sum(similarity) / np.sum(y_dirs)


def smooth_classes(posteriors, a):

    # Normalize data for better threshold
    posteriors_normalized = normalize(posteriors, axis=1)
    posteriors_cleaned = np.where(posteriors_normalized > a, 1, 0)

    return posteriors_cleaned


def weighted_binary_loss(y_true, y_pred):

    epsilon = 1e-7
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    y_true = tf.dtypes.cast(y_true, tf.float32)
    alpha = float(0.05)
    first = (1 - alpha) * y_true * math_ops.log(y_pred + epsilon)
    second = alpha * (1 - y_true) * math_ops.log(1 - y_pred + epsilon)

    return -(first + second)


def to_utterance(posteriors, threshold, window):
    """

    :param posteriors: Output of neural network nxm matrix
    :param threshold: Value that decides if 1 or 0
    :param window: Length of one frame to smooth from
    :return: nxm matrix
    """
    x, y = posteriors.shape
    new_posteriors = np.zeros((x, 60))

    posteriors_context = fe.add_context(posteriors, left_context=window, right_context=window)

    for i in range(len(posteriors_context)):
        mean_posteriors = normalize(np.mean(posteriors_context[i], axis=1).reshape(1, -1), axis=1)

        new_posteriors[i, :] = np.where(mean_posteriors > threshold, 1, 0)

    return new_posteriors


def create_network(parameter):

    # Input Shape
    number_features = parameter['left_context'] + parameter['right_context'] + 1
    input_shape = (parameter['num_bins'], number_features, 1)

    # Optimizer Settings
    learning_rate = 0.001
    optimizer = Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

    last_filter_size = parameter['last_filter']
    input = Input(shape=input_shape)

    # First Block
    x = BatchNormalization()(input)
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Group 1
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)
    aux_output1 = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.Add()([x, aux_output1])

    # Group 2
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)
    aux_output2 = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.Add()([x, aux_output2])

    # Group 3
    x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)
    aux_output3 = Conv2D(int(last_filter_size/2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size/2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size/2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = layers.Add()([x, aux_output3])

    x = MaxPooling2D(pool_size=(4, 1), strides=(4, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(int(last_filter_size), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='LastConv')(x)

    model = Model(input, x)
    model.summary()
    output_shape = model.get_layer('LastConv').output_shape

    input2 = Input(shape=(output_shape[1], output_shape[2], output_shape[3]))

    x = Reshape((output_shape[1] * output_shape[2], output_shape[3]))(input2)
    x = tf.keras.layers.Bidirectional(LSTM(256, activation='sigmoid'))(x)
    x = Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)

    x = Dense(parameter['classes'], activation='sigmoid')(x)

    model_2 = Model(input2, x)

    # concatenate both parts
    conv_part = model(input)
    lstm_part = model_2(conv_part)
    model_all = tf.keras.Model(input, lstm_part)
    model_all.summary()

    # compiling and building of the Model
    model_all.compile(loss=weighted_binary_loss, optimizer=optimizer, metrics=['accuracy'])
    model_all.summary()

    return model_all


def create_conv_network(parameter):

    # Input Shape
    number_features = parameter['left_context'] + parameter['right_context'] + 1
    input_shape = (parameter['num_bins'], number_features, 1)

    last_filter_size = parameter['last_filter']
    input = Input(shape=input_shape)

    # Compiling and Building of the Model
    learning_rate, optimizer, initializer = optimizer_settings()

    # First Block
    x = BatchNormalization()(input)

    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 8), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Group 1
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Group 2
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 4), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)

    # Group 2
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)

    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Conv2D(int(last_filter_size / 2), kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)

    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(parameter['classes'], activation='sigmoid')(x)

    model = Model(input, x)
    model.summary()

    model.compile(loss=weighted_binary_loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def efficient_net(parameter):

    # Input Shape
    number_features = parameter['left_context'] + parameter['right_context'] + 1
    input_shape = (parameter['num_bins'], number_features, 1)

    input = Input(shape=input_shape)

    block_1 = 16
    block_2 = 24
    block_3 = 40
    block_4 = 80
    block_5 = 112
    block_6 = 192

    # Compiling and Building of the Model
    learning_rate, optimizer, initializer = optimizer_settings()

    # First Block
    x = BatchNormalization()(input)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(block_1, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = inverted_residual_block(x, int(block_1*6), block_1 )
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(block_2, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = inverted_residual_block(x,int(block_2*6), block_2)
    x = inverted_residual_block(x,int(block_2*6), block_2)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(block_3, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = inverted_residual_block(x, int(block_3*6) ,block_3)
    x = inverted_residual_block(x,int(block_3*6) ,block_3)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(block_4, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = inverted_residual_block(x, int(block_4*6), block_4)
    x = inverted_residual_block(x, int(block_4*6), block_4)
    x = inverted_residual_block(x, int(block_4*6), block_4)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(block_5, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = inverted_residual_block(x, int(block_5*6), block_5)
    x = inverted_residual_block(x, int(block_5*6), block_5)
    x = inverted_residual_block(x, int(block_5*6), block_5)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(block_6, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = inverted_residual_block(x, int(block_6*6), block_6)
    x = inverted_residual_block(x, int(block_6*6), block_6)
    x = inverted_residual_block(x, int(block_6*6), block_6)
    x = inverted_residual_block(x, int(block_6*6), block_6)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(320, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = inverted_residual_block(x, int(block_6*6), 320)

    x = Conv2D(320, kernel_size=(1, 1), strides=(1, 1), activation=tf.nn.relu6, padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(parameter['classes'], activation='sigmoid')(x)

    model = Model(input, x)
    model.summary()

    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model


def inverted_residual_block(x, expand=64, squeeze=16):

    block = Conv2D(expand, (1, 1), activation=tf.nn.relu6, padding='same')(x)
    block = DepthwiseConv2D((3, 3), activation=tf.nn.relu6, padding='same')(block)
    block = Conv2D(squeeze, (1, 1), activation=tf.nn.relu6, padding='same')(block)

    return Add()([block, x])