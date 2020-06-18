import numpy as np
import global_parameters as gp
from librosa import core as rosa
import pandas as pd
import time
from os import path
import tools
import sys

def create_labels1(label_vec, frames, position_frame):
    increment_vec = np.arange(0, len(label_vec), frames)
    labels = []
    time_vec = label_vec['Time'].to_numpy()
    test = label_vec['F0'].reset_index(drop=True)

    for i in range(len(increment_vec)-1):
        values = np.where((time_vec > increment_vec[i]*0.0058) * (time_vec < increment_vec[i+1]*0.0058))
        labels.append(test[values[0][int(round(len(values[0])/2))]])

    return labels


def create_multi_labels(label_vec, frames, position_frame):
    if position_frame == "middle":
        label_position = int(round(frames / 2))
    elif position_frame == 'first':
        label_position = 1
    else:
        label_position = frames - 1

    increment_vec = np.arange(0, len(label_vec), frames)

    labels = []
    test1 = label_vec['F0_1'].reset_index(drop=True)
    test2 = label_vec['F0_2'].reset_index(drop=True)
    test3 = label_vec['F0_3'].reset_index(drop=True)
    test4 = label_vec['F0_4'].reset_index(drop=True)
    test5 = label_vec['F0_5'].reset_index(drop=True)

    for i in range(len(increment_vec)):
        labels.append([test1[increment_vec[i] + label_position], test2[increment_vec[i] + label_position],
                       test3[increment_vec[i] + label_position], test4[increment_vec[i] + label_position],
                       test5[increment_vec[i] + label_position]])

    return labels


def hz_to_note(freq):
    midi = 12 * (np.log2(np.asanyarray(freq)) - np.log2(440.0)) + 69

    if midi == float('-inf'):
        midi = 0

    note_num = int(np.round(midi))

    return note_num


def create_labels(label_vec, frames, position_frame):
    """

    :param label_vec: Panda Frame with Time F0 and Name
    :param frames: Number of Frame for new label
    :param position_frame: Position of label to take from
    :return: label vector
    """
    increment_vec = np.arange(0, len(label_vec), frames)
    label_position = int(round(frames / 2))
    labels = []
    ground_frequencies = label_vec['F0'].reset_index(drop=True)

    for i in range(len(increment_vec)):
        convert_midi = hz_to_note(ground_frequencies[increment_vec[i] + label_position])
        labels.append(convert_midi)

    return labels

def reshaping_data(batched_data, length_feature, number_features):
    # Reshapes Vector into 4-dimensions
    new_shape = []
    run_vector = list(range(int(np.shape(batched_data)[1] / number_features) + 1))
    for i in range(len(run_vector) - 1):
        new_shape.append(batched_data[:, number_features * run_vector[i]:number_features * run_vector[i + 1]])

    final_form = np.reshape(new_shape, (run_vector[-1], length_feature, number_features, 1))

    return final_form


def batchmaking():

    # Loading all the Data
    loading_time_on = time.perf_counter()

    training_files = pd.read_csv(gp.audio_dataframe_Path)
    training_labels = pd.read_csv(gp.label_dataframe_Path)

    loading_time_off = time.perf_counter()
    print(f"Loading Time: {loading_time_off - loading_time_on:0.4f} seconds")

    list_of_names = training_files.Name.unique()

    for i in range(len(list_of_names)):
        #print("Batch Number {} / {}".format(i, len(list_of_names)))

        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(round(20/len(list_of_names)*i)), 100/len(list_of_names) * i))
        sys.stdout.flush()

        # Create Labels depending on the length of one input frame
        if gp.loaded_database == 'Benjamin':
            real_name = list_of_names[i] + '_Labels'
        else:
            real_name = list_of_names[i]

        batched_labels = training_labels.loc[training_labels['Name'] == real_name][:]
        audio_file = training_files.loc[training_files['Name'] == list_of_names[i]]['Audio_Data'].to_numpy()

        len_label_file = int(round(len(batched_labels)*gp.number_features*10))
        len_data_file = len(audio_file)

        len_to_take = min([len_data_file, len_label_file])

        audio_file = audio_file[:len_to_take]

        empty_feature_vec = np.abs(
            rosa.cqt(audio_file, sr=gp.sample_rate, fmin=rosa.note_to_hz('D2'), bins_per_octave=36,
                     n_bins=gp.number_bins, hop_length=gp.number_features*10))

        # Processing for Extracted Features

        number_training_samples = int(round(len_to_take/160)) - int(round(len_to_take/160)) % gp.number_features
        batched_data = empty_feature_vec[:, :number_training_samples]
        batched_labels = batched_labels.iloc[:number_training_samples, :]

        # Get Labels with length of prior computed feature vector
        #batched_labels = training_labels.loc[training_labels['Name'] == real_name][:]

        if gp.loaded_database == 'Benjamin':
            labels = create_multi_labels(batched_labels, gp.number_features, position_frame='middle')
        else:
            labels = create_labels(batched_labels, gp.number_features, position_frame='middle')

        # Create Data depending on the length of on input frame
        reshaped_data = reshaping_data(batched_data, gp.number_bins, gp.number_features)

        # Split into Test and Training Set and Reshaping into 4-dimensional input for Neural Network
        if i < (int(round(len(list_of_names)*(4/5)))):
            split = int(round(np.shape(reshaped_data)[0]*0.8))

            # Reshaping into final format
            training_batch = reshaped_data[:split, :, :, :]
            eval_batch = reshaped_data[split:, :, :, :]

            training_labels_batch = labels[:split]
            eval_labels_batch = labels[split:]

            if i == 0:
                train_label = training_labels_batch
                eval_label = eval_labels_batch
                train = training_batch
                eval = eval_batch

            # Concatenating Training and Testing Data
            train_label = np.concatenate((train_label, training_labels_batch))
            eval_label = np.concatenate((eval_label, eval_labels_batch))

            train = np.concatenate((train, training_batch))
            eval = np.concatenate((eval, eval_batch))

        else:

            if i == (int(round(len(list_of_names)*(4/5)))):
                test_label = labels
                test1 = reshaped_data

            # Concatenating Testing Data
            test1 = np.concatenate((test1, reshaped_data))
            test_label = np.concatenate((test_label, labels))

            np.save(gp.label_eval_data, eval)
            np.save(gp.label_eval_labels, eval_label)
            np.save(gp.label_train_data, train)
            np.save(gp.label_train_labels, train_label)
            np.save(gp.label_test_data, test1)
            np.save(gp.label_test_labels, test_label)

            if path.exists(gp.label_test_data):
                print('Saving Successful')

    return train, eval, test1, train_label, eval_label, test_label


def make_frames(audio_data, window_size, hop_size, sampling_rate):
    """
    Splits an audio signal into subsequent frames.

    :param audio_data: array representing the audio signal.
    :param sampling_rate: sampling rate in Hz.
    :param window_size: window size in seconds.
    :param hop_size: hop size (frame shift) in seconds.
    :return: n x m array of signal frames, where n is the number of frames and m is the window size in samples.
    """

    # transform window size in seconds to samples and calculate next higher power of two
    window_size_samples = 2 ** tools.next_pow2(window_size)

    # assign hamming window
    hamming_window = np.hamming(window_size_samples)

    # transform hop size in seconds to samples

    # get number of frames from function in tools.py
    n_frames = int(np.floor(len(audio_data) / sampling_rate * 100))

    # initialize nxm matrix (n is number of frames, m is window size)
    # initialize with zeros to avoid zero padding
    frame_mat = np.zeros([n_frames, window_size_samples], dtype=float)

    # write frames in matrix
    for i in range(n_frames):
        start = i * hop_size
        end = i * hop_size + window_size_samples
        frame_mat[i, 0:len(audio_data[start:end])] = audio_data[start:end]
        frame_mat[i, :] = frame_mat[i, :] * hamming_window

    return frame_mat



