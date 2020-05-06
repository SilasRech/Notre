import numpy as np
import global_parameters as gp
from librosa import core as rosa
from librosa import display as dsp
import pandas as pd
import matplotlib.pyplot as plt
import time

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
    increment_vec = np.arange(0, len(label_vec), frames)
    label_position = int(round(frames / 2))
    labels = []
    test1 = label_vec['F0_1'].reset_index(drop=True)
    test2 = label_vec['F0_2'].reset_index(drop=True)
    test3 = label_vec['F0_3'].reset_index(drop=True)

    for i in range(len(increment_vec)):
        labels.append([test1[increment_vec[i] + label_position], test2[increment_vec[i] + label_position],
                       test3[increment_vec[i] + label_position]])

    return labels


def create_labels(label_vec, frames, position_frame):
    increment_vec = np.arange(0, len(label_vec), frames)
    label_position = int(round(frames / 2))
    labels = []
    test = label_vec['F0'].reset_index(drop=True)

    for i in range(len(increment_vec)):
        labels.append(test[increment_vec[i] + label_position])

    return labels


def round_to_nearest_multiple(x , base):
    return base * np.floor(x/base)


def reshaping_data(batched_data, length_feature, number_features):
    # Reshapes Vector into 4-dimensions
    new_shape = []
    run_vector = list(range(int(np.shape(batched_data)[1] / number_features)+1))
    for i in range(len(run_vector)-1):
        new_shape.append(batched_data[:, number_features*run_vector[i]:number_features*run_vector[i+1]])

    final_form = np.reshape(new_shape, (run_vector[-1], length_feature, number_features, 1))

    return final_form


def batchmaking():

    # Loading all the Data
    loading_time_on = time.perf_counter()

    training_files = pd.read_csv(gp.audio_dataframe_Path)
    training_labels = pd.read_csv(gp.label_dataframe_Path)

    loading_time_off = time.perf_counter()
    print(f"Loading Time: {loading_time_off - loading_time_on:0.4f} seconds")

    test = training_files['Audio_Data']
    test2 = training_files.Name.unique()

    for i in range(len(test2)):
        print("Batch Number {}".format(i))

        if gp.extraction == 'CQT':
            loading_time_on2 = time.perf_counter()
            feature_set = rosa.cqt(training_files.loc[training_files['Name'] == test2[i]]['Audio_Data'].to_numpy(), sr=gp.sample_rate, hop_length=256)

            loading_time_off2 = time.perf_counter()
            print(f"CQT-Calculation: {loading_time_off2 - loading_time_on2:0.4f} seconds")
        else:
            print("FFT {} starting... ".format(i))
            # Calculating STFT and cutting them to multiple of numbers of features for reshape
            # checkl = training_files.loc[training_files['Name'] == test2[i]]['Audio_Data']
            feature_set = np.abs(rosa.stft(training_files.loc[training_files['Name'] == test2[i]]['Audio_Data'].to_numpy(), n_fft=1024, win_length=512, hop_length=256))

            #dsp.specshow(librosa.amplitude_to_db(feature_set, ref = np.max),
            #y_axis = 'log', x_axis = 'time')
            #plt.title('Power spectrogram')
            #plt.colorbar(format='%+2.0f dB')
            #plt.tight_layout()
            #plt.show()

        number_training_samples = np.shape(feature_set)[1] - np.shape(feature_set)[1] % gp.number_features
        batched_data = feature_set[:, :number_training_samples]

        length_features = np.shape(batched_data)[0]

        # Create Labels depending on the length of one input frame
        if gp.loaded_database == 'Benjamin':
            real_name = test2[i] + 'Labels'
        else:
            real_name = test2[i]
        #
        #
        #
        #

        # HIER WEITERARBEITEN
        batched_labels = training_labels.loc[training_labels['Name'] == real_name][:number_training_samples]

        if gp.loaded_database == 'Benjamin':
            labels = create_multi_labels(batched_labels, gp.number_features, position_frame='middle')
        else:
            labels = create_labels(batched_labels, gp.number_features, position_frame='middle')

        # Create Data depending on the length of on input frame
        reshaped_data = reshaping_data(batched_data, length_features, gp.number_features)

        # Split into Test and Training Set and Reshaping into 4-dimensional input for Neural Network
        split = int(round(np.shape(reshaped_data)[0]*0.8))

        loading_time_on3 = time.perf_counter()

        # Reshaping into final format
        training_batch = reshaped_data[:split, :, :, :]
        test_batch = reshaped_data[split:, :, :, :]

        training_labels_batch = labels[:split]
        test_labels_batch = labels[split:]

        loading_time_off3 = time.perf_counter()
        print(f"Reshaping: {loading_time_off3 - loading_time_on3:0.4f} seconds")

        if i == 0:
            train_label = training_labels_batch
            test_label = test_labels_batch
            train = training_batch
            test = test_batch

        loading_time_on4 = time.perf_counter()

        # Concatenating Training and Testing Data
        train_label = np.concatenate((train_label, training_labels_batch))
        test_label = np.concatenate((test_label, test_labels_batch))

        train = np.concatenate((train, training_batch))
        test = np.concatenate((test, test_batch))

        loading_time_off4 = time.perf_counter()
        print(f"Concatenating Time: {loading_time_off4 - loading_time_on4:0.4f} seconds")

        if i % 100 == 0:
            np.save(gp.label_test_data.format(i), test)
            np.save(gp.label_test_data.format(i), test_label)
            np.save(gp.label_train_data.format(i), train)
            np.save(gp.label_train_labels.format(i), train_label)

    return train, test, train_label, test_label




