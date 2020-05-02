import numpy as np
import global_parameters as gp
from librosa import core as rosa
from librosa import display as dsp
import pandas as pd
import matplotlib.pyplot as plt

def create_labels(label_vec, frames, position_frame):
    increment_vec = np.arange(0, len(label_vec), frames)
    labels = []
    time_vec = label_vec['Time'].to_numpy()
    test = label_vec['F0'].reset_index(drop=True)
    for i in range(len(increment_vec)-1):
        values = np.where((time_vec > increment_vec[i]*0.0058) * (time_vec < increment_vec[i+1]*0.0058))
        labels.append(test[values[0][int(round(len(values[0])/2))]])
    return labels

def round_to_nearest_multiple(x , base):
    return base * np.floor(x/base)

def batchmaking():
    training_files = pd.read_csv(gp.audio_dataframe_Path1)
    training_labels = pd.read_csv(gp.label_dataframe_Path1)

    test = training_files['Audio_Data']
    test2 = training_files.Name.unique()

    #for i in range(2):
    for i in range(len(test2)):
        if gp.extraction == 'CQT':
            feature_set = rosa.cqt(training_files.loc[training_files['Name'] == test2[i]]['Audio_Data'].to_numpy(), sr=16000, hop_length=256)

        else:
            print("Batch {} starting... ".format(i))
            # Calculating STFT and cutting them to multiple of numbers of features for reshape
            # checkl = training_files.loc[training_files['Name'] == test2[i]]['Audio_Data']
            feature_set = np.abs(rosa.stft(training_files.loc[training_files['Name'] == test2[i]]['Audio_Data'].to_numpy(), n_fft=1024, win_length=512, hop_length=256))

            #dsp.specshow(librosa.amplitude_to_db(feature_set, ref = np.max),
            #y_axis = 'log', x_axis = 'time')
            #plt.title('Power spectrogram')
            #plt.colorbar(format='%+2.0f dB')
            #plt.tight_layout()
            #plt.show()

        print("Batch Number {}".format(i))
        number_training_samples = np.shape(feature_set)[1] - np.shape(feature_set)[1] % gp.number_features
        batched_data = feature_set[:, :number_training_samples]

        length_features = np.shape(feature_set)[0]

        # Create Labels depending on the length of one input frame
        batched_labels = training_labels.loc[training_labels['Name'] == test2[i]]
        labels = create_labels(batched_labels, gp.number_features, position_frame='middle')

        # Split into Test and Training Set and Reshaping into 4-dimensional input for Neural Network
        split = int(round_to_nearest_multiple(number_training_samples * 0.8, gp.number_features))

        training_batch = np.reshape(batched_data[:, :split], (-1, length_features, gp.number_features, 1))
        test_batch = np.reshape(batched_data[:, split:], (-1, length_features, gp.number_features, 1))

        label_length_test = np.shape(test_batch)[0]
        label_length_train = np.shape(training_batch)[0]

        if i == 0:
            train_label = labels[:label_length_train]
            test_label = labels[-label_length_test:]
            train = training_batch
            test = test_batch

        train_label = np.concatenate((train_label, labels[:label_length_train]))
        test_label = np.concatenate((test_label, labels[-label_length_test:]))

        train = np.concatenate((train, training_batch))
        test = np.concatenate((test, test_batch))

    np.save('/home/jonny/Desktop/Trainingsdatenbank/train_features/test_data.npy', test)
    np.save('/home/jonny/Desktop/Trainingsdatenbank/train_features/test_label.npy', test_label)
    np.save('/home/jonny/Desktop/Trainingsdatenbank/train_features/train_data.npy', train)
    np.save('/home/jonny/Desktop/Trainingsdatenbank/train_features/train_label.npy', train_label)

    return train, test, train_label, test_label




