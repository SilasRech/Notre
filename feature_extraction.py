from scipy.io import wavfile
import pandas as pd
import numpy as np
import librosa as rosa
from keras import utils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pumpp


def extract_features(audio_file, parameters):
    """
    :param audio_file: path to audio file
    :param sampling_rate: sample_rate
    :param number_bins: number of extracted bins for CQT
    :param hopsize: hop size (frame shift) in samples.
    :return:
    """
    # load and read audio signal
    sampling_rate, signal = wavfile.read(audio_file)

    # normalize audio signal to -1 to 1
    signal_norm = signal / max(signal)

    features = np.abs(rosa.cqt(signal_norm, sr=sampling_rate, fmin=rosa.note_to_hz(parameters['f_min']), bins_per_octave=parameters['bins_per_octave'],
                 n_bins=parameters['num_bins'], hop_length=parameters['hop_size'])).T

    features_with_context = add_context(features, parameters['left_context'], parameters['right_context'])

    return features_with_context


def create_multi_labels(label_path, classes):
    """
   :param labels: path to label
   :return: New list of labels in form of a dataframe
   """

    labels = pd.read_excel(label_path, header=None)

    # Delete first six rows as it contains only information
    mod_labels = labels.drop(list(range(6))).to_numpy()[:,1:]

    hot_encoded_label = np.zeros((mod_labels.shape[0], classes +1))

    # Make one hot encoded vector from all classes
    for i in range(5):
        hot_encoded_label += utils.to_categorical(mod_labels[:, i], num_classes=classes +1)

    hot_encoded_label = np.where(hot_encoded_label == 0, 0, 1)

    return hot_encoded_label[:, 1:]


def create_labels(label_path, classes):
    """
   :param labels: path to label
   :return: New list of labels in form of a dataframe
   """

    labels = pd.read_excel(label_path, header=None)

    ## Delete first six rows as it contains only information
    #mod_labels = labels.drop(list(range(6))).to_numpy()[:,1:]

    #hot_encoded_label = np.zeros((mod_labels.shape[0], classes+1, 5))

    ## Make one hot encoded vector from all classes
    #for i in range(5):
    #    hot_encoded_label[:, :, i] = utils.to_categorical(mod_labels[:, i], num_classes=classes +1)

    #return hot_encoded_label[:, 1:, :]
    return labels.to_numpy()


def add_context(feats, left_context=7, right_context=7):
    """
    Adds context to the features.

    :param feats: extracted features of size (n x d) array of features, where n is the number of frames and d is the
           feature dimension.
    :param left_context: Number of predecessors.
    :param right_context: Number of successors.
    :return: Features with context of size (n x d x c), where c = left_context + right_context + 1
    """

    feats_pre = feats
    feats_post = feats

    for i in range(left_context):
        feats_pre = np.roll(feats_pre, 1, axis=0)
        feats_pre[0, :] = feats_pre[1, :]
        feats = np.dstack((feats_pre, feats))

    for m in range(right_context):
        feats_post = np.roll(feats_post, -1, axis=0)
        feats_post[-1, :] = feats_post[-2, :]
        feats = np.dstack((feats, feats_post))

    return np.reshape(feats, (feats.shape[0], feats.shape[1], feats.shape[2], 1))
