import NeuralNetwork as nn
import numpy as np
from tensorflow.keras import models
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import feature_extraction as fe
import os
import librosa as rosa
from scipy.io import wavfile
import pumpp


def visualization_network(model_dir, audio_file, label_dir, parameters):

    model = models.load_model(model_dir)
    label = fe.create_multi_labels(label_dir, parameters['classes'])

    # load and read audio signal
    sampling_rate, signal = wavfile.read(audio_file)

    # normalize audio signal to -1 to 1
    signal_norm = signal / max(signal)
    Feature_Extractor = pumpp.feature.CQT(name='CQT', sr=16000, hop_length=parameters['hop_size'],n_octaves=5, fmin=73.43)

    features = Feature_Extractor.transform_audio(signal_norm)['mag']
    #features = np.abs(rosa.cqt(signal_norm, sr=16000, fmin=rosa.note_to_hz(parameters['f_min']), bins_per_octave=parameters['bins_per_octave'],
    #             n_bins=parameters['num_bins'], hop_length=parameters['hop_size']))

    #features = np.abs(rosa.stft(signal_norm,  n_fft=1024, hop_length=160, win_length=256))
    #features = fe.add_context(features)
    #features = np.reshape(features, newshape=(1001, 84, 31))

    flatui = ["#ffffff", "#000000"]
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    # CQT Feature Spectrum
    plt.imshow(features.T, origin='lower', aspect='auto')
    plt.xlabel('Frames')
    plt.title('CQT Features')
    plt.ylabel('CQT-Bins')
    plt.colorbar()
    plt.savefig('test/OriginalFeatures_CQT_600.png')
    plt.show()

    flatui = ["#ffffff", "#000000"]
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    # CQT Feature Spectrum
    plt.imshow(features.T, origin='lower', aspect='auto', cmap='jet')
    plt.xlabel('Frames')
    plt.title('CQT Features')
    plt.ylabel('CQT-Bins')
    plt.colorbar()
    plt.savefig('test/OriginalFeatures.png')
    plt.show()

    # show true label
    colormap = plt.imshow(label.transpose(), origin='lower', cmap=my_cmap, aspect='auto', extent=[0, label.shape[0], 0, label.shape[1]])
    plt.title('True Label per Frame')
    plt.xlabel('Frames')
    plt.ylabel('MIDI-Note')
    cbar = plt.colorbar(colormap)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    plt.savefig('test/OriginalLabel.png')
    plt.show()

    # show detected classes by the network
    posterior = nn.wav_to_posterior(model, audio_file, parameters)

    # Keep only the classes with the five highest probabilities
    posterior_cleaned = nn.smooth_classes(posterior, a=0.1)

    # Predicted classes for frame
    colormap = plt.imshow(posterior_cleaned.transpose(), origin='lower', cmap=my_cmap, aspect='auto', extent=[0, posterior_cleaned.shape[0], 0, posterior_cleaned.shape[1]])
    plt.xlabel('Frames')
    plt.title('Predicted Label per Frame')
    plt.ylabel('MIDI-Note')
    cbar = plt.colorbar(colormap)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    plt.savefig('test/NetworkLabel.png')
    plt.show()

    return posterior


if __name__ == "__main__":

    # path to database
    basic_path = 'D:/Backup/Trainingsdatenbank/BenjaminDaten/InstrumentNo1/'

    # parameters for the feature extraction
    parameter = { 'hop_size': 160,
                  'f_min': 'D2',
                  'bins_per_octave': 36,
                  'num_bins': 168,
                  'left_context': 15,
                  'right_context': 15,
                  'sampling_rate': 16000,
                  'classes': 60,
                  'last_filter': 256,
                  }

    # define a name for the model
    model_name = 'model_binary'
    # directory for the model
    model_dir = os.path.join('model', model_name + '.h5')
    if not os.path.exists('model'):
        os.makedirs('model')

    # train neural network and save model to model_dir
    #history = nn.train_model(model_dir, basic_path, parameter)

    # test the network with unknown data
    #accuracy = nn.testing_network(model_dir, basic_path, parameter)

    #print('--' * 40)
    #print("Total Testing Accuracy: {} %".format(accuracy*100))

    visualization_network(model_dir, 'test/Random_rSeed101_Noise1.wav', 'test/Random_rSeed101_Noise1_Labels.xls', parameter)



