import NeuralNetwork as nn
import numpy as np
from tensorflow.keras import models
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import feature_extraction as fe
import os
import librosa as rosa


def visualization_network(model_dir, audio_file, label_dir, parameters):

    relu6 = tf.nn.relu6

    model = models.load_model(model_dir, compile=False, custom_objects={'relu6': relu6})

    learning_rate, optimizer, initializer = nn.optimizer_settings()
    model.compile(loss=nn.weighted_binary_loss, optimizer=optimizer, metrics=['accuracy'])

    label = fe.create_labels(label_dir)

    # load and read audio signal
    signal, s = rosa.load(audio_file, parameters['sampling_rate'])

    # normalize audio signal to -1 to 1
    signal_norm = signal / max(signal)

    bins_per_octave = 36
    features = np.abs(rosa.cqt(signal_norm, sr=parameters['sampling_rate'], fmin=rosa.note_to_hz(parameters['f_min']),
                               bins_per_octave=bins_per_octave,
                               n_bins=int(5*bins_per_octave), hop_length=parameters['hop_size']))

    features = np.abs(rosa.stft(signal_norm, n_fft=256, hop_length=160, win_length=256))

    flatui = ["#ffffff", "#000000"]
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    #plt.imshow(features_fft, origin='lower', aspect='auto')
    #plt.xlabel('Frames')
    #plt.title('FFT Features')
    #plt.ylabel('FFT-Bins')
    #plt.colorbar()
    #plt.savefig('test/FFT_{}_{}.png'.format(parameters['left_context'], parameters['last_filter']))
    #plt.show()

    # CQT Feature Spectrum
    plt.imshow(features, origin='lower', aspect='auto')
    plt.xlabel('Frames')
    plt.title('CQT Features')
    plt.ylabel('CQT-Bins')
    plt.colorbar()
    plt.savefig('test/OriginalFeatures_{}_{}.png'.format(parameters['left_context'], parameters['last_filter']))
    plt.show()

    # show true label
    colormap = plt.imshow(label.transpose(), origin='lower', cmap=my_cmap, aspect='auto', extent=[0, label.shape[0], 0, label.shape[1]])
    plt.title('True Label per Frame')
    plt.xlabel('Frames')
    plt.ylabel('MIDI-Note')
    cbar = plt.colorbar(colormap)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    plt.savefig('test/OriginalLabel_{}_{}.png'.format(parameters['left_context'], parameters['last_filter']))
    plt.show()

    print('Starting predicting values...')
    # show detected classes by the network
    posterior = nn.wav_to_posterior(model, audio_file, parameters)

    # Keep only the classes with the five highest probabilities

    posterior_utterance = nn.to_utterance(posterior, parameter['threshold'], parameter['num_frames_utterance'])

    # Total Accuracy and Accuracy for played notes
    accuracy = nn.calculate_accuracy(posterior_utterance, label)
    accuracy_biased = nn.calculate_biased_accuracy(posterior_utterance, label)

    # Utterance Accuracy
    accuracy_list = []
    #steps = 40
    #values_threshold = np.linspace(0, 1, steps)
    #for i in range(1, 25):
    #    accuracy_threshold = []
    #   for j in range(steps):
    #       accuracy_threshold_utterance = []
    #        posterior_utterance = nn.to_utterance(posterior, values_threshold[j], i)
    #        accuracy_utterance = nn.calculate_accuracy(posterior_utterance, label)
    #        accuracy_threshold.append(accuracy_utterance)
    #    accuracy_list.append(accuracy_threshold)

    #options_array = np.asarray(accuracy_list)
    #np.save("test_settings/threshold_options", options_array)

    #maximum = np.unravel_index(options_array.argmax(), options_array.shape)

    posterior_utterance = nn.to_utterance(posterior_utterance, parameter['threshold'], parameter['num_frames_utterance'])
    accuracy_utterance = nn.calculate_accuracy(posterior_utterance, label)

    accuracy_biased_utterance = nn.calculate_biased_accuracy(posterior_utterance, label)

    # Predicted classes for utterance frame
    colormap = plt.imshow(posterior_utterance.transpose(), origin='lower', cmap=my_cmap, aspect='auto')
    plt.xlabel('Frames')
    plt.title('Predicted Label per Frame in Utterance Form')
    plt.ylabel('MIDI-Note')
    cbar = plt.colorbar(colormap)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    plt.savefig('test/NetworkLabel_Utterance_{}_{}.png'.format(parameters['left_context'], parameters['last_filter']))
    plt.show()

    # Predicted classes for frame
    colormap = plt.imshow(posterior_utterance.transpose(), origin='lower', cmap=my_cmap, aspect='auto', extent=[0, posterior_utterance.shape[0], 0, posterior_utterance.shape[1]])
    plt.xlabel('Frames')
    plt.title('Predicted Label per Frame')
    plt.ylabel('MIDI-Note')
    cbar = plt.colorbar(colormap)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    plt.savefig('test/NetworkLabel_{}_{}.png'.format(parameters['left_context'], parameters['last_filter']))
    plt.show()

    print('--' * 70)
    print("Testing Accuracy of one test sample: {} %".format(accuracy))
    print('--' * 70)
    print("Biased Testing Accuracy of one test sample: {} % of played notes were detected correctly".format(accuracy_biased))
    print('--' * 70)
    print("Utterance Testing Accuracy of one test sample: {} % of played notes were detected correctly".format(accuracy_utterance))
    print('--' * 70)
    print("Biased Utterance Testing Accuracy of one test sample: {} % of played notes were detected correctly".format(accuracy_biased_utterance))
    print('--' * 70)

    return accuracy, accuracy_biased, accuracy_utterance


if __name__ == "__main__":

    # path to database
    basic_path = 'D:/Backup/Trainingsdatenbank/BenjaminDaten/InstrumentNo1/'

    # parameters for the feature extraction
    parameter = {'hop_size': 160,
                  'f_min': 'D2',
                  'bins_per_octave': 36,
                  'num_bins': 168,
                  'left_context': 15,
                  'right_context': 1,
                  'sampling_rate': 16000,
                  'classes': 60,
                  'last_filter': 256,
                  'threshold': 0.2,
                  'num_frames_utterance': 10,
                  }

    # define a name for the model
    model_name = 'model_binary'
    # directory for the model
    model_dir = os.path.join('model', model_name + '.h5')
    if not os.path.exists('model'):
        os.makedirs('model')

    # train neural network and save model to model_dir
    history = nn.train_model(model_dir, basic_path, parameter)

    # test the network with unknown data
    accuracy, accuracy_biased, accuracy_utterance = nn.testing_network(model_dir, basic_path, parameter)

    visualization_network(model_dir, 'test/Random_rSeed101_Noise1.wav', 'test/Random_rSeed101_Noise1_Labels.xls', parameter)

    visualization_network(model_dir, 'test/FeelGood.wav', 'test/FeelGood_Labels.xls', parameter)


