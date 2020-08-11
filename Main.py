import NeuralNetwork as nn
import global_parameters as gp
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt

import feature_extraction as fe
import os


def visualization_network(model_dir, audio_file, label_dir, parameters):

    model = models.load_model(model_dir)
    model = keras.Sequential([model, keras.layers.Softmax()])

    label = fe.create_multi_labels(label_dir, parameters['classes'])

    features = fe.extract_features(audio_file, parameters)
    features = np.reshape(features, newshape=(features.shape[0], features.shape[1], features.shape[2]))[30]

    plt.imshow(features.T, origin='lower')
    plt.xlabel('Bins')
    plt.ylabel('Frames')
    plt.colorbar()
    plt.savefig('test/OriginalFeatures.png')
    plt.show()


    # show true label
    plt.imshow(label.transpose(), origin='lower')
    plt.xlabel('Frames')
    plt.ylabel('Notes')
    plt.colorbar()
    plt.savefig('test/OriginalLabel.png')
    plt.show()

    # show detected classes by the network
    posterior = nn.wav_to_posterior(model, audio_file, parameters)

    # Keep only the classes with the five highest probabilities
    posterior_sorted = -np.sort(-posterior)
    posterior_cleaned = np.where(posterior > posterior_sorted[4], 1, 0)

    plt.imshow(posterior_cleaned.transpose(), origin='lower')
    plt.xlabel('Frames')
    plt.ylabel('Notes')
    plt.colorbar()
    plt.savefig('test/NetworkLabel.png')
    plt.show()

    return posterior


if __name__ == "__main__":

    # path to database
    basic_path = 'D:/Backup/Trainingsdatenbank/BenjaminDaten/'

    # parameters for the feature extraction
    parameter = {'window_size': 25e-3,
                  'hop_size': 160,
                  'f_min': 'D2',
                  'bins_per_octave': 36,
                  'num_bins': 168,
                  'left_context': 6,
                  'right_context': 6,
                  'sampling_rate': 16000,
                  'classes': 120,
                  'last_filter': 32,
                  }

    # define a name for the model
    model_name = 'model1'
    # directory for the model
    model_dir = os.path.join('model', model_name + '.h5')
    if not os.path.exists('model'):
        os.makedirs('model')

    #history = nn.train_model(model_dir, basic_path, parameter)

    # Plot training & validation accuracy values
    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    #plt.title('Model accuracy')
    #plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper right')
    #plt.savefig('model/Accuracy.png')
    #plt.show()

    # Plot training & validation loss values
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('Model loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.legend(['Train', 'Test'], loc='upper right')
    #plt.savefig('model/Loss.png')
    #plt.show()

    # Test the network with unknown data
    accuracy = nn.testing_network(model_dir, basic_path, parameter)
    print('--' * 40)
    print("Total Testing Accuracy: {}".format(accuracy[1]*100))

    visualization_network(model_dir, 'test/Random_rSeed90_Noise3.wav', 'test/Random_rSeed90_Noise3_Labels.xls', parameter)



