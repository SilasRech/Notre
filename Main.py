import NeuralNetwork as nn
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import feature_extraction as fe
import os


def visualization_network(model_dir, audio_file, label_dir, parameters):

    model = models.load_model(model_dir)
    model = keras.Sequential([model, keras.layers.Softmax()])

    label = fe.create_multi_labels(label_dir, parameters['classes'])

    features = fe.extract_features(audio_file, parameters)
    features = np.reshape(features, newshape=(features.shape[0], features.shape[1], features.shape[2]))[30]

    flatui = ["#ffffff", "#000000"]
    my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())

    plt.imshow(features, origin='lower', aspect='auto', extent=[0, features.shape[0], 0, features.shape[1]])
    plt.xlabel('Bins')
    plt.ylabel('Frames')
    plt.colorbar()
    plt.savefig('test/OriginalFeatures.png')
    plt.show()

    # show true label
    colormap = plt.imshow(label.transpose(), origin='lower', cmap=my_cmap, aspect='auto', extent=[0, label.shape[0], 0, label.shape[1]])
    plt.xlabel('Frames')
    plt.ylabel('Notes')
    cbar = plt.colorbar(colormap)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
    plt.savefig('test/OriginalLabel.png')
    plt.show()

    # show detected classes by the network
    posterior = nn.wav_to_posterior(model, audio_file, parameters)

    # Keep only the classes with the five highest probabilities
    posterior_sorted = -np.sort(-posterior)
    posterior_cleaned = np.zeros(posterior_sorted.shape)
    for i in range(posterior_sorted.shape[0]):
        # select only one row
        posterior_one_row = posterior_sorted[i, :]
        posterior_cleaned[i, :] = np.where(posterior[i, :] > posterior_one_row[4], 1, 0)

    colormap =  plt.imshow(posterior_cleaned.transpose(), origin='lower', cmap=my_cmap, aspect='auto', extent=[0, posterior_cleaned.shape[0], 0, posterior_cleaned.shape[1]])
    plt.xlabel('Frames')
    plt.ylabel('Notes')
    cbar = plt.colorbar(colormap)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([0, 1])
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
                  'left_context': 15,
                  'right_context': 15,
                  'sampling_rate': 16000,
                  'classes': 120,
                  'last_filter': 256,
                  }

    # define a name for the model
    model_name = 'model_binary'
    # directory for the model
    model_dir = os.path.join('model', model_name + '.h5')
    if not os.path.exists('model'):
        os.makedirs('model')

    history = nn.train_model(model_dir, basic_path, parameter)

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('model/Accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig('model/Loss.png')
    plt.show()

    # Test the network with unknown data
    accuracy = nn.testing_network(model_dir, basic_path, parameter)
    print('--' * 40)
    print("Total Testing Accuracy: {} %".format(accuracy[1]*100))

    visualization_network(model_dir, 'test/Random_rSeed90_Noise3.wav', 'test/Random_rSeed90_Noise3_Labels.xls', parameter)



