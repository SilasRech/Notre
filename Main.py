import PreProcessing as pp
import NeuralNetwork as nn
import global_parameters as gp
import numpy as np
print("Starting Process")


def load_data():
    # i is index for loading data
    i = 400
    training = np.load(gp.label_train_data.format(i))
    train_label = np.load(gp.label_train_labels.format(i))
    testing = np.load(gp.label_test_data.format(i))
    test_label = np.load(gp.label_test_labels.format(i))

    return training, testing, train_label, test_label


def main():

    if gp.loaded == 0:
        print("No Data Loaded, beginning batch making")
        training, testing, train_label, test_label = pp.batchmaking()
    else:
        print("Starting Loading Ready Data")
        training, testing, train_label, test_label = load_data()
        print("Finished Loading Data")

    nn.neural_network(training, testing, train_label, test_label)


if __name__ == "__main__":
    main()
    x=1