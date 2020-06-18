import PreProcessing as pp
import NeuralNetwork as nn
import global_parameters as gp
import numpy as np
print("Starting Process")


def load_data():
    train = np.load(gp.label_train_data)
    train_label = np.load(gp.label_train_labels)
    test = np.load(gp.label_test_data)
    test_label = np.load(gp.label_test_labels)
    eval = np.load(gp.label_eval_data)
    eval_label = np.load(gp.label_eval_labels)

    return train, eval, test, train_label, eval_label, test_label


def main():

    if gp.loaded == 0:
        print("No Data Loaded, Beginning Batch Making")
        train, eval, test, train_label, eval_label, test_label = pp.batchmaking()
    else:
        print("Starting Loading Ready Data")
        train, eval, test, train_label, eval_label, test_label = load_data()
        print("Finished Loading Data")

    nn.neural_network(train, eval, test, train_label, eval_label, test_label)


if __name__ == "__main__":
    main()
    x=1