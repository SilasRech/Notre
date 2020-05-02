import PreProcessing as pp
import NeuralNetwork as nn
import global_parameters

training, testing, train_label, test_label = pp.batchmaking()

nn.neural_network(training, testing, train_label, test_label)
x=1