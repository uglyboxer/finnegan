import pickle
import numpy as np

from finnegan.network import Network

def run_mnist(vector, epochs=0, layers=0, neuron_count=0):
    """ Run Mnist dataset and output a guess list on the Kaggle test_set

    Parameters
    ----------
    epochs : int
        Number of iterations of the the traininng loop for the whole dataset
    layers : int
        Number of layers (not counting the input layer, but does count output
        layer)
    neuron_count : list
        The number of neurons in each of the layers (in order), does not count
        the bias term

    Attributes
    ----------
    target_values : list
        The possible values for each training vector

    """

    with open('train_ext.txt', 'r') as f:
        for line in f:
            ans_train.append(line[0])
            train_set.append(line[1])

    train_set = utils.resample(train_set, random_state=2)
    ans_train = utils.resample(ans_train, random_state=2)

    network = Network(layers, neuron_count, train_set[0])
    network.train(train_set, ans_train, epochs)

    g = open('finnegan/my_net.pickle', 'wb')
    pickle.dump(network, g)
    g.close()
    return None


if __name__ == '__main__':
    epochs = 250
    layers = 3
    layer_list = [75, 73, 10]
    run_mnist([], epochs, layers, layer_list)