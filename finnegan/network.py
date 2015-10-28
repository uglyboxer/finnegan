""" Author: Cole Howard
Title: Finnegan

An extinsible neural net designed to explore Convolutional Neural Networks and
Recurrent Neural Networks via extensive visualizations.

"""
import sys

from finnnegan.layer import Layer


class Network:
    """ A multi layer neural net with backpropogation.

    Parameters
    ----------
    layers : int
        Number of layers to use in the network.
    neuron_count : list
        A list of integers that represent the number of neurons present in each
        hidden layer.  (Size of input/output layers are dictated by dataset)

    """

    def __init__(self, layers, neuron_count):
        self.num_layers = layers
        self.neuron_count = neuron_count
        self.dataset = dataset
        self.layers = [Layer(self.neuron_count[x], self.neuron_count[x-1])
                       for x in self.num_layers]

    def _pass_through_net(self, vector):
        """ Sends a vector into the net

        Parameters
        ----------
        vector : numpy array
            A numpy array representing a training input (without the target)

        Returns
        -------
        numpy array
            Output of the last layer in the chain

        """
        x = 0
        while True:
            vector = self.layers[x]._vector_pass(vector)
            x += 1
            if x > len(self.layers):
                return vector

    def _backprop(self, vector):
        """ Takes the output of the net and initiates the backpropogation

        Parameters
        ----------
        vector : numpy array
            The output from the last layer during a training pass

        Returns
        -------
        True
            As evidence of execution

        """
        pass

    def train(self, dataset, epochs):
        """ Runs the training dataset through the network a given number of
        times.

        Parameters
        ----------
        dataset : Numpy nested array
        The collection of training data (vectors and the associated target
            value)

        """
        for x in epochs:
            for vector in dataset:
                y = self._pass_through_net(vector)
                self._backprop(y)


if __name__ == '__main__':
    layers, neuron_count, dataset, epochs = sys.argv[1:4]
    network = Network(layers, neuron_count)
    network.train(dataset, epochs)
