""" Author: Cole Howard
Title: Finnegan

An extinsible neural net designed to explore Convolutional Neural Networks and
Recurrent Neural Networks via extensive visualizations.

"""
import sys

import numpy as np
from operator import add

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
        self.possible = [x for x in range(10)]
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

    def _softmax(w, t=1.0):
        """Author: Jeremy M. Stober, edits by Martin Thoma
        Program: softmax.py
        Date: Wednesday, February 29 2012 and July 31 2014
        Description: Simple softmax function.
        Calculate the softmax of a list of numbers w.

        Parameters
        ----------
        w : list of numbers
        t : float

        Return
        ------
        a list of the same length as w of non-negative numbers

        Examples
        --------
        >>> softmax([0.1, 0.2])
        array([ 0.47502081,  0.52497919])
        >>> softmax([-0.1, 0.2])
        array([ 0.42555748,  0.57444252])
        >>> softmax([0.9, -10])
        array([  9.99981542e-01,   1.84578933e-05])
        >>> softmax([0, 10])
        array([  4.53978687e-05,   9.99954602e-01])

        """
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist

    def _backprop(self, guess_vector, target_vector):
        """ Takes the output of the net and initiates the backpropogation

        Parameters
        ----------
        vector : numpy array
            The output from the last layer during a training pass
        target : list
            List of expected values

        Attributes
        ----------
        error_matrix : numpy array


        Returns
        -------
        True
            As evidence of execution

        """
        alt_matrix = np.multiply(guess_vector, -1)
        error_matrix = np.fromiter(map(add, target_vector, alt_matrix, np.float))
        self.layers[-1]._layer_level_backprop(error_matrix)

        """
        for each neuron in output layer:
            error = outα (1 - outα) (Targetα - outα) 
        for each weight IN EACH NEURON in that layer:
            w += l_rate * error * input amout to that weight for n-1 layer

        for each neuron in the layer before:
            error = outA (1 – outA) sum (errorout * weightout) for all nextlayer 
                                        neurons - which is a dot product of the 
                                        two matrices
        for each weight IN EACH NEURON in that layer:
            w += l_rate * error * input amout to that weight for n-1 layer    



        ErrorA = Output A (1 - Output A)(ErrorB WAB + ErrorC WAC) 
        """
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
            for vector, target in dataset:
                target_vector = [0 if x != target else 1 for x in self.possible]
                y = self._pass_through_net(vector)
                z = self._softmax(y)
                self._backprop(z, target_vector)

        # Add in test loop
        # Add in report guesses

        # Check for better sigmoid function


if __name__ == '__main__':
    layers, neuron_count, dataset, epochs = sys.argv[1:4]
    network = Network(layers, neuron_count)
    network.train(dataset, epochs)
