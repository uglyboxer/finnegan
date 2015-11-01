""" Author: Cole Howard
Title: Finnegan

An extinsible neural net designed to explore Convolutional Neural Networks and
Recurrent Neural Networks via extensive visualizations.

"""
# import sys
# import ipdb
import numpy as np
from sklearn import datasets, utils

from layer import Layer

from time import sleep

from matplotlib import cm
from matplotlib import pyplot as plt


class Network:
    """ A multi layer neural net with backpropogation.

    Parameters
    ----------
    layers : int
        Number of layers to use in the network.
    neuron_count : list
        A list of integers that represent the number of neurons present in each
        hidden layer.  (Size of input/output layers are dictated by dataset)
    vector : list
        Example vector to get size of initial input

    """

    def __init__(self, layers, neuron_count, vector):
        self.num_layers = layers
        self.neuron_count = neuron_count
        # self.dataset = dataset
        self.possible = [x for x in range(10)]
        self.layers = [Layer(self.neuron_count[x], self.neuron_count[x-1]) if
                       x > 0 else Layer(self.neuron_count[x], len(vector))
                       for x in range(self.num_layers)]

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
            # ipdb.set_trace(vector[0])
            vector = self.layers[x]._vector_pass(vector)
            x += 1
            if x >= len(self.layers):
                return vector

    def _softmax(self, w, t=1.0):
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
        # e_calc = np.exp(np.array(w) / t)
        # print(e_calc)
        # dist = e_calc / np.sum(e_calc)
        # print(sum(dist))
        # return dist
        e_x = np.exp(w - np.max(w))
        out = e_x / e_x.sum()
        return out

    def _backprop(self, guess_vector, target_vector):
        """ Takes the output of the net and initiates the backpropogation

        In output layer:
          generate error matrix [(out * (1-out) * (Target-out)) for each neuron]
          update weights matrix [[+= l_rate * error_entry * input TO that
          amount] for each neuron ]

        In hidden layer
          generate error matrix [out * (1-out) * dotproduct(entry in n+1 error
          matrix, n+1 weight of that entry)] update weights matrix [[+= l_rate
          for each weight] for each neuron]

        Parameters
        ----------
        guess_vector : numpy array
            The output from the last layer during a training pass
        target_vector : list
            List of expected values

        Attributes
        ----------


        Returns
        -------
        True
            As evidence of execution

        """
        backwards_layer_list = list(reversed(self.layers))
        for i, layer in enumerate(backwards_layer_list):
            if i == 0:
                hidden = False
                layer_ahead = None
            else:
                hidden = True
                # -1 because layers was reversed for index
                layer_ahead = backwards_layer_list[i-1]

            if layer._layer_level_backprop(guess_vector, layer_ahead, target_vector, hidden):
                continue
            else:
                print("Backprop failed on layer: " + str(i))
        for layer in self.layers:
            layer.error_matrix = []
        return True

    def train(self, dataset, answers, epochs):
        """ Runs the training dataset through the network a given number of
        times.

        Parameters
        ----------
        dataset : Numpy nested array
        The collection of training data (vectors and the associated target
            value)

        """
        for x in range(epochs):
            for vector, target in zip(dataset, answers):
                target_vector = [0 if x != target else 1 for x in self.possible]
                y = self._pass_through_net(vector)
                z = self._softmax(y)
                # print(z, target)
                self._backprop(z, target_vector)
        
        # Add in test loop
        # Add in report guesses

        # Check for better sigmoid function


def visualization(vector, vector_name):
    y = np.reshape(vector, (8,8))
    plt.imshow(y, cmap=cm.Greys_r)
    plt.suptitle(vector_name)
    plt.axis('off')
    plt.pause(0.0001)
    plt.show()

if __name__ == '__main__':
    # layers, neuron_count, dataset, epochs = sys.argv[1:5]

    # Imported from linear_neuron
    temp_digits = datasets.load_digits()
    digits = utils.resample(temp_digits.data, random_state=0)
    temp_answers = utils.resample(temp_digits.target, random_state=0)
    # images = utils.resample(temp_digits.images, random_state=0)
    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_of_training_vectors = 1250 
    answers, answers_to_test, validation_answers = temp_answers[:num_of_training_vectors], temp_answers[num_of_training_vectors:num_of_training_vectors+500], temp_answers[num_of_training_vectors+500:]
    training_set, testing_set, validation_set = digits[:num_of_training_vectors], digits[num_of_training_vectors:num_of_training_vectors+500], digits[num_of_training_vectors+500:]


# look at round where last backprop runs.  Maybe peel off one iteration?
# Get over it and append bias to forward pass, but not backward pass 
    ###########
    # visualization(training_set[10], answers[10])
    # visualization(training_set[11], answers[11])
    # visualization(training_set[12], answers[12])
    epochs = 20
    layers = 1
    neuron_count = [10]
    network = Network(layers, neuron_count, training_set[0])
    network.train(training_set, answers, epochs)
