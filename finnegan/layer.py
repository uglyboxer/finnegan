"""
Author: Cole Howard
"""

import numpy as np
from scipy.special import expit


class Layer:
    """ A matrix representation of the neurons in a layer
    Inspired by: I Am Trask
                 http://iamtrask.github.io/2015/07/12/basic-python-network/

    Parameters
    ----------
    num_neurons : int
        The number of instances of the class Neuron in each layer.
    vector_size : int
        The number of inputs from the previous layer/original input.  Also,
        equivalent to the length of incoming input vector.

    Attributes
    ----------
    """

    def __init__(self, num_neurons, vector_size):
        self.num_neurons = num_neurons
        np.random.seed(1)
        self.weights = np.random.normal(0, (vector_size**(-.5)/2),
                                        (vector_size, num_neurons))
        self.mr_output = []
        self.mr_input = []
        self.deltas = np.array((vector_size, 1))
        self.l_rate = 1

    def _vector_pass(self, vector):
        """ Takes the vector through the neurons of the layer

        Parameters
        ----------
        vector : numpy array
            The input array to the layer

        Returns
        -------
        numpy array
            The ouput of the layer

        """
        self.mr_input = vector
        x = np.dot(self.weights.T, vector)
        self.mr_output = expit(x)
        return self.mr_output

    def _act_derivative(self, vector):
        """ Calculate the derivative of the activation function

        Parameters
        ----------
        vector : numpy array
            A vector representing the most recent output of a given layer

        Returns
        -------
        numpy array

        """
        return vector * (1 - vector)

    def _layer_level_backprop(self, output, layer_ahead, target_vector,
                              hidden=True):
        """ Calculates the error at this level

        Parameters
        ----------
        hidden : bool
            Whether or not the current layer is hidden (default: True)

        Returns
        -------
        True
            For acknoledgment of execution

        """
        if not hidden:
            self.error = target_vector - self.mr_output
            self.deltas = self.error * self._act_derivative(self.mr_output)
        else:
            self.deltas = layer_ahead.deltas.dot(layer_ahead.weights.T) *\
                          (self._act_derivative(self.mr_output))
        return True

    def _update_weights(self):
        """ Update the weights of each neuron based on the backprop
        calculation """
        self.weights += np.outer(self.mr_input, self.deltas) * self.l_rate
        return
