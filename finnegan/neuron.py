"""Author: Cole Howard
Email: uglyboxer@gmail.com
neuron.py is a basic linear neuron, that can be used in a perceptron
Information on that can be found at:
https://en.wikipedia.org/wiki/Perceptron

It was written as a class specifically for network ()

Usage
-----
  From any python script:

      from neuron import Neuron

"""

from math import e
import numpy as np
from operator import add
from scipy.special import expit


class Neuron:
    """ A class model for a single neuron

    Parameters
    ----------
    vector_size : int
        An integer defining the dimensions of an input vector (array)

    Attributes
    ----------
    threshold : float
        The tipping point at which the neuron fires (speifically in relation
        to the dot product of the sample vector and the weight set)
    weights : numpy array
        The "storage" of the neuron.  These are changed with each training
        case and then used to determine if new cases will cause the neuron
        to fire.  o

    """

    def __init__(self, vector_size):

        self.threshold = .5
        self.weights = (np.random.random(vector_size).flatten())
        # self.weights = np.zeros(vector_size).flatten()

    def _sigmoid(self, z):
        """ Calculates the output of a logistic function

        Parameters
        ----------
        z : float
            The dot product of a sample vector (1-d array) and an associated
            weights set (1-d array), which arrives as a scalar

        Returns
        -------
        float
            It will return something between 0 and 1 inclusive
        """

        if -700 < z < 700:
            return 1 / (1 + e ** (-z))
        elif z < -700:
            return 0
        else:
            return 1

    def update_weights(self, error, vector):
        """ Updates the weights stored in the receptors

        Parameters
        ----------
        error : int
            The distance from the expected value of a particular training
            case
        vector : list
            A sample vector

        Attributes
        ----------
        l_rate : float
            A number between 0 and 1, it will modify the error to control
            how much each weight is adjusted. Higher numbers will
            train faster (but risk unresolvable oscillations), lower
            numbers will train slower but be more stable.

        Returns
        -------
        True
            As evidence of execution

        """

        l_rate = .05
        correction = l_rate * error
        corr_matrix = np.multiply(vector, correction)
        self.weights = np.fromiter(map(add, self.weights, corr_matrix), np.float)
        return True

    def fires(self, vector):
        """ Takes an input vector and decides if neuron fires or not

        Parameters
        ----------
        vector : list
            A sample vector

        Returns
        -------
        bool
            Did it fire? True(yes) or False(no)
        float
            The dot product of the vector and weights
        """

        w_with_bias = np.append(self.weights, 1)
        dp = np.dot(vector, w_with_bias)
        if self._sigmoid(dp) > self.threshold:
            return True, dp
        else:
            return False, dp
