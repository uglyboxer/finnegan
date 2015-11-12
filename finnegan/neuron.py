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

import numpy as np
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
        self.weights = (np.random.random(vector_size).flatten()-.5)/25
        # self.weights = np.zeros(vector_size).flatten()


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

        # w_with_bias = np.append(self.weights, 1)
        dp = np.dot(vector, self.weights)
        sig = expit(dp)
        if sig > self.threshold:
            return True, sig
        else:
            return False, sig
