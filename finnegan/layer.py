""" Author: Cole Howard

A layer constructor class designed for use in the Finnegan Network model

"""

from finnegan.neuron import Neuron

class Layer():
    """  This is a model for each layer (hidden or visible) of a neural net.

    Parameters
    ----------
    num_neurons : int
        The number of instances of the class Neuron in each layer.

    Attributes
    ----------
    neurons : list
        A list of the actual instances of the Neuron class, based on a
        vector_size = the number of neurons in the preceding layer, or
        the size of the input vector (on the input layer)

    """
    
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [Neuron(vector_size) for x in range(self.num_neurons)]