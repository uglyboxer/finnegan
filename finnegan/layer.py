""" Author: Cole Howard

A layer constructor class designed for use in the Finnegan Network model

"""
import numpy as np

from finnegan.neuron import Neuron


class Layer():
    """  This is a model for each layer (hidden or visible) of a neural net.

    Parameters
    ----------
    num_neurons : int
        The number of instances of the class Neuron in each layer.
    incoming_tot : int
        The number of inputs from the previous layer/original input.  Also,
        equivalent to the length of incoming input vector.

    Attributes
    ----------
    neurons : list
        A list of dictionaries of:
            ("neuron": instance of Neuron obj,
             "forward": list of forward connections (strings),
             "backward": list of backward connections (strings))
    error_matrix : list
        The collection of floats that represent the error per neuron in the
        layer.
    mr_input : list
        The most recent input vector to pass through #haxor alert
    mr_output : list
        An ordered list of the most recent output of the layer
    l_rate : float
        A constant determining the the learning rate of the network
        Bigger learns faster - risks creating oscilations
        Smaller is slower - risks settling in local minimums

    """

    def __init__(self, num_neurons, incoming_tot):
        self.num_neurons = num_neurons
        self.neurons = [{"neuron": Neuron(incoming_tot),
                         "forward": set(),
                         "backward": set()} for x in range(num_neurons)]
        self.error_matrix = []
        self.mr_input = []
        self.mr_output = []
        self.l_rate = .1

    def _layer_level_backprop(self, layer_ahead, hidden=True):
        """ Calculates the error at this level

        Parameters
        ----------
        hidden : bool
            Whether or not the current layer is hidden (default: True)

        """
        if not hidden:
            self.error_matrix = [self.mr_output[i] * (1 - self.mr_output[i])
                                 for i, neuron in enumerate(self.neurons)]
            for neuron in self.neurons:
                neuron.weights = [weight + np.multipy(self.mr_input, 
                                                      (self.l_rate * 
                                                      self.error_matrix[j]))
                                  for j, weight in enumerate(neuron.weights)]

        else:
            self.error_matrix = [self.mr_output[i] * (1 - self.mr_output[i])
                                 * np.dot(layer_ahead.error_matrix,
                                 layer_ahead.neurons[i].weight)
                                 for i, neuron in enumerate(self.neurons)]

            for neuron in self.neurons:
                neuron.weights = [weight + np.multipy(self.mr_input, 
                                                      (self.l_rate * 
                                                      self.error_matrix[j]))
                                  for j, weight in enumerate(neuron.weights)]

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
        output = []
        self.mr_input = vector
        for neur_inst in self.neurons:
            output.append(neur_inst["neuron"].fires[vector][1])
        self.mr_output = output[:]
        return output






