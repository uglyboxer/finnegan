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
    incoming_tot : int
        The number of inputs from the previous layer/original input.  Also,
        equivalent to the length of incoming input vector.

    Attributes
    ----------
    neurons : dict
        A dictionary with keys that are ints (just a count of total neurons in
        the layer), with values that are dictionaries of:
            ("neuron": instance of Neuron obj,
             "forward": list of forward connections (strings),
             "backward": list of backward connections (strings))

    """

    def __init__(self, num_neurons, incoming_tot):
        self.num_neurons = num_neurons
        self.neurons = {x: {"neuron": Neuron(incoming_tot),
                            "forward": set(),
                            "backward": set()} for x in range(num_neurons)}

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
        for neuron in self.neurons:
            output.append(neuron.fires[vector][1])
        return output






