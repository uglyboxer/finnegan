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
    neurons : list
        A list of dictionaries of:
            ("neuron": instance of Neuron obj,
             "forward": list of forward connections (strings),
             "backward": list of backward connections (strings))
    mr_output : list
        An ordered list of the most recent output of the layer

    """

    def __init__(self, num_neurons, incoming_tot):
        self.num_neurons = num_neurons
        self.neurons = [{"neuron": Neuron(incoming_tot),
                         "forward": set(),
                         "backward": set()} for x in range(num_neurons)]
        self.mr_output = []

    def _layer_level_backprop(error, hidden=True):
        """ Calculates the error at this level

        Parameters
        ----------
        error : float
            The error from the layer above (or from the output in last layer)
        hidden : bool
            Whether or not the current layer is hidden (default: True)

        """
        


        error_matrix = [out[i] * (1 - out[i]) 
                               * np.dot(erro_matrix[n+1], n+1_neuron_weight[i])
                                for i, neuron in enumerate(neurons)]

        [[weight += np.multipy(input_vector, l_rate * error_matrix[j]) 
                    for j, weight in enumerate(neuron.weights)] 
                    for neuron in self.neurons]

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
        for neur_inst in self.neurons:
            output.append(neur_inst["neuron"].fires[vector][1])
        self.mr_output = output[:]
        return output






