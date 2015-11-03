import pytest


from .network import Network
from .layer import Layer
from .neuron import Neuron


class Test_Layer():

    def test_layer_level_backprop(self):
        network = Network(2, [2, 1], [.35, .9])
        network.layers[0].neurons[0]["neuron"].weights = [.1, .8]
        network.layers[0].neurons[1]["neuron"].weights = [.4, .6]
        network.layers[1].neurons[0]["neuron"].weights = [.3, .9]
        network.train([[.35, .9]], [.5], 1)
        assert network.layers[1].neurons[0]["neuron"].weights ==  [0.272392, 0.87305]
        assert network.layers[0].neurons[0]["neuron"].weights ==  [0.09916, 0.7978]
        assert network.layers[0].neurons[1]["neuron"].weights ==  [0.3972, 0.5928]