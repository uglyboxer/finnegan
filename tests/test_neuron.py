import pytest
import numpy as np
from finnegan.neuron import Neuron

class TestNeuron:

    def test_update_weights(self):
        self.n = Neuron((2, 2))
        self.n.weights = np.ones((2, 2)).flatten()
        vector = np.ones((2,2)).flatten()
        if self.n.update_weights(2, vector):
            assert self.n.weights[0] == 1.1

    def test_fires(self):
        self.n = Neuron((2, 2))
        self.n.weights = np.ones((2,2)).flatten()
        vector = np.ones((2,2)).flatten()
        assert self.n.fires(vector) == (True, 4)
