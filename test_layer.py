import numpy as np
import unittest

from finnegan.layer import Layer


class TestClass(unittest.TestCase):
    def test_vector_pass(self):
        layer = Layer(2, 2)
        layer.weights = .5 * np.ones((2, 2))
        assert list(layer._vector_pass([0, 0])) == [.5, .5]
        self.assertAlmostEqual(layer._vector_pass([1, 1])[0], 0.73105858,
                               places=8)
        self.assertAlmostEqual(layer._vector_pass([1, 1])[1], 0.73105858,
                               places=8)

    def test_act_derivative(self):
        layer = Layer(2, 2)
        assert np.array_equal(layer._act_derivative(np.array([1, 1])), np.array([0, 0]))
        assert np.array_equal(layer._act_derivative(np.array([2, 2])), np.array([-2, -2]))

    def test_update_weights(self):
        layer = Layer(2, 2)
        layer.weights = .5 * np.ones((2, 2))
        layer.deltas = np.ones((2, 2)) * .2
        layer.mr_input = np.array([1, 1])
        x = np.ones((2, 2)) * .9
        layer._update_weights()
        assert np.array_equal(layer.weights, x)

    def test_layer_level_backprop(self):
        # For output layer
        layer = Layer(2, 2)
        layer.weights = .5 * np.ones((2, 2))
        layer.mr_output = layer._vector_pass(np.array([1, 1]))
        layer._layer_level_backprop(layer.mr_output, None, [1, 1], False)
        layer._update_weights()
        assert np.allclose(layer.weights, np.array([[.605754, .605754],
                                                    [.605754, .605754]]))
