""" Author: Cole Howard
Title: Finnegan

An extinsible neural net designed to explore Convolutional Neural Networks and
Recurrent Neural Networks via extensive visualizations.

"""
# import sys
# import ipdb
import csv
import numpy as np
from sklearn import datasets, utils

from layer import Layer

from time import sleep

#from matplotlib import cm
#from matplotlib import pyplot as plt


class Network:
    """ A multi layer neural net with backpropogation.

    Parameters
    ----------
    layers : int
        Number of layers to use in the network.
    neuron_count : list
        A list of integers that represent the number of neurons present in each
        hidden layer.  (Size of input/output layers are dictated by dataset)
    vector : list
        Example vector to get size of initial input

    """

    def __init__(self, layers, neuron_count, vector):
        self.num_layers = layers
        self.neuron_count = neuron_count
        # self.dataset = dataset
        self.possible = [x for x in range(10)]
        self.layers = [Layer(self.neuron_count[x], self.neuron_count[x-1]) if
                       x > 0 else Layer(self.neuron_count[x], len(vector))
                       for x in range(self.num_layers)]

    def _pass_through_net(self, vector):
        """ Sends a vector into the net

        Parameters
        ----------
        vector : numpy array
            A numpy array representing a training input (without the target)

        Returns
        -------
        numpy array
            Output of the last layer in the chain

        """
        x = 0
        while True:
            # ipdb.set_trace(vector[0])
            vector = self.layers[x]._vector_pass(vector)
            x += 1
            if x >= len(self.layers):
                return vector

    def _softmax(self, w, t=1.0):
        """Author: Jeremy M. Stober, edits by Martin Thoma
        Program: softmax.py
        Date: Wednesday, February 29 2012 and July 31 2014
        Description: Simple softmax function.
        Calculate the softmax of a list of numbers w.

        Parameters
        ----------
        w : list of numbers
        t : float

        Return
        ------
        a list of the same length as w of non-negative numbers

        Examples
        --------
        >>> softmax([0.1, 0.2])
        array([ 0.47502081,  0.52497919])
        >>> softmax([-0.1, 0.2])
        array([ 0.42555748,  0.57444252])
        >>> softmax([0.9, -10])
        array([  9.99981542e-01,   1.84578933e-05])
        >>> softmax([0, 10])
        array([  4.53978687e-05,   9.99954602e-01])

        """
        # e_calc = np.exp(np.array(w) / t)
        # print(e_calc)
        # dist = e_calc / np.sum(e_calc)
        # print(sum(dist))
        # return dist
        e_x = np.exp(w - np.max(w))
        out = e_x / e_x.sum()
        return out

    def _backprop(self, guess_vector, target_vector):
        """ Takes the output of the net and initiates the backpropogation

        In output layer:
          generate error matrix [(out * (1-out) * (Target-out)) for each neuron]
          update weights matrix [[+= l_rate * error_entry * input TO that
          amount] for each neuron ]

        In hidden layer
          generate error matrix [out * (1-out) * dotproduct(entry in n+1 error
          matrix, n+1 weight of that entry)] update weights matrix [[+= l_rate
          for each weight] for each neuron]

        Parameters
        ----------
        guess_vector : numpy array
            The output from the last layer during a training pass
        target_vector : list
            List of expected values

        Attributes
        ----------


        Returns
        -------
        True
            As evidence of execution

        """
        backwards_layer_list = list(reversed(self.layers))
        for i, layer in enumerate(backwards_layer_list):
            if i == 0:
                hidden = False
                layer_ahead = None
            else:
                hidden = True
                # -1 because layers was reversed for index
                layer_ahead = backwards_layer_list[i-1]

            if layer._layer_level_backprop(guess_vector, layer_ahead, target_vector, hidden):
                continue
            else:
                print("Backprop failed on layer: " + str(i))
        for layer in self.layers:
            layer.error_matrix = []
        return True

    def train(self, dataset, answers, epochs):
        """ Runs the training dataset through the network a given number of
        times.

        Parameters
        ----------
        dataset : Numpy nested array
        The collection of training data (vectors and the associated target
            value)

        """
        for x in range(epochs):
            for vector, target in zip(dataset, answers):
                target_vector = [0 if x != target else 1 for x in self.possible]
                y = self._pass_through_net(vector)
                z = self._softmax(y)
                self._backprop(z, target_vector)
     
        # Add in test loop
        # Add in report guesses

        # Check for better sigmoid function

    def run_unseen(self, test_set):
        """ Makes guesses on the unseen data, and switches over the test
        answers to validation set if the bool is True

        For each vector in the collection, each neuron in turn will either
        fire or not.  If a vector fires, it is collected as a possible
        correct guess.  Not firing is collected as well, in case
        there an no good guesses at all.  The method will choose the
        vector with the highest dot product, from either the fired list
        or the dud list.

        Parameters
        ----------
        validation : bool
            Runs a different set of vectors through the guessing
            process if validation is set to True

        Returns
        -------
        list
            a list of ints (the guesses for each vector)
        """
        guess_list = []
        for idy, vector in enumerate(test_set):
            temp = self._pass_through_net(vector)
            guess_list.append(temp.index(max(temp)))
        return guess_list




    def report_results(self, guess_list, answers):
        """ Reports results of guesses on unseen set

        Parameters
        ----------
        guess_list : list

        """


        successes = 0
        for idx, item in enumerate(guess_list):
            if answers[idx] == item:
                successes += 1
        print("Successes: {}  Out of total: {}".format(successes,
              len(guess_list)))
        print("For a success rate of: ", successes/len(guess_list))


def visualization(vector, vector_name):
    y = np.reshape(vector, (28,28))
    plt.imshow(y, cmap=cm.Greys_r)
    plt.suptitle(vector_name)
    plt.axis('off')
    plt.pause(0.0001)
    plt.show()

if __name__ == '__main__':
    # layers, neuron_count, dataset, epochs = sys.argv[1:5]

    # Imported from linear_neuron
    # temp_digits = datasets.load_digits()
    # digits = utils.resample(temp_digits.data, random_state=0)
    # temp_answers = utils.resample(temp_digits.target, random_state=0)
    # # images = utils.resample(temp_digits.images, random_state=0)
    # target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # num_of_training_vectors = 1258
    # answers, answers_to_test, validation_answers = temp_answers[:num_of_training_vectors], temp_answers[num_of_training_vectors:num_of_training_vectors+270], temp_answers[num_of_training_vectors+270:]
    # training_set, testing_set, validation_set = digits[:num_of_training_vectors], digits[num_of_training_vectors:num_of_training_vectors+270], digits[num_of_training_vectors+270:]

    # For Kaggle submission

    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        t = list(reader)
        train = [[int(x) for x in y] for y in t[1:]]

    # print(train_set[1][0], train_set[1][1:])
    with open('test.csv', 'r') as f:
        reader = csv.reader(f)
        raw_nums = list(reader)
        test_set = [[int(x) for x in y] for y in raw_nums[1:]]

    ans_train = [x[0] for x in train]
    train_set = [x[1:] for x in train]
    ans_train.pop(0)
    train_set.pop(0)



    # temp_digits = datasets.load_digits()
    # digits = utils.resample(train_set, random_state=0)
    # temp_answers = utils.resample(ans_train, random_state=0)

    target_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # num_of_training_vectors = 29400
    # answers, answers_to_test, validation_answers = temp_answers, temp_answers[num_of_training_vectors:num_of_training_vectors+6300], temp_answers[num_of_training_vectors+6300:]
    # training_set, testing_set, validation_set = digits, test_set, digits[num_of_training_vectors+6300:]
    # epoch = 75
    # network = Network(target_values, training_set, answers, epoch, testing_set,
    #                   answers_to_test, validation_set, validation_answers)
    # network.learn_run()
    # network.report_results(network.run_unseen())
    # network.report_results(network.run_unseen(True), True)
    

# look at round where last backprop runs.  Maybe peel off one iteration?
# Get over it and append bias to forward pass, but not backward pass 
    ###########
    # visualization(train_set[10], ans_train[10])
    # visualization(train_set[11], ans_train[11])
    # visualization(train_set[12], ans_train[12])
    epochs = 1
    layers = 1
    neuron_count = [10]
    network = Network(layers, neuron_count, train_set[0])
    network.train(train_set, ans_train, epochs)
    
    guess_list = network.run_unseen(test_set)
    with open('digits.txt', 'w') as d:
        for elem in guess_list:
            d.write(str(elem)+'\n')

    # guess_list = network.run_unseen(testing_set)
    # network.report_results(guess_list, answers_to_test)
    # valid_list = network.run_unseen(validation_set)
    # network.report_results(valid_list, validation_answers)
