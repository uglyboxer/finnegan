""" Author: Cole Howard
Build additional training vectors, which are rotations, skews, and pans, 
of the orginal MNIST dataset
"""

import csv

import numpy as np

from random import randrange

from skimage.transform import rotate

# from matplotlib import pyplot as plt
# from matplotlib import cm

def spin_o_rama(vector, ang):
    vector = rotate(vector, ang, mode='constant')
    return vector

# def visualization(vector):
#     plt.imshow(vector, cmap=cm.Greys_r)
#     plt.axis('off')
#     plt.pause(0.0001)
#     plt.show()




if __name__ == '__main__':
    with open('train.csv', 'r') as f:
        reader = csv.reader(f)
        t = list(reader)
        train = [[int(x) for x in y] for y in t[1:]]

    ans_train = [x[0] for x in train]
    train_set = [x[1:] for x in train]
    ans_train.pop(0)
    train_set.pop(0)

    temp_train = [np.array(elem, dtype=float) for elem in train_set]
    train_set = temp_train

    for _ in range(10):
        for idx, elem in enumerate(train_set):
            x = elem.reshape((28, 28))
            rand_ang = randrange(-120, 120)
            y = spin_o_rama(x, rand_ang).flatten()
            train_set.append(y)
            ans_train.append(ans_train[idx])

    with open('train_ext.txt', 'w') as g:
        for idx, elem in enumerate(train_set):
            g.writeline(ans_train[idx], elem)





    # visualization(x)
    # visualization(spin_o_rama(x, 25))