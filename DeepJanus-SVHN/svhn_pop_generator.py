import random
from os import makedirs
from os.path import exists

import h5py
from tensorflow import keras
import sys
from config import DATASET, POPSIZE, MODEL, MODEL2, EXPLABEL
import numpy as np
from utils import input_reshape, reshape
import time
import csv
import os
from scipy.io import loadmat
from keras.layers import Input
import ModelA
from config import NGEN, \
    POPSIZE, INITIALPOP, \
    RESEEDUPPERBOUND, GENERATE_ONE_ONLY, DATASET, \
    STOP_CONDITION, STEPSIZE, DJ_DEBUG, EXPLABEL, \
    IMG_SIZE, IMG_CHN




def calculate_dist(index_item1, index_item2):
    """
    Compute distance between two inputs,
    the memoization allows higher speed
    :param index_item1: index of the input in the original dataset(int)
    :param index_item2: index of the input in the original dataset(int)
    :return: distance between the inputs
            """
    def memoized_dist(index_item1, index_item2):
        index_str = tuple(sorted([index_item1, index_item2]))
        if index_str in cache:
            return cache[index_str]
        ind = reshape(x_train[index_item1])
        initial_ind = reshape(x_train[index_item2])
        result = np.linalg.norm(initial_ind - ind)
        cache.update({index_str : result})
        return result

    return memoized_dist(index_item1, index_item2)


def get_min_distance_from_set(ind_index, initial_pop):
    """
    returns the minimum distance of an input from a set of inputs
    :param ind_index: index of the input (int)
    :param initial_pop: list of indexes of inputs([int])
    :return: minimum distance of the input from the set (float)
    """
    distances = list()
    for initial_ind_index in initial_pop:
        d = calculate_dist(ind_index, initial_ind_index)
        distances.append(d)
    distances.sort()
    return distances[0]


def generate(diversity=True):
    """
    Generates a dataset of inputs
    as h5 file saved at the path DATASET
    of size POPSIZE
    correctly classified by MODEL1 and MODEL2
    having label EXPLABEL (any label if EXPLABEL is None)
    :param diversity: (bool) if True greedily builds a diverse dataset
    """
    y_t = y_train.reshape(len(y_train), )
    gtruth_set = np.argwhere(y_t == EXPLABEL)

    print("Evaluating with model 1")
    # Load the pre-trained model.
    input_shape = (IMG_SIZE, IMG_SIZE, IMG_CHN)

    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)

    model = ModelA.ModelA(input_tensor)
    print("Loaded model from disk")

    prediction = np.argmax(model.predict(x_train), axis=1)
    #prediction = prediction.reshape(len(x_train), 1)

    correctly_predicted = np.argwhere(prediction == y_t)

    correct_set = np.intersect1d(correctly_predicted, gtruth_set)


    if diversity:
        print("Diverse set generation")
        print("Calculating diverse set")
        original_set = list()
        print("Finding element 1")
        # The first element is chosen randomly
        starting_point = random.choice(correct_set)
        original_set.append(starting_point.item())
        correct_set = np.setdiff1d(correct_set, original_set)

        popsize = POPSIZE
        i = 0
        while i < popsize-1:
            print("Finding element "+str(i+2))
            max_dist = 0
            for ind in correct_set:
                dist = get_min_distance_from_set(ind, original_set)
                if dist > max_dist:
                    max_dist = dist
                    best_ind = ind
            original_set.append((best_ind))
            correct_set = np.setdiff1d(correct_set, best_ind).tolist()
            i += 1
        xn = x_train[original_set]
        yn = y_train[original_set]
    else:
        print("Random set generation")
        xn = x_train[correct_set]
        yn = y_t[correct_set]

    #print('Checking correctness...')
    #for item in range(len(xn)):
    #   assert(model.predict_classes(reshape(xn[int(item)])) == yn[int(item)])
    #   print (model.predict_classes(reshape(xn[int(item)])))

    dst = "original_dataset"
    if not exists(dst):
        makedirs(dst)
    print('Creating dataset...')
    f = h5py.File(DATASET, 'w')
    f.create_dataset("xn", shape=(len(xn),32,32,3), data=xn)
    f.create_dataset("yn", data=yn)
    f.close()

    print('Done')


if __name__ == "__main__":
    time1 = time.time()
    # load the MNIST dataset
    cache = dict()
    datasetLoc = '/home/vin/PycharmProjects/dola/DistributionAwareDNNTesting/SVHN_dx/dataset/test_32x32.mat'
    test_data = loadmat(datasetLoc)
    x_train = np.array(test_data['X'])
    # Normalize data.
    x_train = np.moveaxis(x_train, -1, 0)

    y_train = test_data['y']
    y_train[y_train == 10] = 0

    #x_train = x_train[:30]
    #y_train = y_train[:30]

    generate(diversity=False)

    time2 = time.time()
    elapsed_time = (time2 - time1)

    info_file = 'pop_info.csv'

    if os.path.exists(info_file):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(info_file, append_write) as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow([str(elapsed_time)])
    print("Time spent: " + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

