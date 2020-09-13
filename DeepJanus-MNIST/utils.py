import matplotlib
matplotlib.use('Agg')

from os import makedirs
from os.path import exists, basename
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import glob
import keras
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import xml.etree.ElementTree as ET
import potrace
import re
import math

# local imports

import vectorization_tools
from predictor import Predictor
from properties import MODEL, IMG_SIZE, RESULTS_PATH, INTERVAL

NAMESPACE = '{http://www.w3.org/2000/svg}'

# load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0

    return x_reshape


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def print_archive(archive):
    path = RESULTS_PATH+'/archive'
    dst = path + '/'
    if not exists(dst):
        makedirs(dst)
    for i, ind in enumerate(archive):
        filename1 = dst + basename(
            'archived_' + str(i) + '_mem1_l_' + str(ind.member1.predicted_label) + '_seed_' + str(ind.seed))
        plt.imsave(filename1, ind.member1.purified.reshape(28, 28), cmap=cm.gray, format='png')
        loaded_label = (Predictor.predict(ind.member1.purified))
        assert (ind.member1.predicted_label == loaded_label[0])
        assert (ind.member1.predicted_label == Predictor.model.predict_classes(ind.member1.purified))
        np.save(filename1, ind.member1.purified)
        loaded_label = Predictor.predict(np.load(filename1 + '.npy'))[0]
        assert (ind.member1.predicted_label == loaded_label)
        assert (np.array_equal(ind.member1.purified, np.load(filename1 + '.npy')))

        filename2 = dst + basename(
            'archived_' + str(i) + '_mem2_l_' + str(ind.member2.predicted_label) + '_seed_' + str(ind.seed))
        plt.imsave(filename2, ind.member2.purified.reshape(28, 28), cmap=cm.gray, format='png')
        loaded_label = (Predictor.predict(ind.member2.purified))
        assert (ind.member2.predicted_label == loaded_label[0])
        assert (ind.member2.predicted_label == Predictor.model.predict_classes(ind.member2.purified))
        np.save(filename2, ind.member2.purified)
        loaded_label = Predictor.predict(np.load(filename2 + '.npy'))[0]
        assert (ind.member2.predicted_label == loaded_label)
        assert (np.array_equal(ind.member2.purified, np.load(filename2 + '.npy')))


# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v

def print_image(filename, image, cmap=''):
    if cmap == 'gray':
        plt.imsave(filename, image.reshape(28, 28), cmap='gray') 
    else:
        plt.imsave(filename, image.reshape(28, 28)) 
        np.save(filename, image)


def bitmap_count(digit, threshold):    
    bw = np.asarray(digit.purified).copy()    
    #bw = bw / 255.0    
    count = 0    
    for x in np.nditer(bw):
        if x > threshold:
            count += 1    
    return count

def move_distance(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('([\d\.]+),([\d\.]+)\sM\s([\d\.]+),([\d\.]+)')
    segments = pattern.findall(svg_path) 
    if len(segments) > 0:
        dists = [] # distances of moves
        for segment in segments:     
            x1 = float(segment[0])
            y1 = float(segment[1])
            x2 = float(segment[2])
            y2 = float(segment[3])
            dist = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
            dists.append(dist)
        return int(np.sum(dists))
    else:
        return 0

def new_orientation_calc(digit, threshold):
    x = []
    y = []
    bw = np.asarray(digit.purified).copy()  
    for iz,ix,iy,ig in np.ndindex(bw.shape):
        if bw[iz,ix,iy,ig] > threshold:
            x.append([iy])          
            y.append(ix)
    X = np.array(x)
    Y = np.array(y)
    lr = LinearRegression(fit_intercept=True).fit(X, Y)
    normalized_ori = (-lr.coef_ + 2)/4
    # scale to be between 0 and 100
    new_ori = normalized_ori * 100
    return int(new_ori)       


def rescale(solutions, perfs, new_min = 0, new_max = 24):
    max_shape = new_max + 1
    output1 = np.full((max_shape,max_shape), None,dtype=(object))
    output2 = np.full((max_shape,max_shape), np.inf, dtype=(float))
            
    old_min_i = 0
    old_min_j = 0
    old_max_i, old_max_j = solutions.shape[0], solutions.shape[1]

    for (i,j), value in np.ndenumerate(perfs):
        new_i = int(((new_max - new_min) / (old_max_i - old_min_i)) * (i - old_min_i) + new_min)
        new_j = int(((new_max - new_min) / (old_max_j - old_min_j)) * (j - old_min_j) + new_min)
        if value != np.inf:
            if output2[new_i, new_j] == np.inf or value < output2[new_i,new_j]:
                output2[new_i,new_j] = value
                output1[new_i,new_j] = solutions[i,j]
    return output1, output2

def generate_reports(filename, dir_path): 
    filename = filename + ".csv"
    fw = open(filename, 'w')
    cf = csv.writer(fw, lineterminator='\n')

    # write the header
    cf.writerow(["Features","Time","Covered seeds","Filled cells","Filled density", "Misclassified seeds","Misclassification","Misclassification density"])
    
    jsons = [f for f in glob.glob(f"{dir_path}/*.json") if "Bitmaps_Moves" in f]
    id = INTERVAL/60
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)             
            cf.writerow(["Bitmaps,Moves",id,data["Covered seeds"],data["Filled cells"],data["Filled density"],data["Misclassified seeds"],data["Misclassification"],data["Misclassification density"]])
            id += (INTERVAL/60)

    jsons = [g for g in glob.glob(f"{dir_path}/*.json") if "Orientation_Moves" in g]
    id = INTERVAL/60
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)             
            cf.writerow(["Orientation,Moves",id,data["Covered seeds"],data["Filled cells"],data["Filled density"],data["Misclassified seeds"],data["Misclassification"],data["Misclassification density"]])
            id += (INTERVAL/60)

    jsons = [h for h in glob.glob(f"{dir_path}/*.json") if "Bitmaps_Orientation" in h]
    id = INTERVAL/60
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)             
            cf.writerow(["Bitmaps,Orientation",id,data["Covered seeds"],data["Filled cells"],data["Filled density"],data["Misclassified seeds"],data["Misclassification"],data["Misclassification density"]])
            id += (INTERVAL/60)

    fw.close()
