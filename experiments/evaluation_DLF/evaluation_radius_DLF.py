import glob
import os
import csv

import numpy as np
import imageio


def euclidean_distance(imageA, imageB):
    dist = np.linalg.norm(imageA - imageB)
    return dist


def get_radius_reference(solution, reference):
    # Calculate the distance between each misclassified digit and the seed (mindist metric)
    min_distances = list()
    for ind in solution:
        dist = euclidean_distance(ind, reference)
        min_distances.append(dist)
    mindist = np.mean(min_distances)
    return mindist


def create_csv_results_file(file_name):
    if os.path.exists(file_name):
        open(file_name, "r+")
    else:
        open(file_name, "w")


def create_csv_results_file_header(file_name):
    create_csv_results_file(file_name)
    if file_name is not None:
        with open(file_name, mode='w', newline='') as result_file:
            csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            fieldnames = ["Model", "Run", "Diameter_L2_Max", "Diameter_L2_Avg", "Diameter_MSE_Max", "Diameter_MSE_Avg"]
            writer = csv.DictWriter(result_file, fieldnames=fieldnames)
            writer.writeheader()
            result_file.flush()
            result_file.close()


def writeCsvLine(filename, row):
    if filename is not None:
        with open(filename, mode='a') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                lineterminator='\n')
            writer.writerow(row)
            result_file.flush()
            result_file.close()
    else:
        create_csv_results_file_header(filename)


models = ["LQ", "HQ"]
runs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    v = v.reshape(v.shape[0], 28, 28, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v


reference_filename = 'cinque.npy'
reference = np.load(reference_filename)


class Individual:
    def __init__(self, filename):
        im = imageio.imread(filename)
        im = np.asarray(im)
        im = reshape(im)
        self.img = im
        self.seed = os.path.basename(filename).split("_")[0]


for model in models:
    for run in runs:
        path = os.path.join('exp-DLF-FSE', model, run)
        # read the image in the archive labeled with 5
        image_list = []
        filelist = [p for p in glob.glob(path + '/*.png') if "_out" in p]
        for filename in filelist:
            ind = Individual(filename).img
            image_list.append(ind)

        print('Read %d images...' % len(image_list))

        print("Calculating radius for %s model (run %s)" % (model, run))
        radius = get_radius_reference(image_list, reference)

        writeCsvLine("Table3-DLF.csv", ["DLF_MNIST-" + model, run, radius])
