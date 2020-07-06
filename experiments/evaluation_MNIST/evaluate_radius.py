import numpy as np
import glob
import os
import csv

models = ["LQ", "HQ"]
runs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

folder = r'control/'
experiments_folder = r'experiment-MNIST-FSE'
reference_filename = 'ref_digit/cinque_rp.npy'
reference = np.load(reference_filename)


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



for m in models:
    for r in runs:
        path = os.path.join(experiments_folder, m, r, 'results','archive')
        filelist = [p for p in glob.glob(path + '/*.npy') if "l_5" in p]

        image_list = []
        for filename in filelist:
            solution = np.load(filename)
            image_list.append(solution)

        print("Calculating inner radius: "+r)
        radius = get_radius_reference(image_list, reference)

        print(radius)

        writeCsvLine("Table3-MNIST-in.csv", ["DJ_MNIST-IN " + m + ' ' + str(radius)])


for m in models:
    for r in runs:
        path = os.path.join(experiments_folder, m, r, 'results','archive')

        filelist = [p for p in glob.glob(path + '/*.npy') if "l_5" not in p]

        image_list = []
        for filename in filelist:
            solution = np.load(filename)
            image_list.append(solution)

        print("Calculating outer radius: "+r)
        radius = get_radius_reference(image_list, reference)

        print(radius)

        writeCsvLine("Table3-MNIST-out.csv", ["DJ_MNIST-OUT " + m + ' ' + str(radius)])
