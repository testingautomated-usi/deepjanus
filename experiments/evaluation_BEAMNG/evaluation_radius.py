import glob
import json
import os
import numpy as np
import csv
from edit_distance_polyline import iterative_levenshtein


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

    return None


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


def get_radius_edit_straight(solution, reference):
    # Calculate the distance between each misclassified digit and the seed (mindist metric)
    radius_list = list()
    for ind in solution:
        dist = iterative_levenshtein(ind, reference)
        radius_list.append(dist)
    mindist = np.mean(radius_list)
    return mindist


models = ["LQ", "HQ"]
runs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

reference_filename = 'straight_road'

straight_road_file = json.loads(open("straight_road").read())
straight_road = np.array(straight_road_file["sample_nodes"])[:,:2]

for model in models:
    for run in runs:
        path = os.path.join('experiment-BEAMNG-FSE', model, run, "archive")

        # read the json in the archive
        road_list = []
        for filename in glob.glob(path + '/*.json'):
            desc1 = json.loads(open(filename).read())
            if desc1['m1']['distance_to_boundary'] >= 0.0:
                spine1 = (desc1['m1']['sample_nodes'])
            else:
                spine1 = (desc1['m2']['sample_nodes'])

            road_list.append(spine1)

        print('Read %d files...' % len(road_list))

        print("Calculating distances for %s model (run %s)" % (model, run))

        radius = get_radius_edit_straight(road_list, straight_road)

        writeCsvLine("Table3_BeamNG_radius_in.csv",
                 ["radius-in-" + model, run, radius])


for model in models:
    for run in runs:
        path = os.path.join('experiment-BEAMNG-FSE', model, run, "archive")

        # read the json in the archive
        road_list = []
        for filename in glob.glob(path + '/*.json'):
            desc1 = json.loads(open(filename).read())
            if desc1['m1']['distance_to_boundary'] < 0.0:
                spine1 = (desc1['m1']['sample_nodes'])
            else:
                spine1 = (desc1['m2']['sample_nodes'])

            road_list.append(spine1)

        print('Read %d files...' % len(road_list))

        print("Calculating distances for %s model (run %s)" % (model, run))

        radius = get_radius_edit_straight(road_list, straight_road)

        writeCsvLine("Table3_BeamNG_radius_out.csv",
                 ["radius-out-" + model, run, radius])


