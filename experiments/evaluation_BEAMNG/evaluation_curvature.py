import glob
import json
import os
import numpy as np
import csv
from curvature import findCircle

models = ["LQ", "HQ"]
runs = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
threshold = 47


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


all_curv = []
for model in models:
    for run in runs:
        path = os.path.join('experiment-BEAMNG-FSE', model, run, "archive")

        # read the json in the archive
        road_list = []
        valid = []
        invalid = []
        for filename in glob.glob(path + '/*.json'):
            desc1 = json.loads(open(filename).read())
            if desc1['m1']['distance_to_boundary'] < 0.0:
                spine1 = (desc1['m1']['sample_nodes'])
            else:
                spine1 = (desc1['m2']['sample_nodes'])

            road_list.append(spine1)

        print('Read %d files...' % len(road_list))

        print("Calculating distances for %s model (run %s)" % (model, run))

        mincurv = []
        for index in range(len(road_list)):
            curvatures = []
            for j in range(10):
                road = np.array(road_list[index])[:, :2][int(j)::10 + int(j)]
                res = list(map(list, zip(list(road), list(road[1:]), list(road[2:]))))

                for i in range(len(res)):
                    joined_lists = [*res[i][0], *res[i][1], *res[i][2]]
                    curvatures.append(
                        findCircle(joined_lists[0], joined_lists[1], joined_lists[2], joined_lists[3], joined_lists[4],
                                   joined_lists[5]))
            radius = np.min(curvatures)*3.280839895
            mincurv.append(radius)
            writeCsvLine("Table2_curvatures_raw.csv",
                         [model, radius])

            all_curv.append(radius)
        curvature = np.mean(mincurv)
        for curv in mincurv:
            if curv < threshold:
                invalid.append(curv)
            else:
                valid.append(curv)

        writeCsvLine("Table2_curvature.csv",
                 ["BEAMNG-out" + model, run, len(valid), len(invalid), curvature])

        print(all_curv)






