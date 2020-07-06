
import os, glob, json
from random import shuffle, choice
from shutil import copy

from self_driving.edit_distance_polyline import iterative_levenshtein


def get_spine(member):
    spine = json.loads(open(member).read())['sample_nodes']
    return spine

def get_min_distance_from_set(ind, solution):
    distances = list()
    ind_spine = get_spine(ind)
    for road in solution:
        road_spine = get_spine(road)
        distances.append(iterative_levenshtein(ind_spine, road_spine))
    distances.sort()
    return distances[0]

path = r'C:\Users\tig\Documents\deepjanus2020\data\member_seeds\test_roads_bad_130/*.json'
#path2 = r'C:\Users\tig\Documents\deepjanus2020\data\member_seeds\test_roads_bad_/*.json'

all_roads = [filename for filename in glob.glob(path)]
#all_roads += [filename for filename in glob.glob(path2)]

shuffle(all_roads)

roads = all_roads[:40]

starting_point = choice(roads)

original_set = list()
original_set.append(starting_point)

popsize = 12

i = 0
while i < popsize-1:
    max_dist = 0
    for ind in roads:
        dist = get_min_distance_from_set(ind, original_set)
        if dist > max_dist:
            max_dist = dist
            best_ind = ind
    original_set.append(best_ind)
    i += 1

base =r'C:\Users\tig\Documents\deepjanus2020\data\member_seeds\population_LQ11'
if not os.path.exists(base):
    os.makedirs(base)
for index, road in enumerate(original_set):
    dst = os.path.join(base,'seed'+str(index)+'.json')
    copy(road,dst)