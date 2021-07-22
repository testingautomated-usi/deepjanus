import numpy as np

from distance_calculator import calc_angle_distance, calc_distance_total
from properties import K, K_SD, MISB_TSHD


def evaluate_ff1(A, B):
    dist = calc_distance_total(A, B)
    return dist


def evaluate_misb(pred, true):
    diff = calc_angle_distance(pred, true)
    diff = np.abs(np.degrees(diff))
    ff = MISB_TSHD - diff
    return ff


# calculate the misclassification ff
def evaluate_ff2(eval1, eval2):
    P3 = eval1 * eval2

    if P3 < 0:
        P3 = -1.0
    return P3


def evaluate_aggregate_ff(sparseness, distance):
    result = sparseness - (K_SD * distance)
    return result


def dist_from_nearest_archived(ind, population, k):
    neighbors = list()
    for ind_pop in population:
        if ind_pop.id != ind.id:
            d = eval_dist_individuals(ind, ind_pop)
            if d > 0.0:
                neighbors.append(d)

    if len(neighbors) == 0:
        return -1.0

    neighbors.sort()
    nns = neighbors[:k]
    if k > 1:
        dist = np.mean(nns)
    elif k == 1:
        dist = nns[0]
    if dist == 0.0:
        print(ind)
    return dist


def evaluate_sparseness(ind, individuals):
    N = (len(individuals))
    # Sparseness is evaluated only if the archive is not empty
    # Otherwise the sparseness is 1
    if (N == 0) or (N == 1 and individuals[0] == ind):
        sparseness = 1
    else:
        sparseness = dist_from_nearest_archived(ind, individuals, K)

    return sparseness


def eval_dist_individuals(ind1, ind2):

    a1 = calc_distance_total(ind1.member1.model_params, ind2.member1.model_params)
    a2 = calc_distance_total(ind1.member1.model_params, ind2.member2.model_params)

    b1 = calc_distance_total(ind1.member2.model_params, ind2.member1.model_params)
    b2 = calc_distance_total(ind1.member2.model_params, ind2.member2.model_params)

    a = np.minimum(a1, a2)
    b = np.minimum(b1, b2)
    c = np.minimum(a1, b1)
    d = np.minimum(a2, b2)

    dist = np.mean([a, b, c, d])
    return dist


def is_misb(sample,prediction):
    diff = calc_angle_distance(prediction[0], sample.eye_angles_rad)
    diff = np.abs(np.degrees(diff))
    if MISB_TSHD - diff < 0.0:
        return True
    else:
        return False


def eval_archive_dist(ind1, ind2):
    if ind1.member1.is_misb:
        ind1_correct = ind1.member2.model_params
        ind1_misclass = ind1.member1.model_params
    else:
        ind1_correct = ind1.member1.model_params
        ind1_misclass = ind1.member2.model_params

    if ind2.member1.is_misb:
        ind2_correct = ind2.member2.model_params
        ind2_misclass = ind2.member1.model_params
    else:
        ind2_correct = ind2.member1.model_params
        ind2_misclass = ind2.member2.model_params

    dist1 = calc_distance_total(ind1_correct, ind2_correct)
    dist2 = calc_distance_total(ind1_misclass, ind2_misclass)

    dist = np.mean([dist1, dist2])
    return dist
