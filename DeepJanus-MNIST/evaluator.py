import numpy as np
import utils
from properties import K, K_SD, EXPECTED_LABEL


def evaluate_ff1(A, B):
    dist = utils.get_distance(A, B)
    return dist


# calculate the misclassification ff
def evaluate_ff2(P_class_A, P_notclass_A, P_class_B, P_notclass_B):
    P1 = P_class_A - P_notclass_A
    P2 = P_class_B - P_notclass_B
    P3 = P1 * P2

    # TODO test
    if P3 < 0:
        P3 = -0.1
    return P3


def evaluate_aggregate_ff(sparseness, distance):
    result = sparseness - (K_SD * distance)
    return result


def dist_from_nearest_archived(ind, population, k):
    neighbors = list()
    for ind_pop in population:
        if ind_pop != ind:
            d = eval_dist_individuals(ind, ind_pop)
            neighbors.append(d)
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

    a1 = utils.get_distance(ind1.member1.purified, ind2.member1.purified)
    a2 = utils.get_distance(ind1.member1.purified, ind2.member2.purified)

    b1 = utils.get_distance(ind1.member2.purified, ind2.member1.purified)
    b2 = utils.get_distance(ind1.member2.purified, ind2.member2.purified)

    a = np.minimum(a1, a2)
    b = np.minimum(b1, b2)
    c = np.minimum(a1, b1)
    d = np.minimum(a2, b2)

    dist = np.mean([a, b, c, d])
    return dist


def eval_archive_dist(ind1, ind2):

    if ind1.member1.predicted_label == EXPECTED_LABEL:
        ind1_correct = ind1.member1.purified
        ind1_misclass = ind1.member2.purified
    else:
        ind1_correct = ind1.member2.purified
        ind1_misclass = ind1.member1.purified

    if ind2.member1.predicted_label == EXPECTED_LABEL:
        ind2_correct = ind2.member1.purified
        ind2_misclass = ind2.member2.purified
    else:
        ind2_correct = ind2.member2.purified
        ind2_misclass = ind2.member1.purified

    dist1 = utils.get_distance(ind1_correct, ind2_correct)
    dist2 = utils.get_distance(ind1_misclass, ind2_misclass)

    dist = np.mean([dist1, dist2])
    return dist