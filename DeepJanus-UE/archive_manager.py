import json
from os import makedirs
from os.path import exists

from distance_calculator import calc_distance_total
from evaluator import eval_archive_dist, is_misb
import numpy as np

from properties import ARCHIVE_THRESHOLD, RESULTS_PATH, POPSIZE, NGEN, \
    RESEEDUPPERBOUND, K_SD, MODEL, MISB_TSHD


class Archive:

    def __init__(self):
        self.archive = list()
        self.archived_seeds = set()

    def get_archive(self):
        return self.archive

    def update_archive(self, ind):
        if ind not in self.archive:
            if len(self.archive) == 0:
                self.archive.append(ind)
                self.archived_seeds.add(ind.seed)
            else:
                # Find the member of the archive that is closest to the candidate.
                closest_archived = None
                d_min = np.inf
                i = 0
                while i < len(self.archive):
                    distance_archived = eval_archive_dist(
                        ind,
                        self.archive[i])
                    if distance_archived < d_min:
                        closest_archived = self.archive[i]
                        d_min = distance_archived
                    i += 1
                # Decide whether to add the candidate to the archive
                # Verify whether the candidate is close to the existing member of the archive
                # Note: 'close' is defined according to a user-defined threshold
                if d_min <= ARCHIVE_THRESHOLD:
                    # The candidate replaces the closest archive member if its members' distance is better
                    dist_ind = ind.distance
                    dist_archived_ind = closest_archived.distance
                    if dist_ind <= dist_archived_ind:
                        self.archive.remove(closest_archived)
                        self.archive.append(ind)
                        self.archived_seeds.add(ind.seed)
                else:
                    # Add the candidate to the archive if it is distant from all the other archive members
                    self.archive.append(ind)
                    self.archived_seeds.add(ind.seed)

    def create_report(self, generation):
        # Retrieve the solutions belonging to the archive.
        solution = [ind for ind in self.archive]
        N = (len(solution))

        # Obtain misclassified member of an individual on the frontier.
        misclassified_inputs = []
        # Obtain correctly classified member of an individual on the frontier.
        correctly_classified_frontier = []
        for ind in solution:
            if ind.member1.is_misb:
                misclassified_member = ind.member1
                correct_member = ind.member2
            else:
                misclassified_member = ind.member2
                correct_member = ind.member1
            misclassified_inputs.append(misclassified_member)
            correctly_classified_frontier.append(correct_member)

        print("Final solution N is: " + str(N))

        stats = [None] * 4
        final_seeds = []
        if len(misclassified_inputs) > 0:
            final_seeds = self.get_seeds()
            stats = self.get_dist_members()

        report = {
            'archive_len': str(N),
            'covered_seeds': len(self.archived_seeds),
            'final seeds': str(len(final_seeds)),
            'min_members_dist':str(stats[0]),
            'max_members_dist': str(stats[1]),
            'avg_members_dist': str(stats[2]),
            'std_members_dist': str(stats[3]),
        }

        if not exists(RESULTS_PATH):
            makedirs(RESULTS_PATH)
        dst = RESULTS_PATH + '/report_'+str(generation)+'.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()

        config = {
            'popsize': str(POPSIZE),
            'generations': str(NGEN),
            'archive tshd': str(ARCHIVE_THRESHOLD),
            'misb_tshd': str(MISB_TSHD),
            'reseed': str(RESEEDUPPERBOUND),
            'K': str(K_SD),
            #'model': str(MODEL)
        }
        dst = RESULTS_PATH + '/config.json'
        config_string = json.dumps(config)

        file = open(dst, 'w')
        file.write(config_string)
        file.close()

    def get_seeds(self):
        seeds = set()
        for ind in self.get_archive():
            seeds.add(ind.seed)
        return seeds

    def get_dist_members(self):
        distances = list()
        stats = [None]*4

        for ind in self.get_archive():
            distances.append(ind.distance)

        if len(distances) > 0:
            stats[0] = np.min(distances)
            stats[1] = np.max(distances)
            stats[2] = np.mean(distances)
            stats[3] = np.std(distances)
        return stats
