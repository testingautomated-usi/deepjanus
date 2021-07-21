import csv
import json
from os.path import join

from folder import Folder
from timer import Timer
from utils import get_distance
from evaluator import eval_archive_dist
import numpy as np

from config import ARCHIVE_THRESHOLD, POPSIZE, NGEN, MUTLOWERBOUND, MUTUPPERBOUND, \
    RESEEDUPPERBOUND, K_SD, MODEL, EXPLABEL, STOP_CONDITION, RUNTIME, REPORT_NAME, STEPSIZE
from metrics import get_diameter, get_radius_reference


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
                    distance_archived = eval_archive_dist(ind, self.archive[i])
                    if distance_archived < d_min:
                        closest_archived = self.archive[i]
                        d_min = distance_archived
                    i += 1
                # Decide whether to add the candidate to the archive
                # Verify whether the candidate is close to the existing member of the archive
                # Note: 'close' is defined according to a user-defined threshold
                if d_min <= ARCHIVE_THRESHOLD:
                    # The candidate replaces the closest archive member if its members' distance is better
                    dist_ind = ind.members_distance
                    dist_archived_ind = get_distance(closest_archived.m1.purified,
                                                     closest_archived.m2.purified)
                    if dist_ind <= dist_archived_ind:
                        self.archive.remove(closest_archived)
                        self.archive.append(ind)
                        self.archived_seeds.add(ind.seed)
                else:
                    # Add the candidate to the archive if it is distant from all the other archive members
                    self.archive.append(ind)
                    self.archived_seeds.add(ind.seed)

    def get_min_distance_from_archive(self, seed):
        distances = list()
        for archived_ind in self.archive:
            dist_member1 = np.linalg.norm(archived_ind.m1.purified - seed)
            dist_member2 = np.linalg.norm(archived_ind.m2.purified - seed)
            avg_dist = (dist_member1 + dist_member2) / 2
            distances.append(avg_dist)
        min_dist = min(distances)
        return min_dist

    def create_report(self, x_test, seeds, generation):
        # Retrieve the solutions belonging to the archive.
        if generation == STEPSIZE:
            dst = join(Folder.DST, REPORT_NAME)
            with open(dst, mode='w') as report_file:
                report_writer = csv.writer(report_file,
                                           delimiter=',',
                                           quotechar='"',
                                           quoting=csv.QUOTE_MINIMAL)

                report_writer.writerow(["run",
                                        'iteration',
                                        'timestamp',
                                        'archive_len',
                                        'sparseness',
                                        'total_seeds',
                                        'covered_seeds',
                                        'final seeds',
                                        'members_dist_min',
                                        'members_dist_max',
                                        'members_dist_avg',
                                        'members_dist_std',
                                        'radius_ref_out',
                                        'radius_ref_in',
                                        'diameter_out',
                                        'diameter_in',
                                        'iteration'])
        solution = [ind for ind in self.archive]
        n = len(solution)

        # Obtain misclassified member of an individual on the frontier.
        outer_frontier = []
        # Obtain correctly classified member of an individual on the frontier.
        inner_frontier = []
        for ind in solution:
            if ind.m1.predicted_label != ind.m1.expected_label:
                misclassified_member = ind.m1
                correct_member = ind.m2
            else:
                misclassified_member = ind.m2
                correct_member = ind.m1
            outer_frontier.append(misclassified_member)
            inner_frontier.append(correct_member)

        avg_sparseness = 0
        if n > 1:
            # Calculate sparseness of the solutions
            sumsparseness = 0

            for dig1 in outer_frontier:
                sumdistances = 0
                for dig2 in outer_frontier:
                    if dig1 != dig2:
                        sumdistances += np.linalg.norm(dig1.purified - dig2.purified)
                dig1.sparseness = sumdistances / (n - 1)
                sumsparseness += dig1.sparseness
            avg_sparseness = sumsparseness / n

        out_radius = None
        in_radius = None
        out_radius_ref = None
        in_radius_ref = None
        out_diameter = None
        in_diameter = None
        stats = [None] * 4
        final_seeds = []
        if len(solution) > 0:
            reference_filename = 'ref_digit/cinque_rp.npy'
            reference = np.load(reference_filename)
            out_diameter = get_diameter(outer_frontier)
            in_diameter = get_diameter(inner_frontier)
            final_seeds = self.get_seeds()
            stats = self.get_dist_members()
            out_radius_ref = get_radius_reference(outer_frontier, reference)
            in_radius_ref = get_radius_reference(inner_frontier, reference)

        if STOP_CONDITION == "iter":
            budget = NGEN
        elif STOP_CONDITION == "time":
            budget = RUNTIME
        else:
            budget = "no budget"
        config = {
            'popsize': str(POPSIZE),
            'budget': str(budget),
            'budget_type': str(STOP_CONDITION),
            # TODO: unbound ind
            # 'label': str(ind.member1.expected_label),
            'label': str(EXPLABEL),
            'archive tshd': str(ARCHIVE_THRESHOLD),
            'mut low': str(MUTLOWERBOUND),
            'mut up': str(MUTUPPERBOUND),
            'reseed': str(RESEEDUPPERBOUND),
            'K': str(K_SD),
            'model': str(MODEL),
        }

        dst = join(Folder.DST, "config.json")

        # dst = RESULTS_PATH + '/config.json'
        with open(dst, 'w') as f:
            (json.dump(config, f, sort_keys=True, indent=4))

        dst = join(Folder.DST, REPORT_NAME)
        with open(dst, mode='a') as report_file:
            report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            timestamp, elapsed_time = Timer.get_timestamps()
            report_writer.writerow([Folder.run_id,
                                    str(generation),
                                    str(elapsed_time),
                                    str(n),
                                    str(avg_sparseness),
                                    str(len(seeds)),
                                    str(len(self.archived_seeds)),
                                    str(len(final_seeds)),
                                    str(stats[0]),
                                    str(stats[1]),
                                    str(stats[2]),
                                    str(stats[3]),
                                    str(out_radius_ref),
                                    str(in_radius_ref),
                                    str(out_radius),
                                    str(in_radius),
                                    str(out_diameter),
                                    str(in_diameter),
                                    str(generation)])

    def get_seeds(self):
        seeds = set()
        for ind in self.get_archive():
            seeds.add(ind.seed)
        return seeds

    def get_dist_members(self):
        distances = list()
        stats = [None] * 4
        for ind in self.get_archive():
            distances.append(ind.members_distance)

        stats[0] = np.min(distances)
        stats[1] = np.max(distances)
        stats[2] = np.mean(distances)
        stats[3] = np.std(distances)
        return stats
