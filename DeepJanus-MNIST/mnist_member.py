import json
from os import makedirs

from os.path import join, exists
import numpy as np

import rasterization_tools
from folder import Folder
from timer import Timer
from utils import get_distance

import matplotlib.pyplot as plt


class MnistMember:
    COUNT = 0

    def __init__(self, desc, label, seed):
        self.timestamp, self.elapsed_time = Timer.get_timestamps()
        self.id = MnistMember.COUNT
        self.seed = seed
        self.xml_desc = desc
        self.purified = rasterization_tools.rasterize_in_memory(self.xml_desc)
        self.expected_label = label
        self.predicted_label = None
        self.confidence = None
        self.correctly_classified = None
        self.attention = None
        MnistMember.COUNT += 1

    def clone(self):
        clone_digit = MnistMember(self.xml_desc, self.expected_label)
        return clone_digit

    def to_dict(self, ind_id):
        performance = self.confidence
        return {'id': str(self.id),
                'sol_id': str(ind_id),
                'seed': str(self.seed),
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'misbehaviour': str(not self.correctly_classified),
                'performance': str(performance),
                'timestamp': str(self.timestamp),
                'elapsed': str(self.elapsed_time),
        }

    def dump(self, filename, id):
        data = self.to_dict(id)
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_png(self, filename):
        plt.imsave(filename+'.png', self.purified.reshape(28, 28), cmap='gray', format='png')

    def save_npy(self, filename):
        np.save(filename, self.purified)
        test_img = np.load(filename+'.npy')
        diff = self.purified - test_img
        assert(np.linalg.norm(diff) == 0)

    def save_svg(self, filename):
        data = self.xml_desc
        filedest = filename + ".svg"
        with open(filedest, 'w') as f:
            f.write(data)

    def export(self, ind_id):
        if not exists(Folder.DST_ARC):
            makedirs(Folder.DST_ARC)
        dst = join(Folder.DST_ARC, "mbr"+str(self.id))
        self.dump(dst, ind_id)
        self.save_npy(dst)
        self.save_png(dst)
        self.save_svg(dst)

    def distance(self, other):
        return get_distance(self.purified, other.purified)
