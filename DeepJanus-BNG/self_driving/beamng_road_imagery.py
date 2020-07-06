import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

from core.folder_storage import SeedStorage
from self_driving.beamng_member import BeamNGMember
from self_driving.road_points import RoadPoints


class BeamNGRoadImagery:
    def __init__(self, road_points: RoadPoints):
        self.road_points = road_points
        self._fig, self._ax = None, None

    def plot(self):
        self._close()
        self._fig, self._ax = plt.subplots(1)
        self.road_points.plot_on_ax(self._ax)
        self._ax.axis('equal')

    def save(self, image_path):
        if not self._fig:
            self.plot()
        self._fig.savefig(image_path)

    @classmethod
    def from_sample_nodes(cls, sample_nodes):
        return BeamNGRoadImagery(RoadPoints().add_middle_nodes(sample_nodes))

    def _close(self):
        if self._fig:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def __del__(self):
        self._close()


def main():
    storage = SeedStorage('short')
    for i in range(1, 100):
        member_filepath = storage.get_path_by_index(i)
        if not member_filepath.exists():
            continue

        member = BeamNGMember.from_dict(storage.load_json_by_index(i))
        sample_nodes = member.sample_nodes
        road_imagery = BeamNGRoadImagery.from_sample_nodes(sample_nodes)
        for extension in ['.jpg', '.svg']:
            image_filename = member_filepath.with_suffix(extension)
            print('saving', image_filename)
            road_imagery.save(image_filename)


if __name__ == '__main__':
    main()
