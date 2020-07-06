import json
import os
from typing import Tuple, List, Callable

from core.folders import folders


class RoadStorage:

    def __init__(self, path: str = None):
        if path is None:
            path='test_driving'
        self.folder = str(folders.member_seeds.joinpath(path))
        os.makedirs(self.folder, exist_ok=True)

    def all_files(self) -> List[str]:
        expanded = [os.path.join(self.folder, filename) for filename in os.listdir(self.folder)]
        return [path for path in expanded if os.path.isfile(path)]

    def get_road_path_by_index(self, index) -> str:
        assert index > 0
        path = os.path.join(self.folder, 'road{:03}_nodes.json'.format(index))
        return path

    def get_road_nodes_by_index(self, index) -> List[Tuple[float, float, float, float]]:
        path = self.get_road_path_by_index(index)
        nodes = self.get_road_nodes(path)
        return nodes

    def get_road_nodes(self, path) -> List[Tuple[float, float, float, float]]:
        assert os.path.exists(path), path
        with open(path, 'r') as f:
            nodes = json.loads(f.read())
        return nodes

    def cache(self, road_name: str, get_points: Callable) -> List[Tuple[float, float, float, float]]:
        path = os.path.join(self.folder, road_name + '.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                nodes = json.loads(f.read())
        else:
            nodes = get_points()
            with open(path, 'w') as f:
                f.write(json.dumps(nodes))
        return nodes

    def save(self, road_name: str, contents: str) -> List[Tuple[float, float, float, float]]:
        path = os.path.join(self.folder, road_name + '.json')
        with open(path, 'w') as f:
            f.write(contents)

    def read(self, path) -> List[Tuple[float, float, float, float]]:
        assert os.path.exists(path), path
        with open(path, 'r') as f:
            beamng_member = json.loads(f.read())
        return beamng_member


if __name__ == '__main__':
    for i in range(1, 31):
        nodes = RoadStorage().get_road_nodes_by_index(i)
        print(i, len(nodes))
