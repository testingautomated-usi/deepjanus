import json

from core.config import Config
from core.folder_storage import SeedStorage
from self_driving.road_generator import RoadGenerator

if __name__ == "__main__":
    config = Config()
    seed_storage = SeedStorage(config.seed_folder)
    for i in range(1, 4):
        path = seed_storage.get_path_by_index(i)
        if path.exists():
            print('file ', path, 'already exists')
        else:
            obj = RoadGenerator(
                num_control_nodes=config.num_control_nodes,
                seg_length=config.SEG_LENGTH).generate()
            print('saving', path)
            path.write_text(json.dumps(obj.to_dict()))
