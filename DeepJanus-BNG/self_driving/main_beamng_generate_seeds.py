import json

from core.archive_impl import SmartArchive
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_problem import BeamNGProblem
from core.config import Config
from core.folder_storage import SeedStorage

config_silly = BeamNGConfig()
config_silly.keras_model_file = 'self-driving-car-4600.h5'
config_silly.beamng_close_at_iteration = True

config_smart = BeamNGConfig()
config_smart.keras_model_file = 'self-driving-car-185-2020.h5'
config_smart.beamng_close_at_iteration = True

problem_silly = BeamNGProblem(config_silly, SmartArchive(config_silly.ARCHIVE_THRESHOLD))
problem_smart = BeamNGProblem(config_smart, SmartArchive(config_smart.ARCHIVE_THRESHOLD))

# problem = BeamNGProblem(config, SmartArchive(config.ARCHIVE_THRESHOLD))

if __name__ == '__main__':
    good_members_found = 0
    attempts = 0
    storage = SeedStorage('prova_roads')

    while good_members_found < 100:
        path = storage.get_path_by_index(good_members_found + 1)
        if path.exists():
            print('member already exists', path)
            good_members_found += 1
            continue
        attempts += 1
        print(f'attempts {attempts} good {good_members_found} looking for {path}')
        member = problem_silly.generate_random_member()
        member.evaluate()
        if member.distance_to_boundary <= 0:
            continue
        member_smart = problem_smart.member_class().from_dict(member.to_dict())
        member_smart.config = config_smart
        member_smart.problem = problem_smart
        member_smart.clear_evaluation()
        member_smart.evaluate()
        if member_smart.distance_to_boundary <= 0:
            continue
        member.distance_to_boundary = None
        good_members_found += 1
        path.write_text(json.dumps(member.to_dict()))
