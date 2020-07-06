import logging as log
import os
import time
import traceback
from typing import List, Tuple

from core.folder_storage import SeedStorage
from core.folders import folders
from core.log_setup import get_logger
from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_evaluator import BeamNGEvaluator
from self_driving.beamng_member import BeamNGMember
from self_driving.beamng_tig_maps import maps
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.nvidia_prediction import NvidiaPrediction
from self_driving.simulation_data import SimulationDataRecord, SimulationData
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader
from udacity_integration.beamng_car_cameras import BeamNGCarCameras

log = get_logger(__file__)

FloatDTuple = Tuple[float, float, float, float]


class BeamNGNvidiaOob(BeamNGEvaluator):
    def __init__(self, config: BeamNGConfig):
        self.config = config
        self.brewer: BeamNGBrewer = None
        self.model_file = str(folders.trained_models_colab.joinpath(config.keras_model_file))
        if not os.path.exists(self.model_file):
            raise Exception(f'File {self.model_file} does not exist!')
        self.model = None

    def evaluate(self, members: List[BeamNGMember]):
        for member in members:
            if not member.needs_evaluation():
                log.info(f'{member} is already evaluated. skipping')
                continue
            counter = 20
            attempt = 0
            while True:
                attempt += 1
                if attempt == counter:
                    raise Exception('Exhausted attempts')
                if attempt > 1:
                    log.info(f'RETRYING TO run simulation {attempt}')
                    self._close()
                else:
                    log.info(f'{member} BeamNG evaluation start')
                if attempt > 2:
                    time.sleep(5)
                sim = self._run_simulation(member.sample_nodes)
                if sim.info.success:
                    break

            member.distance_to_boundary = sim.min_oob_distance()
            log.info(f'{member} BeamNG evaluation completed')

    def _run_simulation(self, nodes) -> SimulationData:
        if not self.brewer:
            self.brewer = BeamNGBrewer()
            self.vehicle = self.brewer.setup_vehicle()
            self.camera = self.brewer.setup_scenario_camera()

        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))
        maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

        cameras = BeamNGCarCameras()
        vehicle_state_reader = VehicleStateReader(self.vehicle, beamng, additional_sensors=cameras.cameras_array)
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
        name = self.config.simulation_name.replace('$(id)', simulation_id)
        sim_data_collector = SimulationDataCollector(self.vehicle, beamng, brewer.decal_road, brewer.params,
                                                     vehicle_state_reader=vehicle_state_reader,
                                                     camera=self.camera,
                                                     simulation_name=name)

        sim_data_collector.get_simulation_data().start()
        try:
            brewer.bring_up()
            from keras.models import load_model
            if not self.model:
                self.model = load_model(self.model_file)
            predict = NvidiaPrediction(self.model, self.config)
            iterations_count = 1000
            idx = 0
            while True:
                idx += 1
                if idx >= iterations_count:
                    sim_data_collector.save()
                    raise Exception('Timeout simulation ', sim_data_collector.name)

                sim_data_collector.collect_current_data(oob_bb=False)
                last_state: SimulationDataRecord = sim_data_collector.states[-1]
                if points_distance(last_state.pos, waypoint_goal.position) < 6.0:
                    break

                if last_state.is_oob:
                    break
                img = vehicle_state_reader.sensors['cam_center']['colour'].convert('RGB')
                steering_angle, throttle = predict.predict(img, last_state)
                self.vehicle.control(throttle=throttle, steering=steering_angle, brake=0)
                beamng.step(steps)

            sim_data_collector.get_simulation_data().end(success=True)
        except Exception as ex:
            sim_data_collector.get_simulation_data().end(success=False, exception=ex)
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            if self.config.simulation_save:
                sim_data_collector.save()
                try:
                    sim_data_collector.take_car_picture_if_needed()
                except:
                    pass

            self.end_iteration()

        return sim_data_collector.simulation_data

    def end_iteration(self):
        try:
            if self.config.beamng_close_at_iteration:
                self._close()
            else:
                if self.brewer:
                    self.brewer.beamng.stop_scenario()
        except Exception as ex:
            log.debug('end_iteration() failed:')
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    def _close(self):
        if self.brewer:
            try:
                self.brewer.beamng.close()
            except Exception as ex:
                log.debug('beamng.close() failed:')
                traceback.print_exception(type(ex), ex, ex.__traceback__)
            self.brewer = None


if __name__ == '__main__':
    config = BeamNGConfig()
    inst = BeamNGNvidiaOob(config)
    while True:
        seed_storage = SeedStorage('basic5')
        for i in range(1, 11):
            member = BeamNGMember.from_dict(seed_storage.load_json_by_index(i))
            member.clear_evaluation()
            inst.evaluate([member])
            log.info(member)
