import os
from datetime import datetime
from typing import List, Dict

from udacity_integration.beamng_car_cameras import BeamNGCarCameras
from beamngpy import Vehicle, BeamNGpy

from self_driving.decal_road import DecalRoad
from self_driving.oob_monitor import OutOfBoundsMonitor
from self_driving.road_polygon import RoadPolygon
from self_driving.vehicle_state_reader import VehicleStateReader

CSV_header = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
CSV_idx: Dict[str, int] = {k: v for v, k in enumerate(CSV_header)}


class TrainingDataCollectorAndWriter:

    def __init__(self, vehicle: Vehicle, beamng: BeamNGpy, road: DecalRoad, cameras: BeamNGCarCameras):

        self.vehicle_state_reader = VehicleStateReader(vehicle, beamng, additional_sensors=cameras.cameras_array)
        self.oob_monitor = OutOfBoundsMonitor(RoadPolygon.from_nodes(road.nodes), self.vehicle_state_reader)
        self.beamng = beamng
        self.road = road
        self.log_folder = 'training_recordings/' + datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        self.sequence_index = 0
        os.makedirs(self.log_folder, exist_ok=True)
        self.append_line(CSV_header)

    def collect_and_write_current_data(self):
        self.sequence_index += 1
        self.vehicle_state_reader.update_state()
        car_state = self.vehicle_state_reader.get_state()
        self.last_state = car_state

        sensors = self.vehicle_state_reader.sensors
        base_name = "{:02d}".format(1)

        values: List[any] = [None] * len(CSV_header)

        def save_image(cam_name):
            img = sensors[cam_name]['colour'].convert('RGB')
            filename = 'z{:05d}_{}_{}.jpg'.format(self.sequence_index, car_state.steering_input,
                                                  cam_name.replace('cam_', ''))
            filepath = os.path.join(self.log_folder,
                                    filename)
            img.save(filepath)
            return filename

        values[CSV_idx['center']] = save_image('cam_center')
        values[CSV_idx['left']] = save_image('cam_left')
        values[CSV_idx['right']] = save_image('cam_right')
        values[CSV_idx['steering']] = car_state.steering_input
        values[CSV_idx['throttle']] = car_state.throttle
        values[CSV_idx['brake']] = car_state.brake
        values[CSV_idx['speed']] = car_state.vel_kmh

        self.append_line(values)

    def append_line(self, values: List[any]):
        with open(os.path.join(self.log_folder, 'driving_log.csv'), 'a+') as f:
            f.write(','.join(str(v) for v in values) + '\n')
