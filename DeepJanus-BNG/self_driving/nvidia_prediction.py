import numpy as np

from core.config import Config
from self_driving.simulation_data import SimulationDataRecord
from udacity_integration.udacity_utils import preprocess


class NvidiaPrediction:
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.speed_limit = config.MAX_SPEED

    def predict(self, image, car_state: SimulationDataRecord):
        try:
            image = np.asarray(image)

            image = preprocess(image)
            image = np.array([image])

            steering_angle = float(self.model.predict(image, batch_size=1))

            speed = car_state.vel_kmh
            if speed > self.speed_limit:
                self.speed_limit = self.config.MIN_SPEED  # slow down
            else:
                self.speed_limit = self.config.MAX_SPEED
            throttle = 1.0 - steering_angle ** 2 - (speed / self.speed_limit) ** 2
            return steering_angle, throttle

        except Exception as e:
            print(e)