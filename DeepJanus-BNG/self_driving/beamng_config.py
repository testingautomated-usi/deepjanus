# this class must have simple fields in order to be serialized
from core.config import Config


# this class must have simple fields in order to be serialized
class BeamNGConfig(Config):
    EVALUATOR_FAKE = 'EVALUATOR_FAKE'
    EVALUATOR_LOCAL_BEAMNG = 'EVALUATOR_LOCAL_BEAMNG'
    EVALUATOR_REMOTE_BEAMNG = 'EVALUATOR_REMOTE_BEAMNG'

    def __init__(self):
        super().__init__()

        self.num_control_nodes = 10

        self.MIN_SPEED = 10
        self.MAX_SPEED = 25

        self.beamng_close_at_iteration = False
        self.beamng_evaluator = self.EVALUATOR_LOCAL_BEAMNG
