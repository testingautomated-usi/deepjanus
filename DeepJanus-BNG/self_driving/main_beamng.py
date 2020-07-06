from core import nsga2
from core.archive_impl import SmartArchive
from self_driving.beamng_config import BeamNGConfig
import matplotlib.pyplot as plt

from self_driving.beamng_problem import BeamNGProblem

config = BeamNGConfig()

problem = BeamNGProblem(config, SmartArchive(config.ARCHIVE_THRESHOLD))

if __name__ == '__main__':
    nsga2.main(problem)
    print('done')

    plt.ioff()
    plt.show()
