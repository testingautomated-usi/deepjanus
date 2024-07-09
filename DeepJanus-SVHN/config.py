DJ_DEBUG = 1

# GA Setup
POPSIZE = 200

STOP_CONDITION = "iter"
#STOP_CONDITION = "time"

NGEN = 2000
#NGEN = 100
RUNTIME = 3600
STEPSIZE = 10
# Mutation Hyperparameters
# range of the mutation
MUTLOWERBOUND = 0.01
MUTUPPERBOUND = 0.6

# Reseeding Hyperparameters
# extent of the reseeding operator
RESEEDUPPERBOUND = 10

K_SD = 0.1

# K-nearest
K = 1

# Archive configuration
#ARCHIVE_THRESHOLD = 4.0
ARCHIVE_THRESHOLD = 1.5

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 32
IMG_CHN = 3
num_classes = 10


INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False

MODEL2 = 'models/cnnClassifier_lowLR.h5'
MODEL = 'models/cnnClassifier.h5'
#MODEL = "models/regular3"
#MODEL = 'models/cnnClassifier_001.h5'
#MODEL = 'models/cnnClassifier_op.h5'

RESULTS_PATH = 'results'
REPORT_NAME = 'stats.csv'
DATASET = 'original_dataset/janus_dataset_comparison.h5'
EXPLABEL = 5

#TODO: set interpreter
INTERPRETER = '/home/vin/yes/envs/tf_gpu/bin/python'