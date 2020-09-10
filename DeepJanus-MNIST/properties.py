# GA Setup
POPSIZE = 800
NGEN = 40000

RUNTIME = 3600
INTERVAL = 900

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
ARCHIVE_THRESHOLD = 4.0

# Dataset
EXPECTED_LABEL = 5

#------- NOT TUNING ----------

# mutation operator probability
MUTPB = 0.7
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10


INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False

# Directories
PATH = "vectorized_images_GA"
TESTSOURCEPATH = "source_images_GA"
TRAINSOURCEPATH = "source_images_trainset"

MODEL = 'models/model_mnist.h5'
#MODEL = 'models/cnnClassifier.h5'

ORIGINAL_SEEDS = "bootstraps_five"
RESULTS_PATH = 'results'

BITMAP_THRESHOLD = 0.5