# GA Setup
POPSIZE = 100
NGEN = 4000

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

#------- NOT TUNING ----------

# mutation operator probability
MUTOPPROB = 0.5
MUTOFPROB = 0.5

IMG_SIZE = 28
num_classes = 10


INITIALPOP = 'seeded'

GENERATE_ONE_ONLY = False

MODEL = 'models/cnnClassifier_lowLR.h5'
MODEL2 = 'models/cnnClassifier.h5'

RESULTS_PATH = 'results'
DATASET = 'original_dataset/janus_dataset.h5'
EXPLABEL = 5